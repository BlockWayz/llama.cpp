# llama.cpp — Agens Volundr branch

This branch adds support for **Agens Volundr 32B** (Blockway) to llama.cpp: a new model architecture `volundr`,
its GGUF conversion, and the two extra ggml ops it needs. The GGUF files are on Hugging Face:
**[Blockway/Agens-Volundr-32B-Preview-GGUF](https://huggingface.co/Blockway/Agens-Volundr-32B-Preview-GGUF)**
(model card and benchmarks: [Blockway/Agens-Volundr-32B-Preview](https://huggingface.co/Blockway/Agens-Volundr-32B-Preview)).

Stock llama.cpp refuses these files with `unknown model architecture: 'volundr'`; build this branch instead.

## What the branch adds

Volundr is a 72-layer hybrid: 54 KDA linear-attention layers, 17 BCSA (block-compressed sparse attention) layers,
one dense gated-attention layer, Engram hashed n-gram memory on two layers, and 4-stream mHC hyper-connections.

| area | files |
|---|---|
| model graph (KDA, BCSA, dense attention, Engram, mHC) | `src/models/volundr.cpp`, `src/llama-model.cpp`, `src/llama-arch.*`, `src/llama-hparams.*` |
| BCSA masks and indexer-key side store in the KV cache | `src/llama-kv-cache.*`, `src/llama-context.cpp` |
| new ops `GGML_OP_NGRAM_HASH` (Engram) and `GGML_OP_SINKHORN` (mHC), CPU and CUDA kernels | `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml/src/ggml-cpu/*`, `ggml/src/ggml-cuda/volundr.cu*`, `tests/test-backend-ops.cpp` |
| HF → GGUF conversion (text model) | `conversion/volundr.py`, `gguf-py/gguf/constants.py` |
| chat template: `<|call|>` is not end-of-generation when the vocab also has `<|/call|>`; `<|agens_end|>` ends a turn | `src/llama-vocab.cpp` |
| unused-token mask (see below) | `common/sampling.cpp` |
| parity / debugging tools `llama-volundr-dump`, `llama-volundr-chat-test` | `examples/volundr-dump/` |

The branch starts from llama.cpp commit `cf67f0d24` (July 2026). Everything else is upstream llama.cpp, unchanged.

## Build

CPU:

```bash
git clone --branch volundr https://github.com/BlockWayz/llama.cpp && cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server llama-cli
```

CUDA (tested on sm_89 with CUDA 12.9):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j --target llama-server llama-cli
```

Add `llama-quantize llama-perplexity llama-bench` to `--target` if you need them. If CMake cannot find libcurl, add
`-DLLAMA_CURL=OFF` and download the GGUF yourself.

## Run

```bash
# download one file (Q4_K_M, 26.6 GB)
hf download Blockway/Agens-Volundr-32B-Preview-GGUF Agens-Volundr-32B-Preview-Q4_K_M.gguf --local-dir .   # pip install -U huggingface_hub

# OpenAI-compatible server: thinking, tool calls and reasoning_content parsing work out of the box
./build/bin/llama-server -m Agens-Volundr-32B-Preview-Q4_K_M.gguf --jinja -c 32768 -ngl 99

# interactive chat
./build/bin/llama-cli -m Agens-Volundr-32B-Preview-Q4_K_M.gguf --jinja -c 32768 -ngl 99
```

Per request: `"chat_template_kwargs": {"enable_thinking": false}` turns thinking off;
`"chat_template_kwargs": {"reasoning_effort": "low"}` shortens it (`low` | `medium` | `xhigh`, default `xhigh`).
Sampling defaults stored in the GGUF: temperature 1.0, top-k 20, top-p 0.95.

When offloading to a GPU, the four Engram tables (4.3 GB, pure lookups) can stay in host RAM:
`--override-tensor "engram_embd=CPU"`.

## Unused-token mask

The output layer has 248,320 rows but the tokenizer ends at id 248,076. The trailing ids 248,077–248,319 (type UNUSED
in the GGUF) are never valid output, yet the model can put real probability on some of them; at the position where a
tool call opens, sampling sometimes picks one instead of `<|call|>` and the call comes back as plain text. For a
`volundr` model this branch therefore adds a -inf logit bias on that range to every sampler it builds, in `llama-server`
and `llama-cli` alike (the reference serving stack does the same, and the HF checkpoint lists the ids in
`generation_config.json` `suppress_tokens`). In our test with the Q4_K_M file on one GPU, sampled tool calls parsed:

| | with mask | without mask |
|---|---:|---:|
| no thinking, temperature 0.6 | 16/16 | 7/16 |
| no thinking, temperature 1.0 (default) | 16/16 | 10/16 |
| thinking, temperature 1.0 | 8/8 | 3/8 |

The mask costs nothing measurable: decode stays at ~32 tok/s with a 1.1K-token prompt on one 48 GB GPU.

`VOLUNDR_MASK_UNUSED_TOKENS=0` turns the mask off (e.g. to inspect raw distributions).

## Environment variables

| variable | default | effect |
|---|---|---|
| `VOLUNDR_MASK_UNUSED_TOKENS` | `1` | `0` disables the unused-token mask |
| `VOLUNDR_BCSA_SPARSE` | auto | decode uses the sparse BCSA path above 12,288 cached positions; `0` / `1` forces it off / on |
| `VOLUNDR_BCSA_QCHUNK` | `64` | BCSA query chunk size during prompt processing (bounds memory) |

## Known limits

* Text only: the vision tower is not converted.
* No speculative decoding.
* The two Volundr-specific ops have CPU and CUDA kernels only; on Metal, Vulkan or ROCm the scheduler runs those two
  ops on the CPU.
* BCSA prompt processing scores every query against every cached position before pooling and selection, so long
  prompts get slower roughly quadratically (32K tokens: ~620 tok/s on one 48 GB GPU).
* Recurrent state (KDA, Engram): no context shift and no partial KV removal, as for llama.cpp's other hybrid models.
* `--split-mode layer` works across GPUs but gives no speed-up over one 48 GB GPU; `--split-mode row` does not load.

## Converting a checkpoint yourself

The published files were made with these commands (build the `llama-quantize` target first):

```bash
pip install -r requirements.txt
python convert_hf_to_gguf.py /path/to/Agens-Volundr-32B-Preview --outtype bf16 \
  --model-name Agens-Volundr-32B-Preview --outfile Agens-Volundr-32B-Preview-BF16.gguf

./build/bin/llama-quantize --tensor-type "engram_embd=bf16" \
  Agens-Volundr-32B-Preview-BF16.gguf Agens-Volundr-32B-Preview-Q8_0.gguf Q8_0

./build/bin/llama-quantize --output-tensor-type q8_0 --token-embedding-type q8_0 --tensor-type "engram_embd=bf16" \
  --tensor-type "attn_qkv=q8_0" --tensor-type "attn_gate=q8_0" --tensor-type "ssm_out=q8_0" \
  --tensor-type "ssm_beta=q8_0" --tensor-type "ssm_f_a=q8_0" --tensor-type "ssm_f_b=q8_0" \
  --tensor-type "engram_value=q8_0" --tensor-type "engram_gate=q8_0" --tensor-type "hc_dyn=q8_0" \
  Agens-Volundr-32B-Preview-BF16.gguf Agens-Volundr-32B-Preview-Q4_K_M.gguf Q4_K_M
```

The Q4_K_M recipe keeps every KDA projection, the Engram value/gate projections, the mHC dynamic projections, the
token embedding and the output layer at Q8_0; the Engram tables stay BF16 and llama-quantize never quantises the BCSA
indexer (BF16). Only the text model is converted.

## Licence

Code: MIT, as upstream llama.cpp (see `LICENSE`). Model weights: Apache-2.0, see the Hugging Face repository.
