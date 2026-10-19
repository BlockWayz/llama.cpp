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
| parity / debugging tools `llama-volundr-dump`, `llama-volundr-chat-test`, `llama-volundr-state-test`, `slot_cache_test.py` | `examples/volundr-dump/` |

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

## Multi-turn prompt cache

`llama-server` reuses the cached state when a request extends an earlier one (the next turn of a chat), so only the new
tokens are processed. This works with the default settings; nothing Volundr-specific has to be enabled.

* The BCSA/dense KV cache (including the BCSA indexer keys) is truncated in place to the common prefix.
* The recurrent state (KDA state, conv state and the two previous token ids that Engram hashes at the boundary) cannot
  be rolled back, so the server keeps *context checkpoints* of it, taken during prompt processing at user-message
  starts and 4 and 4 + n_ubatch tokens before the end of each prompt. When the new prompt diverges from the cached one
  (e.g. the chat template drops the previous turn's reasoning), the server restores the nearest checkpoint before the
  divergence and processes from there. If no checkpoint fits, it falls back to a full prompt reprocess.
* Each checkpoint holds 168 MiB of recurrent state in host RAM.

Turn-2 time to first token, Q4_K_M on one 48 GB GPU, `-np 1`, chat completions, greedy:

| history | cold (full prefill) | warm (cache reuse) | tokens processed when warm |
|---:|---:|---:|---:|
| 8K, no thinking | 6.1 s | 0.4-0.6 s | 37 |
| 30K, no thinking | 46.5 s | 0.5 s | 37 |
| 8K, thinking (reasoning dropped from history: checkpoint restore) | 5.9 s | 0.6-0.7 s | 215-238 |
| 30K, thinking (checkpoint restore) | 46.8 s | 1.4 s | 358 |

The restored state is exact: restoring a checkpoint, or a full saved sequence state, and processing the same tokens
again gives bit-identical logits (`llama-volundr-state-test`, below). A warm turn is not bit-identical to a cold
prefill of the same prompt because the prompt is split into different batches. That is the same rounding noise as
changing `-ub`: in our runs, greedy output of warm and cold matched for 300/300 tokens when no checkpoint was needed.
With a checkpoint restore, they matched until the first near-tie (69-214 tokens; two cold runs at `-ub 512` and
`-ub 256` matched for 123). The max logprob difference of the sampled token was 0.04-0.12, against 0.06 between the two
cold runs.

Relevant flags (defaults are fine for a single user):

| flag | default | note |
|---|---|---|
| `--ctx-checkpoints N` | 32 | checkpoints kept per slot; at 168 MiB each, up to 5.3 GiB of host RAM per slot. A multi-turn chat only needs the last few; `--ctx-checkpoints 8` caps it at 1.3 GiB per slot |
| `--checkpoint-min-step N` | 8192 | minimum spacing of the mid-prompt checkpoints; the ones near the end of a prompt and at the last user message are always taken |
| `--cache-ram N` | 8192 | host prompt cache (MiB) that keeps idle conversations, with their checkpoints, when a slot switches to another conversation. A 30K-token conversation takes ~2.5 GiB plus its checkpoints, so raise this for many concurrent conversations |
| `-np N` | auto | one slot per concurrent conversation avoids swapping through the host prompt cache |

`--cache-reuse` (KV shifting) is not supported for this model, and `/slots` save/restore files do not include
checkpoints. A restored slot is therefore reused only when the next prompt extends the saved tokens exactly.

To check a build: `examples/volundr-dump/slot_cache_test.py <port> <history tokens> [--think] [--interleave]` runs
turn 1, then turn 2 warm and cold against a running `llama-server` and compares the outputs.
`llama-volundr-state-test -m model.gguf -f some.txt -ngl 99` checks state save/restore directly, including a negative
control that zeroes the Engram previous ids.

## Environment variables

| variable | default | effect |
|---|---|---|
| `VOLUNDR_MASK_UNUSED_TOKENS` | `1` | `0` disables the unused-token mask |
| `VOLUNDR_BCSA_SPARSE` | auto | decode uses the sparse BCSA path above 12,288 cached positions; `0` / `1` forces it off / on |
| `VOLUNDR_BCSA_QCHUNK` | `64` | BCSA query chunk size during prompt processing (bounds memory) |

## Known limits

* Text only: the vision tower is not converted.
* No speculative decoding.
* The two Volundr-specific ops have CPU and CUDA kernels only; on Metal, Vulkan or ROCm the scheduler should run those
  two ops on the CPU, but those backends have not been tested.
* BCSA prompt processing scores every query against every cached position before pooling and selection, so long
  prompts get slower roughly quadratically (32K tokens: ~620 tok/s on one 48 GB GPU).
* Recurrent state (KDA, Engram): no context shift and no arbitrary partial KV removal, as for llama.cpp's other
  hybrid models. Multi-turn prompt reuse works through context checkpoints (see "Multi-turn prompt cache" above).
* `--split-mode layer` works across GPUs but gives no speed-up over one 48 GB GPU. `--split-mode tensor` is not
  supported for this architecture and is refused at load; `--split-mode row` no longer exists for CUDA upstream.

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
