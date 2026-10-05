"""Agens Volundr (Blockway) -> GGUF, text model only (the vision tower is skipped).

Ground truth for every transform below is the HF remote code shipped with the checkpoint
(modeling_volundr.py, bcsa.py, volundr_extras.py):

  * Qwen3_5RMSNorm is Gemma-style (x_hat * (1 + w)) -> +1 is baked into input/post-attention/final norms, q/k norms and the
    BCSA indexer norm. The KDA gated norm, the Engram norm and the mHC norm are plain (x_hat * w) -> untouched.
  * KDA uses grouped value attention (16 key heads, 48 value heads, head h_v reads key head h_v // 3, FLA's
    repeat_interleave). ggml broadcasts "tiled" (h_v % 16), so every per-value-head tensor is reordered here
    (v rows of in_proj_qkv, conv1d v channels, in_proj_z, in_proj_b, f_up rows, A_log, dt_bias, out_proj columns).
  * A_log -> -exp(A_log) (the graph multiplies softplus(...) by it).
  * mHC static parameters (pre[n], post[n], res[n, n] row-major) are packed into one hc_base[2n + n^2] vector, which is
    the layout of the mHC dyn projection output.
"""
from __future__ import annotations

from typing import Callable, Iterable

import torch
from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger

# volundr_extras._PRIMES (the Engram n-gram hash multipliers)
_ENGRAM_PRIMES = (1000003, 1000033, 1000037, 1000039, 1000081)


@ModelBase.register("VolundrForConditionalGeneration", "VolundrTextModel")
class VolundrModel(TextModel):
    model_arch = gguf.MODEL_ARCH.VOLUNDR

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hc_static: dict[int, dict[str, Tensor]] = {}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _reorder_v_heads(tensor: Tensor, dim: int, num_k_heads: int, num_v_per_k: int, head_dim: int) -> Tensor:
        """grouped [K0: v0..v{r-1}, K1: ...] -> tiled [v0 of K0, v0 of K1, ..., v1 of K0, ...] along `dim`."""
        shape = list(tensor.shape)
        if dim < 0:
            dim += len(shape)
        new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1:]
        tensor = tensor.reshape(*new_shape)
        perm = list(range(len(new_shape)))
        perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
        return tensor.permute(*perm).contiguous().reshape(*shape)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if name.startswith(("model.visual.", "visual.", "model.merger.", "merger.")) or ".visual." in name:
            return None
        if name.startswith("mtp") or ".mtp." in name:
            return None
        return super().filter_tensors(item)

    # ------------------------------------------------------------------ vocab / hparams
    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams
        arch = gguf.MODEL_ARCH_NAMES[self.model_arch]
        w = self.gguf_writer

        mixers: list[str] = hp["mixer_types"]
        n_layer = hp["num_hidden_layers"]
        assert len(mixers) == n_layer, (len(mixers), n_layer)
        assert set(mixers) <= {"kda", "bcsa", "attn"}, set(mixers)

        head_dim = hp["head_dim"]
        rope = hp.get("rope_parameters") or {}
        partial = rope.get("partial_rotary_factor", hp.get("partial_rotary_factor", 1.0))
        w.add_rope_dimension_count(int(head_dim * partial))

        # KDA (Qwen3.5 GDN-style dims)
        assert hp["linear_key_head_dim"] == hp["linear_value_head_dim"]
        w.add_ssm_conv_kernel(hp["linear_conv_kernel_dim"])
        w.add_ssm_state_size(hp["linear_key_head_dim"])
        w.add_ssm_group_count(hp["linear_num_key_heads"])
        w.add_ssm_time_step_rank(hp["linear_num_value_heads"])
        w.add_ssm_inner_size(hp["linear_value_head_dim"] * hp["linear_num_value_heads"])
        w.add_uint32(f"{arch}.kda.gate_rank", hp["kda_gate_rank"])
        w.add_array(f"{arch}.attention.recurrent_layers", [m == "kda" for m in mixers])

        # BCSA (indexer: 4 heads x 128, see BCSAAttention defaults; checked against the tensor shapes below)
        w.add_array(f"{arch}.bcsa.layers", [m == "bcsa" for m in mixers])
        w.add_uint32(f"{arch}.bcsa.window", hp["bcsa_window"])
        w.add_uint32(f"{arch}.bcsa.compress", hp["bcsa_compress"])
        w.add_uint32(f"{arch}.attention.indexer.head_count", 4)
        w.add_uint32(f"{arch}.attention.indexer.key_length", 128)
        w.add_uint32(f"{arch}.attention.indexer.top_k", hp["bcsa_topk"])

        # mHC
        w.add_uint32(f"{arch}.hyper_connection.count", hp.get("mhc_streams", 1))
        w.add_uint32(f"{arch}.hyper_connection.sinkhorn_iterations", hp.get("mhc_sinkhorn_iters", 10))
        w.add_float32(f"{arch}.hyper_connection.epsilon", 1e-9)  # clamp_min of the Sinkhorn row/column sums

        # Engram
        engram_layers = hp.get("engram_layers") or []
        if engram_layers:
            for il in engram_layers:
                assert mixers[il] == "kda", f"Engram layer {il} must be a KDA layer (state lives in its recurrent row)"
            w.add_array(f"{arch}.engram.layers", [il in engram_layers for il in range(n_layer)])
            w.add_array(f"{arch}.engram.orders", [int(o) for o in hp["engram_orders"]])
            w.add_array(f"{arch}.engram.primes", list(_ENGRAM_PRIMES))
            w.add_uint32(f"{arch}.engram.rows", hp["engram_rows"])
            w.add_uint32(f"{arch}.engram.dim", hp["engram_dim"])

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # Engram tables are pure lookups: keep them 16-bit (BF16, the training dtype) whatever the output type
        if ".engram_embd." in new_name:
            return gguf.GGMLQuantizationType.BF16
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    # ------------------------------------------------------------------ tensors
    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        hp = self.hparams
        T = gguf.MODEL_TENSOR
        fmt = self.format_tensor_name

        if name == "model.embed_tokens.weight":
            yield fmt(T.TOKEN_EMBD), data_torch
            return
        if name == "lm_head.weight":
            yield fmt(T.OUTPUT), data_torch
            return
        if name == "model.norm.weight":
            yield fmt(T.OUTPUT_NORM), data_torch + 1
            return

        # ---- mHC (model.mhc.N.*)
        if name.startswith("model.mhc."):
            assert bid is not None
            leaf = name.split(".", 3)[3]
            if leaf == "dyn.weight":
                yield fmt(T.HC_DYN, bid), data_torch
            elif leaf == "norm.weight":
                yield fmt(T.HC_NORM, bid), data_torch                     # plain nn.RMSNorm
            elif leaf in ("pre_static", "post_static", "res_static"):
                st = self._hc_static.setdefault(bid, {})
                st[leaf] = data_torch.float()
                if len(st) == 3:
                    base = torch.cat([st["pre_static"].reshape(-1), st["post_static"].reshape(-1), st["res_static"].reshape(-1)])
                    del self._hc_static[bid]
                    yield fmt(T.HC_BASE, bid), base
            else:
                raise ValueError(f"unexpected mHC tensor {name}")
            return

        if not name.startswith("model.layers."):
            raise ValueError(f"unexpected tensor {name}")
        assert bid is not None
        leaf = name.split(".", 3)[3]   # e.g. "linear_attn.in_proj_qkv.weight"

        # ---- norms / MLP
        if leaf == "input_layernorm.weight":
            yield fmt(T.ATTN_NORM, bid), data_torch + 1
            return
        if leaf == "post_attention_layernorm.weight":
            yield fmt(T.ATTN_POST_NORM, bid), data_torch + 1
            return
        if leaf.startswith("mlp."):
            key = {"mlp.gate_proj.weight": T.FFN_GATE, "mlp.up_proj.weight": T.FFN_UP, "mlp.down_proj.weight": T.FFN_DOWN}[leaf]
            yield fmt(key, bid), data_torch
            return

        # ---- KDA
        if leaf.startswith("linear_attn."):
            nk = hp["linear_num_key_heads"]
            nv = hp["linear_num_value_heads"]
            hk = hp["linear_key_head_dim"]
            hv = hp["linear_value_head_dim"]
            r = nv // nk
            rv = self._reorder_v_heads
            sub = leaf[len("linear_attn."):]
            if sub == "in_proj_qkv.weight":
                q, k, v = torch.split(data_torch, [nk * hk, nk * hk, nv * hv], dim=0)
                yield fmt(T.ATTN_QKV, bid), torch.cat([q, k, rv(v, 0, nk, r, hv)], dim=0)
            elif sub == "in_proj_z.weight":
                yield fmt(T.ATTN_GATE, bid), rv(data_torch, 0, nk, r, hv)
            elif sub == "in_proj_b.weight":
                yield fmt(T.SSM_BETA, bid), rv(data_torch, 0, nk, r, 1)
            elif sub == "f_down.weight":
                assert data_torch.shape[0] == hp["kda_gate_rank"], data_torch.shape
                yield fmt(T.SSM_F_A, bid), data_torch
            elif sub == "f_up.weight":                 # [nv*hk, rank], rows are (value head, key channel)
                yield fmt(T.SSM_F_B, bid), rv(data_torch, 0, nk, r, hk)
            elif sub == "conv1d.weight":               # [conv_dim, 1, d_conv] -> [conv_dim, d_conv]
                c = data_torch.squeeze(1)
                qk, v = torch.split(c, [2 * nk * hk, nv * hv], dim=0)
                yield fmt(T.SSM_CONV1D, bid), torch.cat([qk, rv(v, 0, nk, r, hv)], dim=0)
            elif sub == "A_log":
                a = -torch.exp(data_torch.float())
                yield fmt(T.SSM_A, bid, suffix=""), rv(a.unsqueeze(-1), 0, nk, r, 1).squeeze(-1)
            elif sub == "dt_bias":                     # [nv*hk]
                yield fmt(T.SSM_DT, bid, suffix=".bias"), rv(data_torch.float().unsqueeze(-1), 0, nk, r, hk).squeeze(-1)
            elif sub == "norm.weight":                 # Qwen3_5RMSNormGated: plain weight
                yield fmt(T.SSM_NORM, bid), data_torch
            elif sub == "out_proj.weight":             # [n_embd, nv*hv] -> reorder columns
                yield fmt(T.SSM_OUT, bid), rv(data_torch, 1, nk, r, hv)
            else:
                raise ValueError(f"unexpected KDA tensor {name}")
            return

        # ---- BCSA / dense gated attention
        if leaf.startswith("self_attn."):
            sub = leaf[len("self_attn."):]
            key = {
                "q_proj.weight": T.ATTN_Q, "k_proj.weight": T.ATTN_K, "v_proj.weight": T.ATTN_V, "o_proj.weight": T.ATTN_OUT,
                "idx_q.weight": T.INDEXER_Q_PROJ, "idx_k.weight": T.INDEXER_K_PROJ,
            }.get(sub)
            if key is not None:
                if sub == "idx_q.weight":
                    assert tuple(data_torch.shape) == (4 * 128, hp["hidden_size"]), data_torch.shape
                if sub == "idx_k.weight":
                    assert tuple(data_torch.shape) == (128, hp["hidden_size"]), data_torch.shape
                yield fmt(key, bid), data_torch
            elif sub == "q_norm.weight":
                yield fmt(T.ATTN_Q_NORM, bid), data_torch + 1
            elif sub == "k_norm.weight":
                yield fmt(T.ATTN_K_NORM, bid), data_torch + 1
            elif sub == "idx_norm.weight":             # Qwen3_5RMSNorm (Gemma-style), shared by indexer q and k
                yield fmt(T.INDEXER_K_NORM, bid), data_torch + 1
            else:
                raise ValueError(f"unexpected attention tensor {name}")
            return

        # ---- Engram
        if leaf.startswith("engram."):
            sub = leaf[len("engram."):]
            if sub.startswith("tables."):              # tables.K.weight [rows, dim]
                k = int(sub.split(".")[1])
                yield f"blk.{bid}.engram_embd.{k}.weight", data_torch
            elif sub == "norm.weight":                 # plain nn.RMSNorm
                yield fmt(T.ENGRAM_NORM, bid), data_torch
            elif sub == "gate.weight":                 # [1, n_embd]
                yield fmt(T.ENGRAM_GATE, bid), data_torch
            elif sub == "gate.bias":
                yield fmt(T.ENGRAM_GATE, bid, suffix=".bias"), data_torch
            elif sub == "value.weight":                # [n_embd, dim*n_orders]
                yield fmt(T.ENGRAM_VALUE, bid), data_torch
            else:
                raise ValueError(f"unexpected Engram tensor {name}")
            return

        raise ValueError(f"unexpected tensor {name}")

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._hc_static:
            raise ValueError(f"incomplete mHC static parameters for layers {sorted(self._hc_static)}")
        logger.info("volundr: all tensors mapped")
