// Agens Volundr (Blockway) -- text model.
//
// Reference implementation (ground truth for the numerics): modeling_volundr.py, bcsa.py, volundr_extras.py shipped with the
// HF checkpoint. Layer recipe (x = mHC pre-mix of the residual streams):
//     h  = attn_norm(x)                           (Gemma-style (1+w) RMSNorm, +1 baked in at conversion)
//     y1 = x + mixer(h)                           mixer = KDA | BCSA | dense gated attention
//     y  = y1 + mlp(post_attention_norm(y1))
//     y  = y + engram(y)                          (Engram layers only)
//     H  = res . H + post (x) (y - x)             (mHC post-mix, res = Sinkhorn(...))
// and the final hidden state is output_norm(H[stream 0]).

#include "models.h"

#include "llama-batch.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

//
// graph inputs
//

// Engram works on token ids: e = id + 1 (0 encodes "before the start of the sequence", as the -1 padding of the reference)
class llm_graph_input_engram : public llm_graph_input_i {
public:
    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(ubatch->token && "Volundr: Engram layers need token ids (embedding inputs are not supported)");
        GGML_ASSERT(ggml_backend_buffer_is_host(e_ids->buffer));
        float * d = (float *) e_ids->data;
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            d[i] = (float) ubatch->token[i] + 1.0f; // exact: ids < 2^24
        }
    }

    ggml_tensor * e_ids = nullptr; // F32 [n_tokens]
};

// BCSA masks and block <-> cell maps (shared by all BCSA layers of the graph); see llama_kv_cache::set_input_bcsa
class llm_graph_input_bcsa : public llm_graph_input_i {
public:
    llm_graph_input_bcsa(const llama_kv_cache_context * mctx, uint32_t n_win, uint32_t n_cmp) : mctx(mctx), n_win(n_win), n_cmp(n_cmp) {}

    void set_input(const llama_ubatch * ubatch) override {
        mctx->set_input_bcsa(loc_mask, far_mask, blk_cell, cell_blk, win_cell, win_mask, blk_flat, ubatch, n_win, n_cmp);
    }

    ggml_tensor * loc_mask = nullptr; // F32 [n_kv,  n_seq_tokens, 1, n_seqs]
    ggml_tensor * far_mask = nullptr; // F32 [n_blk, n_seq_tokens, 1, n_seqs] (nullptr if no query can see a far block)
    ggml_tensor * blk_cell = nullptr; // I32 [n_cmp*n_blk, n_seqs]
    ggml_tensor * cell_blk = nullptr; // I32 [n_kv, n_seqs]

    // sparse decode (one query per sequence): gather only the window and the selected far blocks
    ggml_tensor * win_cell = nullptr; // I32 [n_win, n_seqs]   flat cell index (stream*kv_size + cell) of window slot w
    ggml_tensor * win_mask = nullptr; // F32 [n_win, n_seqs]   0 / -inf
    ggml_tensor * blk_flat = nullptr; // I32 [n_cmp*n_blk, n_seqs] flat cell index of position b*n_cmp + c
    bool sparse = false;

    int64_t n_kv  = 0;
    int64_t n_blk = 0;

    const llama_kv_cache_context * mctx;
    const uint32_t n_win;
    const uint32_t n_cmp;
};

//
// hparams / tensors
//

void llama_model_volundr::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // KDA (linear attention) dims, Qwen3.5 GDN conventions: d_inner = n_v_heads*head_dim, group_count = n_k_heads
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    ml.get_key(LLM_KV_KDA_GATE_RANK,      vp.kda_gate_rank);

    std::fill(hparams.is_recr_impl.begin(), hparams.is_recr_impl.end(), 0);
    ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, true);

    // BCSA
    std::fill(vp.is_bcsa.begin(), vp.is_bcsa.end(), 0);
    ml.get_key_or_arr(LLM_KV_BCSA_LAYERS, vp.is_bcsa, hparams.n_layer_all, true);
    ml.get_key(LLM_KV_BCSA_WINDOW,   vp.bcsa_window);
    ml.get_key(LLM_KV_BCSA_COMPRESS, vp.bcsa_compress);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

    bool any_bcsa = false;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        GGML_ASSERT(!(vp.is_bcsa[il] && hparams.is_recr(il)));
        any_bcsa = any_bcsa || vp.is_bcsa[il];
    }
    // per-cell indexer-key side cache (indexer_head_size f32 per cell and attention layer)
    hparams.indexer_kv = any_bcsa;

    // mHC
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               vp.mhc_streams);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, vp.mhc_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             vp.mhc_eps, false);
    GGML_ASSERT(vp.mhc_streams >= 1 && vp.mhc_streams <= 8);
    vp.sk = { (int) vp.mhc_streams, (int) vp.mhc_iters, vp.mhc_eps };

    // Engram
    std::fill(vp.is_engram.begin(), vp.is_engram.end(), 0);
    const bool has_engram = ml.get_key_or_arr(LLM_KV_ENGRAM_LAYERS, vp.is_engram, hparams.n_layer_all, false);
    hparams.n_embd_r_extra = 0;
    if (has_engram) {
        ml.get_key(LLM_KV_ENGRAM_ROWS, vp.engram_rows);
        ml.get_key(LLM_KV_ENGRAM_DIM,  vp.engram_dim);
        ml.get_arr_n(LLM_KV_ENGRAM_ORDERS, vp.n_engram_orders);
        ml.get_arr  (LLM_KV_ENGRAM_ORDERS, vp.engram_orders);
        ml.get_arr_n(LLM_KV_ENGRAM_PRIMES, vp.n_engram_primes);
        ml.get_arr  (LLM_KV_ENGRAM_PRIMES, vp.engram_primes);
        GGML_ASSERT(vp.n_engram_orders >= 1 && vp.n_engram_orders <= 4);
        GGML_ASSERT(vp.n_engram_primes >= 1 && vp.n_engram_primes <= 8);

        uint32_t max_order = 1;
        for (uint32_t k = 0; k < vp.n_engram_orders; ++k) {
            max_order = std::max(max_order, vp.engram_orders[k]);
        }
        for (uint32_t k = 0; k < vp.n_engram_orders; ++k) {
            auto & h = vp.hash[k];
            h.order    = vp.engram_orders[k];
            h.prefix   = max_order - 1;
            h.rows     = vp.engram_rows;
            h.n_primes = vp.n_engram_primes;
            for (uint32_t p = 0; p < vp.n_engram_primes; ++p) {
                h.primes[p] = vp.engram_primes[p];
            }
        }
        // the previous (max_order - 1) token ids of every sequence live in the recurrent "r" state of the Engram layers
        hparams.n_embd_r_extra = max_order - 1;
        for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
            if (vp.is_engram[il] && !hparams.is_recr(il)) {
                throw std::runtime_error("Volundr: Engram layers must be KDA (recurrent) layers in this implementation");
            }
        }
    }

    switch (hparams.n_layer()) {
        case 72: type = LLM_TYPE_32B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_volundr::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), { n_embd, n_vocab }, 0);

    const int64_t head_dim  = hparams.ssm_d_state;  // KDA head dim (K == V)
    const int64_t n_k_heads = hparams.ssm_n_group;
    const int64_t n_v_heads = hparams.ssm_dt_rank;
    const int64_t key_dim   = head_dim*n_k_heads;
    const int64_t value_dim = head_dim*n_v_heads;
    const int64_t conv_dim  = 2*key_dim + value_dim;
    const int64_t rank      = vp.kda_gate_rank;

    const int64_t n_hc       = vp.mhc_streams;
    const int64_t hc_mix_dim = 2*n_hc + n_hc*n_hc;

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, 0);

        if (hparams.is_recr(il)) {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,     "weight", il), { n_embd, conv_dim  }, 0);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,    "weight", il), { n_embd, value_dim }, 0);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,   "weight", il), { hparams.ssm_d_conv, conv_dim }, 0);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,       "bias",   il), { value_dim }, 0);      // dt_bias, per channel
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,           il), { n_v_heads }, 0);      // -exp(A_log)
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,     "weight", il), { n_embd, n_v_heads }, 0);
            layer.ssm_f_a    = create_tensor(tn(LLM_TENSOR_SSM_F_A,      "weight", il), { n_embd, rank }, 0);       // f_down
            layer.ssm_f_b    = create_tensor(tn(LLM_TENSOR_SSM_F_B,      "weight", il), { rank, value_dim }, 0);    // f_up
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,     "weight", il), { head_dim }, 0);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,      "weight", il), { value_dim, n_embd }, 0);
        } else {
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", il), { n_embd, n_embd_head_k*n_head*2 }, 0); // q + sigmoid gate
            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", il), { n_embd, n_embd_k_gqa }, 0);
            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", il), { n_embd, n_embd_v_gqa }, 0);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k*n_head, n_embd }, 0);
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, 0);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, 0);

            if (vp.is_bcsa[il]) {
                const int64_t n_idx = hparams.indexer_head_size;
                layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head*n_idx }, 0);
                layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, n_idx }, 0);
                layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { n_idx }, 0);  // shared by q and k
            }
        }

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), { n_embd, n_ff }, 0);

        if (vp.is_engram[il]) {
            const int64_t e_dim = vp.engram_dim*vp.n_engram_orders;
            for (uint32_t k = 0; k < vp.n_engram_orders; ++k) {
                layer.engram_embd[k] = create_tensor(tn(LLM_TENSOR_ENGRAM_EMBD, "weight", il, k), { vp.engram_dim, vp.engram_rows }, 0);
            }
            layer.engram_norm   = create_tensor(tn(LLM_TENSOR_ENGRAM_NORM,  "weight", il), { e_dim }, 0);
            layer.engram_gate   = create_tensor(tn(LLM_TENSOR_ENGRAM_GATE,  "weight", il), { n_embd, 1 }, 0);
            layer.engram_gate_b = create_tensor(tn(LLM_TENSOR_ENGRAM_GATE,  "bias",   il), { 1 }, 0);
            layer.engram_value  = create_tensor(tn(LLM_TENSOR_ENGRAM_VALUE, "weight", il), { e_dim, n_embd }, 0);
        }

        if (n_hc > 1) {
            layer.hc_dyn  = create_tensor(tn(LLM_TENSOR_HC_DYN,  "weight", il), { n_embd, hc_mix_dim }, 0);
            layer.hc_norm = create_tensor(tn(LLM_TENSOR_HC_NORM, "weight", il), { n_embd }, 0);
            layer.hc_base = create_tensor(tn(LLM_TENSOR_HC_BASE, "weight", il), { hc_mix_dim }, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_volundr::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

//
// graph
//

llama_model_volundr::graph::graph(const llama_model & model_, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(static_cast<const llama_model_volundr &>(model_)) {
    const auto & vp = model.vp;

    const int64_t n_hc = vp.mhc_streams;

    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == ubatch.n_seq_tokens*ubatch.n_seqs);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "inp_embd", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    bool any_engram = false;
    bool any_bcsa   = false;
    for (int il = 0; il < n_layer; ++il) {
        any_engram = any_engram || vp.is_engram[il];
        any_bcsa   = any_bcsa   || vp.is_bcsa[il];
    }

    if (any_engram) {
        auto ie = std::make_unique<llm_graph_input_engram>();
        ie->e_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_tokens);
        ggml_set_input(ie->e_ids);
        e_ids = ie->e_ids;
        res->add_input(std::move(ie));
    }

    llm_graph_input_bcsa * inp_bcsa = nullptr;
    if (any_bcsa) {
        const auto * mctx_attn = inp->get_attn()->mctx;

        const int64_t n_kv   = mctx_attn->get_n_kv();
        const int64_t n_stok = ubatch.n_seq_tokens;
        const int64_t n_seqs = ubatch.n_seqs;
        const int64_t C      = vp.bcsa_compress;
        const int64_t W      = vp.bcsa_window;

        auto ib = std::make_unique<llm_graph_input_bcsa>(mctx_attn, vp.bcsa_window, vp.bcsa_compress);
        ib->n_kv = n_kv;
        ib->loc_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, n_stok, 1, n_seqs);
        ggml_set_input(ib->loc_mask);

        // the far field only exists once a query is past the window by at least one full block; the reserve graph
        // (all positions 0, n_tokens > 1) keeps it so that the compute buffer is sized for it
        llama_pos pmax = -1;
        bool all_zero = true;
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            pmax     = std::max(pmax, ubatch.pos[i]);
            all_zero = all_zero && ubatch.pos[i] == 0;
        }
        const int64_t n_blk = n_kv / C;
        const bool use_far = n_blk > 0 && (pmax >= W + C - 1 || (all_zero && ubatch.n_tokens > 1));

        // sparse decode path: one token per sequence, non-transposed V cache (flash attention on), and enough context
        // for the gather to beat the dense scan; VOLUNDR_BCSA_SPARSE=0 forces the dense path, =1 forces sparse
        const char * sp_env = getenv("VOLUNDR_BCSA_SPARSE");
        const bool sparse_ok = use_far && n_stok == 1 && cparams.flash_attn;
        const bool sparse = sparse_ok && (sp_env ? atoi(sp_env) != 0 : n_kv > W + 4*C*(int64_t) hparams.indexer_top_k);

        if (sparse) {
            ib->sparse   = true;
            ib->win_cell = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, W, n_seqs);
            ib->win_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, W, n_seqs);
            ib->blk_flat = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, C*n_blk, n_seqs);
            ggml_set_input(ib->win_cell);
            ggml_set_input(ib->win_mask);
            ggml_set_input(ib->blk_flat);
        }

        if (use_far) {
            ib->n_blk = n_blk;
            ib->far_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_blk, n_stok, 1, n_seqs);
            ib->blk_cell = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, C*n_blk, n_seqs);
            ib->cell_blk = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_seqs);
            ggml_set_input(ib->far_mask);
            ggml_set_input(ib->blk_cell);
            ggml_set_input(ib->cell_blk);
        }

        inp_bcsa = (llm_graph_input_bcsa *) res->add_input(std::move(ib));
    }

    // mHC: every stream starts as the token embedding
    ggml_tensor * H = inpL;
    if (n_hc > 1) {
        H = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens), n_embd, n_hc, n_tokens, 1);
    }

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * x    = H;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        if (n_hc > 1) {
            // dynamic mixing coefficients from the stream mean
            ggml_tensor * hm = ggml_view_2d(ctx0, H, n_embd, n_tokens, H->nb[2], 0);
            for (int64_t s = 1; s < n_hc; ++s) {
                hm = ggml_add(ctx0, hm, ggml_view_2d(ctx0, H, n_embd, n_tokens, H->nb[2], s*H->nb[1]));
            }
            hm = ggml_scale(ctx0, hm, 1.0f/(float) n_hc);
            hm = build_norm(hm, layer.hc_norm, nullptr, LLM_NORM_RMS, il);

            ggml_tensor * d = build_lora_mm(layer.hc_dyn, hm);          // [2n + n^2, n_tokens]
            d = ggml_add(ctx0, d, layer.hc_base);                         // static parameters [pre | post | res]
            cb(d, "hc_mix", il);

            const size_t es = ggml_element_size(d);
            ggml_tensor * pre = ggml_view_2d(ctx0, d, n_hc,      n_tokens, d->nb[1], 0);
            post              = ggml_view_2d(ctx0, d, n_hc,      n_tokens, d->nb[1], n_hc*es);
            ggml_tensor * rl  = ggml_view_2d(ctx0, d, n_hc*n_hc, n_tokens, d->nb[1], 2*n_hc*es);

            comb = ggml_sinkhorn(ctx0, rl, (int) n_hc, vp.sk.iters, vp.sk.eps);   // [n, n, n_tokens], comb[i, j] = res[i][j]
            cb(comb, "hc_comb", il);

            x = ggml_dsv4_hc_pre(ctx0, H, pre);
        }
        cb(x, "layer_inp", il);

        ggml_tensor * cur = build_norm(x, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * engram_ctx = nullptr;

        if (hparams.is_recr(il)) {
            cur = build_kda(inp->get_recr(), cur, vp.is_engram[il] ? &engram_ctx : nullptr, il);
        } else if (vp.is_bcsa[il]) {
            cur = build_attn_bcsa(inp->get_attn(), inp_bcsa, cur, inp_pos, il);
        } else {
            cur = build_attn_dense(inp->get_attn(), cur, inp_pos, il);
        }
        cb(cur, "mixer_out", il);

        ggml_tensor * delta = cur;

        ggml_tensor * y1 = ggml_add(ctx0, cur, x);
        cb(y1, "attn_residual", il);

        cur = build_norm(y1, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up,   nullptr, layer.ffn_up_s,
                layer.ffn_gate, nullptr, layer.ffn_gate_s,
                layer.ffn_down, nullptr, layer.ffn_down_s,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        delta = ggml_add(ctx0, delta, cur);

        if (vp.is_engram[il]) {
            ggml_tensor * y = ggml_add(ctx0, y1, cur);
            ggml_tensor * e = build_engram(y, nullptr, engram_ctx, il);
            delta = ggml_add(ctx0, delta, e);
        }
        cb(delta, "layer_delta", il);

        if (n_hc > 1) {
            H = ggml_dsv4_hc_post(ctx0, delta, H, post, comb);   // res . H + post (x) delta
        } else {
            H = ggml_add(ctx0, x, delta);
        }
        cb(H, "l_out", il);
    }

    ggml_tensor * cur = n_hc > 1 ? ggml_view_2d(ctx0, H, n_embd, n_tokens, H->nb[2], 0) : H;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// KDA mixer: per-channel gated delta rule (Kimi Delta Attention) with a low-rank forget gate.
//   q,k: n_k_heads, v: n_v_heads (GVA). V heads were reordered to "tiled" order at conversion, so the ggml broadcast
//   (q/k head = v head % n_k_heads) matches the reference repeat_interleave grouping.
ggml_tensor * llama_model_volundr::graph::build_kda(llm_graph_input_rs * inp, ggml_tensor * cur, ggml_tensor ** engram_ctx, int il) {
    const auto & layer    = model.layers[il];
    const auto * mctx_cur = inp->mctx;

    const int64_t head_dim  = hparams.ssm_d_state;
    const int64_t n_k_heads = hparams.ssm_n_group;
    const int64_t n_v_heads = hparams.ssm_dt_rank;
    const int64_t key_dim   = head_dim*n_k_heads;
    const int64_t value_dim = head_dim*n_v_heads;
    const int64_t conv_dim  = 2*key_dim + value_dim;
    const int64_t d_conv    = hparams.ssm_d_conv;
    const int64_t n_conv    = (d_conv - 1)*conv_dim;

    const int64_t n_seqs = ubatch.n_seqs;
    const int64_t n_stok = ubatch.n_seq_tokens;

    const auto kv_head = mctx_cur->get_head();

    ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur);
    qkv = ggml_reshape_3d(ctx0, qkv, conv_dim, n_stok, n_seqs);
    cb(qkv, "kda_qkv", il);

    ggml_tensor * z = build_lora_mm(layer.wqkv_gate, cur);
    cb(z, "kda_z", il);

    ggml_tensor * beta = build_lora_mm(layer.ssm_beta, cur);
    beta = ggml_sigmoid(ctx0, ggml_reshape_4d(ctx0, beta, 1, n_v_heads, n_stok, n_seqs));
    cb(beta, "kda_beta", il);

    // forget gate (log space): g = -exp(A_log) * softplus(f_up(f_down(x)) + dt_bias), per value head and channel
    ggml_tensor * g = build_lora_mm(layer.ssm_f_b, build_lora_mm(layer.ssm_f_a, cur));
    g = ggml_softplus(ctx0, ggml_add(ctx0, g, layer.ssm_dt));
    g = ggml_mul(ctx0, ggml_reshape_3d(ctx0, g, head_dim, n_v_heads, n_tokens), ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_v_heads, 1));
    g = ggml_reshape_4d(ctx0, g, head_dim, n_v_heads, n_stok, n_seqs);
    cb(g, "kda_g", il);

    // recurrent "r" state row: [conv state (d_conv-1)*conv_dim | Engram previous ids]
    ggml_tensor * r_all = mctx_cur->get_r_l(il);
    ggml_tensor * s_all = mctx_cur->get_s_l(il);

    ggml_tensor * r = build_rs(inp, r_all, hparams.n_embd_r(), n_seqs);   // [n_embd_r, n_seqs]

    ggml_tensor * conv_state = ggml_view_3d(ctx0, r, d_conv - 1, conv_dim, n_seqs,
            (d_conv - 1)*ggml_element_size(r), r->nb[1], 0);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_state, ggml_transpose(ctx0, qkv), 0);   // [d_conv-1+n_stok, conv_dim, n_seqs]
    cb(conv_input, "kda_conv_input", il);

    {
        ggml_tensor * last = ggml_view_3d(ctx0, conv_input, d_conv - 1, conv_dim, n_seqs,
                conv_input->nb[1], conv_input->nb[2], n_stok*ggml_element_size(conv_input));
        ggml_tensor * dst = ggml_view_2d(ctx0, r_all, n_conv, n_seqs, r_all->nb[1], kv_head*r_all->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, last, dst));
    }

    if (engram_ctx) {
        // previous token ids of each sequence, then this ubatch's ids
        const int64_t n_extra = hparams.n_embd_r_extra;
        GGML_ASSERT(n_extra > 0 && e_ids);

        ggml_tensor * prev = ggml_view_2d(ctx0, r, n_extra, n_seqs, r->nb[1], n_conv*ggml_element_size(r));
        ggml_tensor * ext  = ggml_concat(ctx0, prev, ggml_reshape_2d(ctx0, e_ids, n_stok, n_seqs), 0);  // [n_extra + n_stok, n_seqs]

        ggml_tensor * last = ggml_view_2d(ctx0, ext, n_extra, n_seqs, ext->nb[1], n_stok*ggml_element_size(ext));
        ggml_tensor * dst  = ggml_view_2d(ctx0, r_all, n_extra, n_seqs, r_all->nb[1],
                kv_head*r_all->nb[1] + n_conv*ggml_element_size(r_all));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, last, dst));

        *engram_ctx = ext;
    }

    ggml_tensor * conv_out = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv_input, layer.ssm_conv1d));   // [conv_dim, n_stok, n_seqs]
    cb(conv_out, "kda_conv_out", il);

    const int64_t nb1_qkv = ggml_row_size(conv_out->type, conv_dim);

    ggml_tensor * q = ggml_view_4d(ctx0, conv_out, head_dim, n_k_heads, n_stok, n_seqs,
            ggml_row_size(conv_out->type, head_dim), nb1_qkv, nb1_qkv*n_stok, 0);
    ggml_tensor * k = ggml_view_4d(ctx0, conv_out, head_dim, n_k_heads, n_stok, n_seqs,
            ggml_row_size(conv_out->type, head_dim), nb1_qkv, nb1_qkv*n_stok, ggml_row_size(conv_out->type, key_dim));
    ggml_tensor * v = ggml_view_4d(ctx0, conv_out, head_dim, n_v_heads, n_stok, n_seqs,
            ggml_row_size(conv_out->type, head_dim), nb1_qkv, nb1_qkv*n_stok, ggml_row_size(conv_out->type, 2*key_dim));

    q = ggml_l2_norm(ctx0, q, hparams.f_norm_rms_eps);
    k = ggml_l2_norm(ctx0, k, hparams.f_norm_rms_eps);

    if (n_k_heads != n_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        // unfused paths need matching head counts (tiled repeat, see the V-head reorder at conversion)
        q = ggml_repeat_4d(ctx0, q, head_dim, n_v_heads, n_stok, n_seqs);
        k = ggml_repeat_4d(ctx0, k, head_dim, n_v_heads, n_stok, n_seqs);
    }

    ggml_tensor * state = build_rs(inp, s_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_v_heads, n_seqs);

    ggml_tensor * o = build_recurrent_attn(inp, s_all, q, k, v, g, beta, state, il);   // [head_dim, n_v_heads, n_stok, n_seqs]
    cb(o, "kda_o", il);

    // gated RMSNorm (plain weight): norm(o) * w * silu(z)
    ggml_tensor * z4 = ggml_reshape_4d(ctx0, z, head_dim, n_v_heads, n_stok, n_seqs);
    o = build_norm(o, layer.ssm_norm, nullptr, LLM_NORM_RMS, il);
    o = ggml_mul(ctx0, o, ggml_silu(ctx0, z4));
    o = ggml_reshape_2d(ctx0, o, value_dim, n_tokens);

    return build_lora_mm(layer.ssm_out, o, layer.ssm_out_s);
}

// Qwen3.5 gated full attention (q has a per-head sigmoid output gate), used by the last ("attn") layer
ggml_tensor * llama_model_volundr::graph::build_attn_dense(llm_graph_input_attn_kv * inp, ggml_tensor * cur, ggml_tensor * inp_pos, int il) {
    const auto & layer = model.layers[il];
    const int64_t D = hparams.n_embd_head_k();

    ggml_tensor * qg = build_lora_mm(layer.wq, cur);   // [(2*D)*n_head, n_tokens]: per head [q | gate]

    ggml_tensor * q = ggml_view_3d(ctx0, qg, D, n_head, n_tokens, ggml_element_size(qg)*D*2, ggml_element_size(qg)*D*2*n_head, 0);
    q = build_norm(q, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);

    ggml_tensor * gate = ggml_view_3d(ctx0, qg, D, n_head, n_tokens, ggml_element_size(qg)*D*2, ggml_element_size(qg)*D*2*n_head,
            ggml_element_size(qg)*D);
    gate = ggml_cont_2d(ctx0, gate, D*n_head, n_tokens);

    ggml_tensor * k = ggml_reshape_3d(ctx0, build_lora_mm(layer.wk, cur), D, n_head_kv, n_tokens);
    k = build_norm(k, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * v = ggml_reshape_3d(ctx0, build_lora_mm(layer.wv, cur), D, n_head_kv, n_tokens);

    q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    k = ggml_rope_ext(ctx0, k, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "Qcur", il);
    cb(k, "Kcur", il);
    cb(v, "Vcur", il);

    cur = build_attn(inp, nullptr, nullptr, nullptr, q, k, v, nullptr, nullptr, nullptr, 1.0f/sqrtf((float) D), il);
    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));

    return build_lora_mm(layer.wo, cur, layer.wo_s);
}

// mean of C consecutive rows (one block) of the gathered cells: s [n_kv, T, h, G] -> [n_blk, T, h, G]
ggml_tensor * llama_model_volundr::graph::pool_kv_scores(ggml_tensor * s, ggml_tensor * blk_cell, int64_t n_blk) {
    const int64_t C  = model.vp.bcsa_compress;
    const int64_t T  = s->ne[1];
    const int64_t h  = s->ne[2];
    const int64_t G  = s->ne[3];

    ggml_tensor * st = ggml_cont(ctx0, ggml_permute(ctx0, s, 2, 0, 1, 3));   // [T, h, n_kv, G]
    st = ggml_reshape_3d(ctx0, st, T*h, s->ne[0], G);

    ggml_tensor * gt = ggml_get_rows(ctx0, st, blk_cell);                   // [T*h, C*n_blk, G]

    ggml_tensor * acc = ggml_view_3d(ctx0, gt, T*h, n_blk, G, C*gt->nb[1], gt->nb[2], 0);
    for (int64_t c = 1; c < C; ++c) {
        acc = ggml_add(ctx0, acc, ggml_view_3d(ctx0, gt, T*h, n_blk, G, C*gt->nb[1], gt->nb[2], c*gt->nb[1]));
    }
    acc = ggml_scale(ctx0, acc, 1.0f/(float) C);
    acc = ggml_reshape_4d(ctx0, acc, T, h, n_blk, G);

    return ggml_cont(ctx0, ggml_permute(ctx0, acc, 1, 2, 0, 3));            // [n_blk, T, h, G]
}

// spread the far-block probabilities back onto their cells (p/C each): p [n_blk, T, h, G] -> [n_kv, T, h, G]
ggml_tensor * llama_model_volundr::graph::scatter_blk_probs(ggml_tensor * p, ggml_tensor * cell_blk, int64_t n_kv) {
    const int64_t C  = model.vp.bcsa_compress;
    const int64_t T  = p->ne[1];
    const int64_t h  = p->ne[2];
    const int64_t G  = p->ne[3];

    ggml_tensor * pt = ggml_cont(ctx0, ggml_permute(ctx0, p, 2, 0, 1, 3));   // [T, h, n_blk, G]
    pt = ggml_reshape_3d(ctx0, pt, T*h, p->ne[0], G);
    pt = ggml_pad(ctx0, pt, 0, 1, 0, 0);                                     // row n_blk = 0 (cells outside complete blocks)

    ggml_tensor * w = ggml_get_rows(ctx0, pt, cell_blk);                     // [T*h, n_kv, G]
    w = ggml_reshape_4d(ctx0, w, T, h, n_kv, G);
    w = ggml_cont(ctx0, ggml_permute(ctx0, w, 1, 2, 0, 3));                  // [n_kv, T, h, G]

    return ggml_scale(ctx0, w, 1.0f/(float) C);
}

// BCSA: one softmax over (a) exact attention to the W most recent positions and (b) the mean-pooled C-token blocks that lie
// entirely before the window and are among the indexer's top-k blocks for the query.
// Implementation note: q.mean(k_block) == mean(q.k), and sum_b p_b * mean(v_block) == sum_cells (p_b/C) v, so the far field
// is evaluated on the full-cache scores (pooled per block) and folded back onto the cells before the single P.V product --
// exact, no pooled K/V needs to be materialised, any cell layout works. Cost is that of dense attention over the cache.
ggml_tensor * llama_model_volundr::graph::build_attn_bcsa(llm_graph_input_attn_kv * inp, llm_graph_input_bcsa * ib,
        ggml_tensor * cur, ggml_tensor * inp_pos, int il) {
    const auto & layer = model.layers[il];

    const int64_t D     = hparams.n_embd_head_k();
    const int64_t n_idx = hparams.indexer_head_size;
    const int64_t n_ih  = hparams.indexer_n_head;
    const int64_t G     = ubatch.n_seqs;
    const int64_t T     = ubatch.n_seq_tokens;

    ggml_tensor * qg = build_lora_mm(layer.wq, cur);

    ggml_tensor * q = ggml_view_3d(ctx0, qg, D, n_head, n_tokens, ggml_element_size(qg)*D*2, ggml_element_size(qg)*D*2*n_head, 0);
    q = build_norm(q, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);

    ggml_tensor * gate = ggml_view_3d(ctx0, qg, D, n_head, n_tokens, ggml_element_size(qg)*D*2, ggml_element_size(qg)*D*2*n_head,
            ggml_element_size(qg)*D);
    gate = ggml_cont_2d(ctx0, gate, D*n_head, n_tokens);

    ggml_tensor * k = ggml_reshape_3d(ctx0, build_lora_mm(layer.wk, cur), D, n_head_kv, n_tokens);
    k = build_norm(k, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * v = ggml_reshape_3d(ctx0, build_lora_mm(layer.wv, cur), D, n_head_kv, n_tokens);

    q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    k = ggml_rope_ext(ctx0, k, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "Qcur", il);
    cb(k, "Kcur", il);
    cb(v, "Vcur", il);

    // indexer key: one 128-d key per position (shared RMSNorm with the query heads), kept in the KV cache side store
    ggml_tensor * ki = ggml_reshape_3d(ctx0, build_lora_mm(layer.index_k_proj, cur), n_idx, 1, n_tokens);
    ki = build_norm(ki, layer.index_k_norm, nullptr, LLM_NORM_RMS, il);

    const auto * mctx_cur = inp->mctx;

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, mctx_cur->cpy_k    (ctx0, k,  inp->get_k_idxs(), il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v    (ctx0, v,  inp->get_v_idxs(), il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_k_idx(ctx0, ki, inp->get_k_idxs(), il));

    ggml_tensor * kc = mctx_cur->get_k(ctx0, il);   // [D, n_head_kv, n_kv, ns]
    ggml_tensor * vc = mctx_cur->get_v(ctx0, il);

    const int64_t n_kv = kc->ne[2];
    GGML_ASSERT(n_kv == ib->n_kv);
    GGML_ASSERT(kc->ne[3] == 1 || kc->ne[3] == G);

    const float kq_scale = 1.0f/sqrtf((float) D);

    ggml_tensor * kcp = ggml_permute(ctx0, kc, 0, 2, 1, 3);                              // [D, n_kv, n_head_kv, ns]

    // V^T [n_kv, D, n_head_kv, ns]
    const bool v_trans = vc->nb[1] > vc->nb[2];
    ggml_tensor * vt = ggml_permute(ctx0, vc, 0, 2, 1, 3);
    if (!v_trans) {
        vt = ggml_cont(ctx0, ggml_transpose(ctx0, vt));
    }

    // indexer queries (mean over the indexer heads; the block score is linear in q) and the cached indexer keys
    ggml_tensor * qm  = nullptr;
    ggml_tensor * kic = nullptr;
    if (ib->n_blk > 0) {
        ggml_tensor * qi = ggml_reshape_3d(ctx0, build_lora_mm(layer.index_q_proj, cur), n_idx, n_ih, n_tokens);
        qi = build_norm(qi, layer.index_k_norm, nullptr, LLM_NORM_RMS, il);
        qm = ggml_view_2d(ctx0, qi, n_idx, n_tokens, qi->nb[2], 0);
        for (int64_t h = 1; h < n_ih; ++h) {
            qm = ggml_add(ctx0, qm, ggml_view_2d(ctx0, qi, n_idx, n_tokens, qi->nb[2], h*qi->nb[1]));
        }
        qm = ggml_scale(ctx0, qm, 1.0f/((float) n_ih*sqrtf((float) n_idx)));
        qm = ggml_reshape_4d(ctx0, qm, n_idx, T, 1, G);

        kic = mctx_cur->get_k_idx(ctx0, il);   // [n_idx, 1, n_kv, ns]
        kic = ggml_view_4d(ctx0, kic, n_idx, n_kv, 1, kic->ne[3], kic->nb[2], kic->nb[3], kic->nb[3], 0);
    }

    if (ib->sparse) {
        // ---- sparse decode: one query per sequence; gather the W window cells and the C cells of each selected block
        GGML_ASSERT(T == 1 && ib->n_blk > 0 && !v_trans);
        const int64_t n_blk   = ib->n_blk;
        const int64_t W       = model.vp.bcsa_window;
        const int64_t C       = model.vp.bcsa_compress;
        const int64_t n_sel   = std::min<int64_t>(hparams.indexer_top_k, n_blk);
        const int64_t n_gqa   = kc->ne[0]*kc->ne[1];
        const int64_t n_kvh   = kc->ne[1];
        const int64_t ns      = kc->ne[3];
        const int64_t kv_size = kc->nb[3]/kc->nb[2];
        const int64_t rows    = (ns - 1)*kv_size + n_kv;

        // flat [n_gqa, rows] views of the caches over the streams of this ubatch (indices are stream*kv_size + cell)
        ggml_tensor * k2 = ggml_view_2d(ctx0, kc, n_gqa, rows, kc->nb[2], 0);
        ggml_tensor * v2 = ggml_view_2d(ctx0, vc, n_gqa, rows, vc->nb[2], 0);

        // indexer over all complete far blocks (128-d keys: cheap) -> top-k
        ggml_tensor * isc = ggml_mul_mat(ctx0, kic, qm);                                  // [n_kv, 1, 1, G]
        ggml_mul_mat_set_prec(isc, GGML_PREC_F32);
        isc = pool_kv_scores(isc, ib->blk_cell, n_blk);                                   // [n_blk, 1, 1, G]
        isc = ggml_add(ctx0, isc, ib->far_mask);
        ggml_tensor * top = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_top_k(ctx0, isc, (int) n_sel)), n_sel, G);
        cb(top, "bcsa_top_k", il);

        ggml_tensor * fidx = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, ib->blk_flat, C, n_blk, G), top);    // [C, n_sel, G]
        ggml_tensor * fsel = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, ib->far_mask, 1, n_blk, G), top);   // [1, n_sel, G]
        fsel = ggml_reshape_4d(ctx0, fsel, n_sel, 1, 1, G);

        ggml_tensor * wi = ggml_reshape_1d(ctx0, ib->win_cell, W*G);
        ggml_tensor * fi = ggml_reshape_1d(ctx0, fidx, C*n_sel*G);

        ggml_tensor * kw = ggml_reshape_4d(ctx0, ggml_get_rows(ctx0, k2, wi), D, n_kvh, W,       G);
        ggml_tensor * kf = ggml_reshape_4d(ctx0, ggml_get_rows(ctx0, k2, fi), D, n_kvh, C*n_sel, G);
        ggml_tensor * vw = ggml_reshape_4d(ctx0, ggml_get_rows(ctx0, v2, wi), D, n_kvh, W,       G);
        ggml_tensor * vf = ggml_reshape_4d(ctx0, ggml_get_rows(ctx0, v2, fi), D, n_kvh, C*n_sel, G);

        ggml_tensor * q4 = ggml_permute(ctx0, ggml_view_4d(ctx0, q, D, n_head, 1, G, q->nb[1], q->nb[2], q->nb[2], 0), 0, 2, 1, 3); // [D, 1, H, G]

        ggml_tensor * sw = ggml_mul_mat(ctx0, ggml_permute(ctx0, kw, 0, 2, 1, 3), q4);     // [W, 1, H, G]
        sw = ggml_add(ctx0, ggml_scale(ctx0, sw, kq_scale), ggml_reshape_4d(ctx0, ib->win_mask, W, 1, 1, G));

        ggml_tensor * sf = ggml_mul_mat(ctx0, ggml_permute(ctx0, kf, 0, 2, 1, 3), q4);     // [C*n_sel, 1, H, G]
        sf = ggml_sum_rows(ctx0, ggml_reshape_4d(ctx0, sf, C, n_sel, n_head, G));        // [1, n_sel, H, G]
        sf = ggml_add(ctx0, ggml_scale(ctx0, ggml_reshape_4d(ctx0, sf, n_sel, 1, n_head, G), kq_scale/(float) C), fsel);

        ggml_tensor * p = ggml_soft_max(ctx0, ggml_concat(ctx0, sw, sf, 0));             // [W + n_sel, 1, H, G]
        cb(p, "bcsa_p", il);

        ggml_tensor * pw = ggml_view_4d(ctx0, p, W,     1, n_head, G, p->nb[1], p->nb[2], p->nb[3], 0);
        ggml_tensor * pf = ggml_cont(ctx0, ggml_view_4d(ctx0, p, n_sel, 1, n_head, G, p->nb[1], p->nb[2], p->nb[3], W*ggml_element_size(p)));
        pf = ggml_repeat_4d(ctx0, ggml_reshape_4d(ctx0, pf, 1, n_sel, n_head, G), C, n_sel, n_head, G);
        pf = ggml_reshape_4d(ctx0, ggml_scale(ctx0, pf, 1.0f/(float) C), C*n_sel, 1, n_head, G);

        ggml_tensor * ow = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, vw, 1, 2, 0, 3)), pw);   // [D, 1, H, G]
        ggml_tensor * of = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, vf, 1, 2, 0, 3)), pf);
        ggml_tensor * o  = ggml_add(ctx0, ow, of);
        o = ggml_cont_2d(ctx0, ggml_permute(ctx0, o, 0, 2, 1, 3), D*n_head, n_tokens);
        cb(o, "bcsa_out", il);

        o = ggml_mul(ctx0, o, ggml_sigmoid(ctx0, gate));
        return build_lora_mm(layer.wo, o, layer.wo_s);
    }

    // queries are processed in chunks so that the [n_kv, chunk, n_head, G] score tensors stay bounded for long prompts
    static const int64_t q_chunk = getenv("VOLUNDR_BCSA_QCHUNK") ? std::max(1, atoi(getenv("VOLUNDR_BCSA_QCHUNK"))) : 64;

    ggml_tensor * o = nullptr;
    for (int64_t t0 = 0; t0 < T; t0 += q_chunk) {
        const int64_t tq = std::min(q_chunk, T - t0);

        // q [D, n_head, n_tokens] -> [D, tq, n_head, G] (tokens are grouped by sequence)
        ggml_tensor * q4 = ggml_permute(ctx0,
                ggml_view_4d(ctx0, q, D, n_head, tq, G, q->nb[1], q->nb[2], q->nb[2]*T, t0*q->nb[2]), 0, 2, 1, 3);

        ggml_tensor * kq = ggml_mul_mat(ctx0, kcp, q4);                                   // [n_kv, tq, n_head, G]
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        cb(kq, "bcsa_kq", il);

        ggml_tensor * lm = ggml_view_4d(ctx0, ib->loc_mask, n_kv, tq, 1, G,
                ib->loc_mask->nb[1], ib->loc_mask->nb[2], ib->loc_mask->nb[3], t0*ib->loc_mask->nb[1]);
        ggml_tensor * s_loc = ggml_add(ctx0, ggml_scale(ctx0, kq, kq_scale), lm);

        ggml_tensor * p_tot = nullptr;

        if (ib->n_blk == 0) {
            p_tot = ggml_soft_max(ctx0, s_loc);
        } else {
            const int64_t n_blk = ib->n_blk;

            ggml_tensor * fm = ggml_view_4d(ctx0, ib->far_mask, n_blk, tq, 1, G,
                    ib->far_mask->nb[1], ib->far_mask->nb[2], ib->far_mask->nb[3], t0*ib->far_mask->nb[1]);

            // indexer: block score = mean over indexer heads of q_h . mean_block(k) * Di^-0.5 (== mean over cells of qbar . k)
            ggml_tensor * qmc = ggml_view_4d(ctx0, qm, n_idx, tq, 1, G, qm->nb[1], qm->nb[2], qm->nb[3], t0*qm->nb[1]);
            ggml_tensor * isc = ggml_mul_mat(ctx0, kic, qmc);                             // [n_kv, tq, 1, G]
            ggml_mul_mat_set_prec(isc, GGML_PREC_F32);
            isc = pool_kv_scores(isc, ib->blk_cell, n_blk);                               // [n_blk, tq, 1, G]
            isc = ggml_add(ctx0, isc, fm);
            cb(isc, "bcsa_idx_score", il);

            const int64_t n_sel = std::min<int64_t>(hparams.indexer_top_k, n_blk);
            ggml_tensor * top = ggml_cont(ctx0, ggml_top_k(ctx0, isc, (int) n_sel));      // [n_sel, tq, 1, G]
            cb(top, "bcsa_top_k", il);

            // selection mask: 0 on the top-k blocks, -inf elsewhere (plus the far-field validity)
            ggml_tensor * sel = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_blk, tq, 1, G), -INFINITY);  // [n_blk, tq, 1, G]
            sel = ggml_view_4d(ctx0, sel, 1, n_blk, tq, G, sel->nb[0], sel->nb[1], sel->nb[3], 0);
            ggml_tensor * zeros = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, n_sel, tq, G), 0.0f);
            sel = ggml_set_rows(ctx0, sel, zeros, ggml_reshape_3d(ctx0, top, n_sel, tq, G));
            sel = ggml_reshape_4d(ctx0, sel, n_blk, tq, 1, G);
            sel = ggml_add(ctx0, sel, fm);

            ggml_tensor * s_far = pool_kv_scores(kq, ib->blk_cell, n_blk);                // [n_blk, tq, n_head, G]
            s_far = ggml_add(ctx0, ggml_scale(ctx0, s_far, kq_scale), sel);

            ggml_tensor * p = ggml_soft_max(ctx0, ggml_concat(ctx0, s_loc, s_far, 0));   // [n_kv + n_blk, tq, n_head, G]
            cb(p, "bcsa_p", il);

            ggml_tensor * p_loc = ggml_view_4d(ctx0, p, n_kv,  tq, n_head, G, p->nb[1], p->nb[2], p->nb[3], 0);
            ggml_tensor * p_far = ggml_view_4d(ctx0, p, n_blk, tq, n_head, G, p->nb[1], p->nb[2], p->nb[3], n_kv*ggml_element_size(p));

            p_tot = ggml_add(ctx0, p_loc, scatter_blk_probs(p_far, ib->cell_blk, n_kv));
        }

        ggml_tensor * oc = ggml_mul_mat(ctx0, vt, p_tot);                                 // [D, tq, n_head, G]
        ggml_mul_mat_set_prec(oc, GGML_PREC_F32);   // P.V over up to n_kv cells: CUDA would otherwise accumulate in F16
        oc = ggml_cont(ctx0, ggml_permute(ctx0, oc, 0, 2, 1, 3));                        // [D, n_head, tq, G]
        o = o ? ggml_concat(ctx0, o, oc, 2) : oc;
    }

    o = ggml_cont_2d(ctx0, o, D*n_head, n_tokens);
    cb(o, "bcsa_out", il);

    o = ggml_mul(ctx0, o, ggml_sigmoid(ctx0, gate));

    return build_lora_mm(layer.wo, o, layer.wo_s);
}

// Engram: hashed n-gram rows of every order, concatenated, RMS-normed, projected, gated by sigmoid(gate(y))
ggml_tensor * llama_model_volundr::graph::build_engram(ggml_tensor * y, ggml_tensor *, ggml_tensor * ext, int il) {
    const auto & layer = model.layers[il];
    const auto & vp    = model.vp;

    GGML_ASSERT(ext && "Volundr: missing Engram context");

    const int64_t n_seqs = ubatch.n_seqs;
    const int64_t n_stok = ubatch.n_seq_tokens;

    ggml_tensor * e = nullptr;
    for (uint32_t kk = 0; kk < vp.n_engram_orders; ++kk) {
        const auto & hp = vp.hash[kk];
        int32_t primes[8];
        for (uint32_t i = 0; i < hp.n_primes; ++i) {
            primes[i] = (int32_t) hp.primes[i];
        }
        ggml_tensor * rows = ggml_ngram_hash(ctx0, ext, (int) hp.order, (int) hp.prefix, (int) hp.rows, primes, (int) hp.n_primes);
        GGML_ASSERT(rows->ne[0] == n_stok && rows->ne[1] == n_seqs);
        rows = ggml_reshape_1d(ctx0, rows, n_tokens);
        cb(rows, "engram_rows", il);

        ggml_tensor * ek = ggml_get_rows(ctx0, layer.engram_embd[kk], rows);   // [engram_dim, n_tokens]
        e = e ? ggml_concat(ctx0, e, ek, 0) : ek;
    }

    e = build_norm(e, layer.engram_norm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * val = build_lora_mm(layer.engram_value, e);                     // [n_embd, n_tokens]

    ggml_tensor * g = build_lora_mm(layer.engram_gate, y);                        // [1, n_tokens]
    g = ggml_sigmoid(ctx0, ggml_add(ctx0, g, layer.engram_gate_b));
    cb(g, "engram_gate", il);

    ggml_tensor * out = ggml_mul(ctx0, val, g);
    cb(out, "engram_out", il);

    return out;
}
