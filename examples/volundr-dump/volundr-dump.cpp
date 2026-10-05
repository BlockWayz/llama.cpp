// llama-volundr-dump: parity helper for the Volundr GGUF port.
// Evaluates a token sequence (int32 file) and writes
//   --out-logits F : float32 [n_tokens, n_vocab] logits of every prompt position (or only the last with --last-only)
//   --n-gen N --out-gen F : greedy continuation ids (int32 [N]) and --out-gen-logits F float32 [N, n_vocab]
//   --dump-layers F : per-layer residual (mHC stream 0) "l_out-<il>" records: int32 il, int32 n_tok, int32 n_embd, float32 data
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct layer_dump {
    FILE * f = nullptr;
    std::vector<float> buf;
};

#include <map>
// --profile: per-node wall time during generation (the callback splits the graph at every node, so absolute numbers include
// one scheduler round-trip per node; use the ranking)
struct node_prof {
    bool on = false;
    int64_t last = 0;
    std::map<std::string, std::pair<int64_t, int64_t>> acc; // key -> (us, count)
};
static node_prof g_prof;

static bool cb_prof(struct ggml_tensor * t, bool ask, void * ud) {
    (void) ud;
    if (ask) {
        return g_prof.on;
    }
    const int64_t now = ggml_time_us();
    std::string nm = t->name;
    const size_t dash = nm.rfind('-');
    if (dash != std::string::npos) nm = nm.substr(0, dash);
    std::string key = std::string(ggml_op_desc(t)) + " : " + nm;
    auto & a = g_prof.acc[key];
    a.first += now - g_prof.last;
    a.second += 1;
    g_prof.last = now;
    return true;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    auto * st = (layer_dump *) ud;
    if (strncmp(t->name, "l_out-", 6) != 0) {
        return false;
    }
    for (const char * c = t->name + 6; *c; ++c) {   // exact "l_out-<il>" only (not the "(view)" children)
        if (*c < '0' || *c > '9') {
            return false;
        }
    }
    if (ask) {
        return true;
    }
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    const int il     = atoi(t->name + 6);
    const int n_embd = (int) t->ne[0];
    const int n_tok  = (int) (ggml_n_dims(t) >= 3 ? t->ne[2] : t->ne[1]);
    st->buf.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, st->buf.data(), 0, ggml_nbytes(t));
    const int32_t hdr[3] = { il, n_tok, n_embd };
    fwrite(hdr, sizeof(int32_t), 3, st->f);
    for (int i = 0; i < n_tok; ++i) {
        // stream 0 of token i: offset i*nb[2] (3D) or i*nb[1] (2D)
        const size_t off = (ggml_n_dims(t) >= 3 ? i*t->nb[2] : i*t->nb[1]) / sizeof(float);
        fwrite(st->buf.data() + off, sizeof(float), n_embd, st->f);
    }
    return true;
}

static ggml_type parse_type(const std::string & s) {
    if (s == "f32")  return GGML_TYPE_F32;
    if (s == "f16")  return GGML_TYPE_F16;
    if (s == "bf16") return GGML_TYPE_BF16;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    fprintf(stderr, "bad cache type %s\n", s.c_str());
    exit(1);
}

int main(int argc, char ** argv) {
    std::string model_path, tok_path, out_logits, out_gen, out_gen_logits, dump_layers, ref_logits, ref_gen;
    int n_gpu_layers = 0;
    std::string split_mode = "layer";
    int n_gen = 0, n_threads = 32, n_ubatch = 512, n_batch = 2048, fa = -1;
    bool last_only = false;
    bool profile = false;
    ggml_type tk = GGML_TYPE_F32, tv = GGML_TYPE_F32;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return std::string(argv[++i]); };
        if      (a == "-m")               model_path = next();
        else if (a == "--tokens")         tok_path = next();
        else if (a == "--out-logits")     out_logits = next();
        else if (a == "--last-only")      last_only = true;
        else if (a == "--profile")        profile = true;
        else if (a == "--ref-logits")     ref_logits = next();
        else if (a == "--ref-gen")        ref_gen = next();
        else if (a == "-ngl")             n_gpu_layers = atoi(next().c_str());
        else if (a == "-sm")              split_mode = next();
        else if (a == "--n-gen")          n_gen = atoi(next().c_str());
        else if (a == "--out-gen")        out_gen = next();
        else if (a == "--out-gen-logits") out_gen_logits = next();
        else if (a == "--dump-layers")    dump_layers = next();
        else if (a == "-t")               n_threads = atoi(next().c_str());
        else if (a == "-ub")              n_ubatch = atoi(next().c_str());
        else if (a == "-b")               n_batch = atoi(next().c_str());
        else if (a == "-fa")              fa = atoi(next().c_str());
        else if (a == "-ctk")             tk = parse_type(next());
        else if (a == "-ctv")             tv = parse_type(next());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    std::vector<llama_token> toks;
    {
        FILE * f = fopen(tok_path.c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", tok_path.c_str()); return 1; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        toks.resize(sz / sizeof(int32_t));
        if (fread(toks.data(), sizeof(int32_t), toks.size(), f) != toks.size()) { return 1; }
        fclose(f);
    }
    const int n_prompt = (int) toks.size();

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    mparams.split_mode   = split_mode == "none" ? LLAMA_SPLIT_MODE_NONE : split_mode == "row" ? LLAMA_SPLIT_MODE_ROW : LLAMA_SPLIT_MODE_LAYER;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    layer_dump ld;
    auto cparams = llama_context_default_params();
    cparams.n_ctx           = ((n_prompt + n_gen + 255)/256 + 1)*256;
    cparams.n_batch         = n_batch;
    cparams.n_ubatch        = n_ubatch;
    cparams.n_seq_max       = 1;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.type_k          = tk;
    cparams.type_v          = tv;
    cparams.flash_attn_type = (enum llama_flash_attn_type) fa;
    if (profile) {
        cparams.cb_eval = cb_prof;
        cparams.cb_eval_user_data = nullptr;
    } else if (!dump_layers.empty()) {
        ld.f = fopen(dump_layers.c_str(), "wb");
        cparams.cb_eval = cb_eval;
        cparams.cb_eval_user_data = &ld;
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    FILE * fl = out_logits.empty() ? nullptr : fopen(out_logits.c_str(), "wb");
    FILE * fr = ref_logits.empty() ? nullptr : fopen(ref_logits.c_str(), "rb");
    std::vector<float> refrow(n_vocab);
    struct { int64_t n = 0, top1 = 0, n_far = 0, top1_far = 0; double kl = 0, kl_far = 0, kl_max = 0, max_abs = 0; int64_t kl_max_pos = -1; } cmp;
    auto compare_row = [&](const float * cur, int64_t pos) {
        if (fread(refrow.data(), sizeof(float), n_vocab, fr) != (size_t) n_vocab) { fprintf(stderr, "ref logits too short\n"); exit(1); }
        double mr = -1e300, mc = -1e300; int ar = 0, ac = 0; double mad = 0;
        for (int v = 0; v < n_vocab; ++v) {
            if (refrow[v] > mr) { mr = refrow[v]; ar = v; }
            if (cur[v] > mc) { mc = cur[v]; ac = v; }
            mad = std::max(mad, (double) std::fabs(refrow[v] - cur[v]));
        }
        double zr = 0, zc = 0;
        for (int v = 0; v < n_vocab; ++v) { zr += std::exp(refrow[v] - mr); zc += std::exp(cur[v] - mc); }
        const double lzr = mr + std::log(zr), lzc = mc + std::log(zc);
        double kl = 0;
        for (int v = 0; v < n_vocab; ++v) {
            const double lp = refrow[v] - lzr, lq = cur[v] - lzc;
            kl += std::exp(lp) * (lp - lq);
        }
        cmp.n++; cmp.top1 += ar == ac; cmp.kl += kl; cmp.max_abs = std::max(cmp.max_abs, mad);
        if (kl > cmp.kl_max) { cmp.kl_max = kl; cmp.kl_max_pos = pos; }
        if (pos >= 4100) { cmp.n_far++; cmp.top1_far += ar == ac; cmp.kl_far += kl; }
    };

    const int64_t t0 = ggml_time_us();
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    for (int p0 = 0; p0 < n_prompt; p0 += n_batch) {
        const int n = std::min(n_batch, n_prompt - p0);
        batch.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            batch.token[i]     = toks[p0 + i];
            batch.pos[i]       = p0 + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = !last_only || (p0 + i == n_prompt - 1);
        }
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed at %d\n", p0); return 1; }
        for (int i = 0; i < n; ++i) {
            if (batch.logits[i]) {
                if (fl) fwrite(llama_get_logits_ith(ctx, i), sizeof(float), n_vocab, fl);
                if (fr) compare_row(llama_get_logits_ith(ctx, i), p0 + i);
            }
        }
    }
    const int64_t t1 = ggml_time_us();
    fprintf(stderr, "prompt: %d tokens in %.2f s (%.1f tok/s)\n", n_prompt, (t1 - t0)/1e6, n_prompt/((t1 - t0)/1e6));
    if (fl) fclose(fl);
    if (fr) {
        fclose(fr);
        fprintf(stderr, "ref-compare: n=%lld top1=%.4f (%lld) mean_KL=%.3e max_KL=%.3e@%lld max_abs=%.4f | pos>=4100: n=%lld top1=%.4f mean_KL=%.3e\n",
            (long long) cmp.n, cmp.n ? (double) cmp.top1/cmp.n : 0.0, (long long) cmp.top1, cmp.n ? cmp.kl/cmp.n : 0.0, cmp.kl_max, (long long) cmp.kl_max_pos, cmp.max_abs,
            (long long) cmp.n_far, cmp.n_far ? (double) cmp.top1_far/cmp.n_far : 0.0, cmp.n_far ? cmp.kl_far/cmp.n_far : 0.0);
    }

    if (n_gen > 0) {
        FILE * fg  = out_gen.empty() ? nullptr : fopen(out_gen.c_str(), "wb");
        std::vector<int32_t> rg;
        if (!ref_gen.empty()) {
            FILE * f = fopen(ref_gen.c_str(), "rb");
            int32_t x;
            while (f && fread(&x, sizeof(int32_t), 1, f) == 1) rg.push_back(x);
            if (f) fclose(f);
        }
        int first_div = -1; float div_margin = 0;
        FILE * fgl = out_gen_logits.empty() ? nullptr : fopen(out_gen_logits.c_str(), "wb");
        const float * lg = llama_get_logits_ith(ctx, -1);
        std::vector<float> cur(lg, lg + n_vocab);
        const int64_t t2 = ggml_time_us();
        for (int s = 0; s < n_gen; ++s) {
            if (profile && s == 2) { g_prof.on = true; g_prof.last = ggml_time_us(); }
            if (fgl) fwrite(cur.data(), sizeof(float), n_vocab, fgl);
            int best = 0;
            for (int v = 1; v < n_vocab; ++v) if (cur[v] > cur[best]) best = v;
            if (fg) fwrite(&best, sizeof(int32_t), 1, fg);
            if (first_div < 0 && s < (int) rg.size() && rg[s] != best) {
                first_div = s;
                div_margin = cur[best] - cur[rg[s]];
            }
            batch.n_tokens = 1;
            batch.token[0] = best; batch.pos[0] = n_prompt + s; batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = true;
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed (gen %d)\n", s); return 1; }
            lg = llama_get_logits_ith(ctx, -1);
            cur.assign(lg, lg + n_vocab);
        }
        const int64_t t3 = ggml_time_us();
        fprintf(stderr, "gen: %d tokens in %.2f s (%.2f tok/s)\n", n_gen, (t3 - t2)/1e6, n_gen/((t3 - t2)/1e6));
        if (fg) fclose(fg);
        if (fgl) fclose(fgl);
        if (!rg.empty()) {
            fprintf(stderr, "ref-gen: %s (first divergence %d, own logit margin there %.4f)\n",
                first_div < 0 ? "identical" : "diverges", first_div, div_margin);
        }
        if (profile) {
            std::vector<std::pair<int64_t, std::string>> v;
            int64_t tot = 0;
            for (auto & kv : g_prof.acc) { v.push_back({kv.second.first, kv.first + " x" + std::to_string(kv.second.second)}); tot += kv.second.first; }
            std::sort(v.rbegin(), v.rend());
            fprintf(stderr, "profile (%d steps, %.1f ms/step attributed):\n", n_gen - 2, tot/1e3/(n_gen - 2));
            for (size_t i = 0; i < v.size() && i < 30; ++i) fprintf(stderr, "  %8.2f ms/step  %s\n", v[i].first/1e3/(n_gen - 2), v[i].second.c_str());
        }
    }

    if (ld.f) fclose(ld.f);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
