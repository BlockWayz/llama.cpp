// llama-volundr-state-test: sequence state save/restore check for the Volundr GGUF port (what llama-server's context
// checkpoints and prompt cache rely on). Tokenizes a text file, takes A = the first --n-a tokens and B = the next --n-b,
// and compares the logits of every position of B:
//   ref     : decode A, save the seq state, decode B
//   partial : restore the PARTIAL_ONLY state (recurrent KDA/conv state + Engram previous ids, as a server checkpoint),
//             drop the KV cells of B, decode B again                      -> must be bit-identical to ref
//   full    : restore the full seq state (KV cache with the BCSA indexer keys + recurrent state) into a zeroed
//             cache, decode B again                                       -> must be bit-identical to ref
//   no-prev : like partial, but with the Engram previous ids in the saved state zeroed (negative control: shows how
//             much the Engram context at the restore boundary matters)
#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<float> decode(llama_context * ctx, const std::vector<llama_token> & toks, int p0, int n, bool logits) {
    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        batch.token[i]     = toks[p0 + i];
        batch.pos[i]       = p0 + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = logits;
    }
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed at %d\n", p0);
        exit(1);
    }
    llama_batch_free(batch);

    std::vector<float> res;
    if (logits) {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        res.resize((size_t) n*n_vocab);
        for (int i = 0; i < n; ++i) {
            memcpy(res.data() + (size_t) i*n_vocab, llama_get_logits_ith(ctx, i), n_vocab*sizeof(float));
        }
    }
    return res;
}

static void report(const char * name, const std::vector<float> & ref, const std::vector<float> & cur, int n_vocab) {
    const int n = (int) (ref.size()/n_vocab);
    double dmax = 0;
    int n_top1 = 0;
    std::string per_pos;
    for (int i = 0; i < n; ++i) {
        const float * a = ref.data() + (size_t) i*n_vocab;
        const float * b = cur.data() + (size_t) i*n_vocab;
        double d = 0;
        int ia = 0, ib = 0;
        for (int v = 0; v < n_vocab; ++v) {
            d = std::max(d, (double) std::fabs(a[v] - b[v]));
            ia = a[v] > a[ia] ? v : ia;
            ib = b[v] > b[ib] ? v : ib;
        }
        dmax = std::max(dmax, d);
        n_top1 += ia == ib;
        if (i < 6) {
            char buf[32];
            snprintf(buf, sizeof(buf), " %.2e", d);
            per_pos += buf;
        }
    }
    printf("%-8s: max abs logit diff %.6e, top-1 agreement %d/%d, first positions:%s\n", name, dmax, n_top1, n, per_pos.c_str());
}

int main(int argc, char ** argv) {
    std::string model_path, text_path;
    int n_gpu_layers = 0, n_a = 6000, n_b = 64, n_ubatch = 512, n_threads = 16;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return std::string(argv[++i]); };
        if      (a == "-m")    model_path = next();
        else if (a == "-f")    text_path = next();
        else if (a == "-ngl")  n_gpu_layers = atoi(next().c_str());
        else if (a == "-ub")   n_ubatch = atoi(next().c_str());
        else if (a == "-t")    n_threads = atoi(next().c_str());
        else if (a == "--n-a") n_a = atoi(next().c_str());
        else if (a == "--n-b") n_b = atoi(next().c_str());
        else { fprintf(stderr, "usage: %s -m model.gguf -f text.txt [-ngl N] [--n-a 6000] [--n-b 64] [-ub 512] [-t 16]\n", argv[0]); return 1; }
    }

    std::string text;
    {
        FILE * f = fopen(text_path.c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", text_path.c_str()); return 1; }
        char buf[65536];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
        fclose(f);
    }

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> toks(text.size() + 16);
    const int n_tok = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), true, false);
    if (n_tok < n_a + n_b) { fprintf(stderr, "text too short: %d tokens < %d\n", n_tok, n_a + n_b); return 1; }
    toks.resize(n_tok);

    auto cparams = llama_context_default_params();
    cparams.n_ctx           = ((n_a + n_b + 255)/256 + 1)*256;
    cparams.n_batch         = std::max(n_a, n_b);
    cparams.n_ubatch        = n_ubatch;
    cparams.n_seq_max       = 1;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    llama_memory_t mem = llama_get_memory(ctx);

    // ref
    decode(ctx, toks, 0, n_a, false);

    std::vector<uint8_t> st_part(llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    llama_state_seq_get_data_ext(ctx, st_part.data(), st_part.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    std::vector<uint8_t> st_full(llama_state_seq_get_size_ext(ctx, 0, 0));
    llama_state_seq_get_data_ext(ctx, st_full.data(), st_full.size(), 0, 0);
    printf("A = %d tokens, B = %d tokens; partial state %.2f MiB, full state %.2f MiB\n", n_a, n_b, st_part.size()/1048576.0, st_full.size()/1048576.0);

    const std::vector<float> ref = decode(ctx, toks, n_a, n_b, true);

    // partial (checkpoint) restore
    if (llama_state_seq_set_data_ext(ctx, st_part.data(), st_part.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != st_part.size()) { fprintf(stderr, "partial restore failed\n"); return 1; }
    if (!llama_memory_seq_rm(mem, 0, n_a, -1)) { fprintf(stderr, "seq_rm failed\n"); return 1; }
    report("partial", ref, decode(ctx, toks, n_a, n_b, true), n_vocab);

    // full restore into a zeroed cache (seq_rm alone keeps the cell data, so a field missing from the state would go unnoticed)
    llama_memory_clear(mem, true);
    if (llama_state_seq_set_data_ext(ctx, st_full.data(), st_full.size(), 0, 0) != st_full.size()) { fprintf(stderr, "full restore failed\n"); return 1; }
    report("full", ref, decode(ctx, toks, n_a, n_b, true), n_vocab);

    // negative control: zero the Engram previous ids (stored as id + 1 at the end of the recurrent "r" rows)
    {
        const float prev[2] = { (float) toks[n_a - 2] + 1.0f, (float) toks[n_a - 1] + 1.0f };
        int n_found = 0;
        for (size_t off = 0; off + sizeof(prev) <= st_part.size(); off += 1) {
            if (memcmp(st_part.data() + off, prev, sizeof(prev)) == 0) {
                memset(st_part.data() + off, 0, sizeof(prev));
                n_found++;
            }
        }
        printf("no-prev : zeroed %d Engram previous-id pairs (%g, %g) in the partial state\n", n_found, prev[0], prev[1]);
        if (n_found > 0) {
            llama_state_seq_set_data_ext(ctx, st_part.data(), st_part.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(mem, 0, n_a, -1);
            report("no-prev", ref, decode(ctx, toks, n_a, n_b, true), n_vocab);
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
