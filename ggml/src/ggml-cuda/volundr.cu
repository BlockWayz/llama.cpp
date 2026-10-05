#include "volundr.cuh"

// one thread per (token, sequence): 61-bit polynomial hash of the trailing `order` (id + 1) values
struct ngram_hash_params {
    int32_t  order;
    int32_t  prefix;
    int32_t  rows;
    int32_t  n_primes;
    uint64_t primes[8];
};

static __global__ void ngram_hash_kernel(const char * src, int32_t * dst, const ngram_hash_params p,
        const int64_t n, const int64_t ns, const size_t nb0, const size_t nb1, const size_t nbd1) {
    const int64_t ir = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (ir >= n*ns) {
        return;
    }
    const int64_t t = ir % n;
    const int64_t s = ir / n;
    const char * e = src + s*nb1;

    const uint64_t mask61 = 0x1FFFFFFFFFFFFFFFull;
    uint64_t code = 0;
    for (int32_t j = 0; j < p.order; ++j) {
        const int64_t  idx = (int64_t) p.prefix + t - (p.order - 1) + j;
        const uint64_t ev  = (uint64_t) (int64_t) *(const float *) (e + idx*nb0);
        code = (code*p.primes[j % p.n_primes] + ev) & mask61;
    }
    *(int32_t *) ((char *) dst + s*nbd1 + t*sizeof(int32_t)) = (int32_t) (code % (uint64_t) p.rows);
}

void ggml_cuda_op_ngram_hash(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->nb[0] == sizeof(int32_t));

    ngram_hash_params p;
    p.order    = ggml_get_op_params_i32(dst, 0);
    p.prefix   = ggml_get_op_params_i32(dst, 1);
    p.rows     = ggml_get_op_params_i32(dst, 2);
    p.n_primes = ggml_get_op_params_i32(dst, 3);
    for (int i = 0; i < 8; ++i) {
        p.primes[i] = i < p.n_primes ? (uint64_t) (uint32_t) ggml_get_op_params_i32(dst, 4 + i) : 1;
    }

    const int64_t n  = dst->ne[0];
    const int64_t ns = dst->ne[1];
    const int block = 256;
    const int64_t grid = (n*ns + block - 1)/block;
    ngram_hash_kernel<<<grid, block, 0, ctx.stream()>>>((const char *) src0->data, (int32_t *) dst->data, p,
            n, ns, src0->nb[0], src0->nb[1], dst->nb[1]);
}

// one thread per token: M = exp(L - max L), n_iter x (row, column) normalisation, n <= 8
static __global__ void sinkhorn_kernel(const char * src, float * dst, const int n, const int n_iter, const float eps,
        const int64_t nt, const size_t nb0, const size_t nb1, const size_t nbd0, const size_t nbd1, const size_t nbd2) {
    const int64_t t = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= nt) {
        return;
    }
    float M[64];
    const char * lp = src + t*nb1;

    float mx = -INFINITY;
    for (int k = 0; k < n*n; ++k) {
        M[k] = *(const float *) (lp + k*nb0);
        mx = fmaxf(mx, M[k]);
    }
    for (int k = 0; k < n*n; ++k) {
        M[k] = expf(M[k] - mx);
    }
    for (int it = 0; it < n_iter; ++it) {
        for (int i = 0; i < n; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < n; ++j) sum += M[i*n + j];
            sum = fmaxf(sum, eps);
            for (int j = 0; j < n; ++j) M[i*n + j] /= sum;
        }
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < n; ++i) sum += M[i*n + j];
            sum = fmaxf(sum, eps);
            for (int i = 0; i < n; ++i) M[i*n + j] /= sum;
        }
    }
    char * dp = (char *) dst + t*nbd2;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            *(float *) (dp + i*nbd0 + j*nbd1) = M[i*n + j];
        }
    }
}

void ggml_cuda_op_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const int   n      = ggml_get_op_params_i32(dst, 0);
    const int   n_iter = ggml_get_op_params_i32(dst, 1);
    const float eps    = ggml_get_op_params_f32(dst, 2);
    GGML_ASSERT(n >= 1 && n <= 8);

    const int64_t nt = dst->ne[2];
    const int block = 128;
    const int64_t grid = (nt + block - 1)/block;
    sinkhorn_kernel<<<grid, block, 0, ctx.stream()>>>((const char *) src0->data, (float *) dst->data, n, n_iter, eps,
            nt, src0->nb[0], src0->nb[1], dst->nb[0], dst->nb[1], dst->nb[2]);
}
