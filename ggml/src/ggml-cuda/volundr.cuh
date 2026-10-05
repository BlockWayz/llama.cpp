#include "common.cuh"

// Agens Volundr ops: Engram n-gram hash and the mHC Sinkhorn projection
void ggml_cuda_op_ngram_hash(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
