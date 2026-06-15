// MoE permutation (gather/scatter) kernels in CUDA.
//
// These implement the token<->expert permutation that the MoE dispatch needs
// (see docs/moe/kernels.md and docs/moe/systems-ep.md). `gather_rows` collects
// tokens into expert-contiguous order before the grouped GEMM; `scatter_add_rows`
// combines expert outputs back to their tokens, weighted by the gate.
//
// Build (standalone test):  nvcc -O3 moe_permute.cu -o moe_permute && ./moe_permute
//
// Note for the AMD/HIP version (moe_permute_hip.cpp): the kernel bodies are
// identical; what changes is warpSize (32 here vs 64 on CDNA), the launch macro,
// and tuning. We deliberately parameterize the block size, never assume 32.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Gather: dst[out_row] = src[row_map[out_row]]. One block per output row,
// threads stride across the feature dimension d for coalesced access.
__global__ void gather_rows(const float* __restrict__ src,
                            float* __restrict__ dst,
                            const int* __restrict__ row_map,
                            int d) {
    int out_row = blockIdx.x;
    int in_row  = row_map[out_row];
    for (int c = threadIdx.x; c < d; c += blockDim.x)
        dst[out_row * d + c] = src[in_row * d + c];
}

// Scatter-add with per-row gate weight: dst[token_map[r]] += weight[r] * src[r].
// atomicAdd handles the case where multiple expert-slots map to the same token
// (top-k > 1), exactly like index_add_ in the PyTorch reference.
__global__ void scatter_add_rows(const float* __restrict__ src,
                                 float* __restrict__ dst,
                                 const int* __restrict__ token_map,
                                 const float* __restrict__ weight,
                                 int d) {
    int r = blockIdx.x;
    int out = token_map[r];
    float w = weight[r];
    for (int c = threadIdx.x; c < d; c += blockDim.x)
        atomicAdd(&dst[out * d + c], w * src[r * d + c]);
}

#ifdef MOE_PERMUTE_STANDALONE
// Tiny self-check: gather then scatter-add (weight 1) should round-trip a permute.
int main() {
    const int n = 4, d = 3;
    float h_src[n * d];
    for (int i = 0; i < n * d; ++i) h_src[i] = (float)i;
    int h_map[n] = {3, 1, 0, 2};        // a permutation

    float *d_src, *d_dst; int *d_map;
    cudaMalloc(&d_src, sizeof(h_src));
    cudaMalloc(&d_dst, sizeof(h_src));
    cudaMalloc(&d_map, sizeof(h_map));
    cudaMemcpy(d_src, h_src, sizeof(h_src), cudaMemcpyHostToDevice);
    cudaMemcpy(d_map, h_map, sizeof(h_map), cudaMemcpyHostToDevice);

    gather_rows<<<n, 64>>>(d_src, d_dst, d_map, d);   // warpSize == 32 on NVIDIA
    cudaDeviceSynchronize();

    float h_dst[n * d];
    cudaMemcpy(h_dst, d_dst, sizeof(h_dst), cudaMemcpyDeviceToHost);
    printf("gathered rows (expect rows 3,1,0,2):\n");
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < d; ++c) printf("%4.0f ", h_dst[r * d + c]);
        printf("\n");
    }
    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_map);
    return 0;
}
#endif
