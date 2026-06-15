// MoE permutation (gather/scatter) kernels in ROCm/HIP -- the AMD counterpart
// of moe_permute.cu. The kernel BODIES are identical to CUDA; this file exists to
// make the (small) platform differences explicit, as discussed in
// docs/moe/kernels.md and docs/performance/cuda-hip-track.md.
//
// Build (standalone test):
//   hipcc -O3 -DMOE_PERMUTE_STANDALONE moe_permute_hip.cpp -o moe_permute_hip
//   ./moe_permute_hip
//
// What differs from CUDA, and why it matters:
//   * warpSize == 64 on CDNA (MI300) vs 32 on NVIDIA. Any warp-level reduction
//     or shuffle MUST use warpSize, never a hardcoded 32/16. (Not needed in the
//     simple copies below, but it's the #1 portability trap -- see the reduction
//     example in docs/performance/cuda-hip-track.md.)
//   * __shared__ memory is "LDS" on AMD; sizing/bank-conflict tuning differs.
//   * Launch via hipLaunchKernelGGL (or the <<<>>> syntax, which hipcc accepts).
//   * A block of 256 threads is 4 wavefronts on AMD vs 8 warps on NVIDIA, so the
//     ideal block size / occupancy trade-off is different -- parameterize it.

#include <cstdio>
#include <hip/hip_runtime.h>

__global__ void gather_rows(const float* __restrict__ src,
                            float* __restrict__ dst,
                            const int* __restrict__ row_map,
                            int d) {
    int out_row = blockIdx.x;
    int in_row  = row_map[out_row];
    for (int c = threadIdx.x; c < d; c += blockDim.x)   // coalesced over d
        dst[out_row * d + c] = src[in_row * d + c];
}

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
int main() {
    const int n = 4, d = 3;
    float h_src[n * d];
    for (int i = 0; i < n * d; ++i) h_src[i] = (float)i;
    int h_map[n] = {3, 1, 0, 2};

    float *d_src, *d_dst; int *d_map;
    hipMalloc(&d_src, sizeof(h_src));
    hipMalloc(&d_dst, sizeof(h_src));
    hipMalloc(&d_map, sizeof(h_map));
    hipMemcpy(d_src, h_src, sizeof(h_src), hipMemcpyHostToDevice);
    hipMemcpy(d_map, h_map, sizeof(h_map), hipMemcpyHostToDevice);

    // Block size chosen with wavefront=64 in mind (a multiple of warpSize).
    const int block = 256;   // = 4 wavefronts on CDNA
    hipLaunchKernelGGL(gather_rows, dim3(n), dim3(block), 0, 0, d_src, d_dst, d_map, d);
    hipDeviceSynchronize();

    float h_dst[n * d];
    hipMemcpy(h_dst, d_dst, sizeof(h_dst), hipMemcpyDeviceToHost);
    printf("gathered rows (expect rows 3,1,0,2):\n");
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < d; ++c) printf("%4.0f ", h_dst[r * d + c]);
        printf("\n");
    }
    printf("device warpSize is 64 on CDNA; tune block size accordingly.\n");
    hipFree(d_src); hipFree(d_dst); hipFree(d_map);
    return 0;
}
#endif
