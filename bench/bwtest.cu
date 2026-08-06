#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA error: %s (%s:%d)\n",cudaGetErrorString(e_),__FILE__,__LINE__); \
    return 1;} } while(0)

__global__ void copyK(const float4* __restrict__ src, float4* __restrict__ dst, size_t n){
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

int main(){
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU         : %s\n", prop.name);
    printf("Compute cap : sm_%d%d\n", prop.major, prop.minor);
    printf("SMs         : %d\n", prop.multiProcessorCount);
    printf("Global mem  : %.2f GiB\n", prop.totalGlobalMem / 1073741824.0);
    printf("Shared/block: %zu KiB\n", prop.sharedMemPerBlock / 1024);

    const size_t bytes = 512ull << 20;              // 512 MiB per buffer
    const size_t n     = bytes / sizeof(float4);
    float4 *d_src = nullptr, *d_dst = nullptr;
    CHECK(cudaMalloc(&d_src, bytes));
    CHECK(cudaMalloc(&d_dst, bytes));
    CHECK(cudaMemset(d_src, 1, bytes));

    const int block = 256, iters = 50;
    const int grid  = (int)((n + block - 1) / block);

    copyK<<<grid, block>>>(d_src, d_dst, n);        // warm-up
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) copyK<<<grid, block>>>(d_src, d_dst, n);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));

    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, t0, t1));
    const double moved = 2.0 * bytes * iters;       // one read + one write per element
    printf("\nMEASURED PEAK BW : %.1f GB/s  (%.3f ms/iter)\n",
           moved / (ms * 1e-3) / 1e9, ms / iters);

    cudaFree(d_src); cudaFree(d_dst);
    return 0;
}
