// ===========================================================================
// THREE EXPERIMENTS
//
// Each isolates one GPU performance rule by changing ONE thing and measuring.
// Read the prediction before each result and see whether the hardware agrees --
// the point is to build intuition you can defend, not numbers you memorised.
//
//   1. COALESCING  -- how threads map to addresses
//   2. DIVERGENCE  -- what a branch costs inside a warp
//   3. PRECISION   -- why fp16 is a performance feature, not a storage detail
//
// Build:  ./build.sh   then   ./build/experiments
// ===========================================================================

#include <cuda_fp16.h>

#include <cstdio>
#include <vector>

#include "tinyci/cuda_util.h"

using namespace tinyci;

namespace {

constexpr int kIters = 30;

void header(const char* n, const char* title, const char* predict) {
    std::printf("\n\033[1m=== %s. %s ===\033[0m\n%s\n\n", n, title, predict);
}

void row(const char* label, float ms, double bytes) {
    std::printf("  %-34s %8.3f ms   %7.1f GB/s\n", label, ms, gbPerSec(bytes, ms));
}

// ---------------------------------------------------------------------------
// 1. COALESCING
//
// Both kernels touch every element exactly once and move identical bytes. The
// ONLY difference is which element a given thread handles.
//
// Treat the buffer as a W x H grid:
//   row-major  -- thread (x,y) touches [y*W + x]. Consecutive threads in x land
//                 on consecutive addresses, so a warp's 32 loads fall inside a
//                 few 32-byte sectors and the hardware merges them.
//   col-major  -- thread (x,y) touches [x*H + y]. Consecutive threads are now H
//                 elements apart, so each of the 32 lanes needs its own memory
//                 transaction.
// ---------------------------------------------------------------------------
__global__ void kRowMajor(const float4* __restrict__ src, float4* __restrict__ dst, int W, int H) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int i = y * W + x;
    dst[i] = src[i];
}

__global__ void kColMajor(const float4* __restrict__ src, float4* __restrict__ dst, int W, int H) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int i = x * H + y;
    dst[i] = src[i];
}

// A bijection with NO spatial locality whatsoever. Multiplying by an odd constant
// modulo a power of two is a permutation, so every element is still touched
// exactly once and the byte count is unchanged -- but neighbouring lanes land in
// unrelated sectors and no other warp reuses them, so L2 cannot rescue it.
__global__ void kPermuted(const float4* __restrict__ src, float4* __restrict__ dst, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned j = (i * 2654435761u) & (n - 1);
    dst[j] = src[j];
}

void experimentCoalescing() {
    header("1", "COALESCING",
           "  Identical work, identical bytes -- only the thread->address mapping changes.\n"
           "  Three patterns: contiguous, transposed, and a locality-free permutation.\n"
           "  WATCH the middle one: it is scattered per warp, yet barely slower. Why?");

    const int W = 4096, H = 4096;
    const std::size_t n = static_cast<std::size_t>(W) * H;   // 2^24, a power of two
    const double bytes = 2.0 * n * sizeof(float4);

    DeviceBuffer<float4> src(n), dst(n);
    CUDA_CHECK(cudaMemset(src.get(), 0, src.bytes()));

    const dim3 block2(32, 8);
    const dim3 grid2((W + block2.x - 1) / block2.x, (H + block2.y - 1) / block2.y);
    const int block1 = 256;
    const int grid1  = gridFor(n, block1);

    CudaTimer t;

    kRowMajor<<<grid2, block2>>>(src.get(), dst.get(), W, H);
    CUDA_CHECK_KERNEL("warm-up");

    t.start();
    for (int i = 0; i < kIters; ++i) kRowMajor<<<grid2, block2>>>(src.get(), dst.get(), W, H);
    const float msRow = t.stop() / kIters;

    t.start();
    for (int i = 0; i < kIters; ++i) kColMajor<<<grid2, block2>>>(src.get(), dst.get(), W, H);
    const float msCol = t.stop() / kIters;

    t.start();
    for (int i = 0; i < kIters; ++i) kPermuted<<<grid1, block1>>>(src.get(), dst.get(), (unsigned)n);
    const float msPerm = t.stop() / kIters;

    row("contiguous    [y*W + x]", msRow, bytes);
    row("transposed    [x*H + y]", msCol, bytes);
    row("permuted      (no locality)", msPerm, bytes);

    std::printf("\n  -> transposed %.2fx slower, permuted %.1fx slower.\n",
                msCol / msRow, msPerm / msRow);
    std::printf("\n     The gap between those two is the lesson. A transposed access IS\n"
                "     scattered within each warp -- 32 lanes, 32 sectors -- but the sectors it\n"
                "     touches are reused almost immediately by neighbouring warps, and a modern\n"
                "     GPU's L2 is large enough to hold them. The pattern is scattered in the\n"
                "     small and dense in the aggregate, so L2 absorbs most of the damage.\n"
                "\n     The permutation destroys reuse as well as locality, and there the\n"
                "     penalty is real: each 32-byte sector fetched delivers one useful 16-byte\n"
                "     pixel, so you pay for bytes you never use.\n"
                "\n     Takeaway for an interview: coalescing still matters, but on current\n"
                "     hardware the thing that actually kills you is losing REUSE. Quoting the\n"
                "     textbook 10x transpose penalty would be repeating a claim from an era of\n"
                "     much smaller caches -- measure it on the machine you have.\n");
}

// ---------------------------------------------------------------------------
// 2. DIVERGENCE
//
// A warp's 32 threads share one instruction pointer. When they disagree about a
// branch, the hardware runs BOTH sides -- each with the non-participating lanes
// disabled -- and you pay for both.
//
// Both kernels execute the same two code paths and the same total instructions.
// The only difference is WHO takes which branch:
//   uniform   -- keyed on blockIdx, so an entire block (and every warp in it)
//                agrees. The branch is free.
//   divergent -- keyed on threadIdx, so lanes alternate and EVERY warp splits.
//
// Deliberately compute-heavy: on a memory-bound kernel the extra ALU work would
// hide behind memory latency and the effect would not show.
// ---------------------------------------------------------------------------
// The two paths must be STRUCTURALLY different, not merely differently
// parameterised. An earlier version used fmaf() with different constants in each
// path; the compiler simply hoisted the constants into a select and ran ONE loop,
// eliminating the divergence entirely and making the experiment measure nothing.
// sin and cos have comparable cost but cannot be merged that way.
__device__ __forceinline__ float pathA(float v) {
    for (int k = 0; k < 32; ++k) v = __sinf(v) + 1.0f;
    return v;
}
__device__ __forceinline__ float pathB(float v) {
    for (int k = 0; k < 32; ++k) v = __cosf(v) + 1.0f;
    return v;
}

__global__ void kUniformBranch(const float* __restrict__ src, float* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = src[i];
    v = (blockIdx.x & 1) ? pathA(v) : pathB(v);   // whole warp agrees
    dst[i] = v;
}

__global__ void kDivergentBranch(const float* __restrict__ src, float* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = src[i];
    v = (threadIdx.x & 1) ? pathA(v) : pathB(v);  // lanes alternate -> warp splits
    dst[i] = v;
}

void experimentDivergence() {
    header("2", "BRANCH DIVERGENCE",
           "  Same two code paths, same instruction count. Only WHO branches changes.\n"
           "  PREDICTION: keying the branch on threadIdx makes every warp execute both\n"
           "  sides serially -- close to 2x slower than keying it on blockIdx.");

    const std::size_t n = 32u << 20;   // 32 M floats
    DeviceBuffer<float> src(n), dst(n);
    CUDA_CHECK(cudaMemset(src.get(), 0, src.bytes()));

    const int block = 256;
    const int grid  = gridFor(n, block);
    CudaTimer t;

    kUniformBranch<<<grid, block>>>(src.get(), dst.get(), (int)n);
    CUDA_CHECK_KERNEL("warm-up");

    t.start();
    for (int i = 0; i < kIters; ++i) kUniformBranch<<<grid, block>>>(src.get(), dst.get(), (int)n);
    const float msUniform = t.stop() / kIters;

    t.start();
    for (int i = 0; i < kIters; ++i) kDivergentBranch<<<grid, block>>>(src.get(), dst.get(), (int)n);
    const float msDivergent = t.stop() / kIters;

    std::printf("  %-34s %8.3f ms\n", "uniform branch   (blockIdx & 1)", msUniform);
    std::printf("  %-34s %8.3f ms\n", "divergent branch (threadIdx & 1)", msDivergent);
    std::printf("\n  -> divergence costs %.2fx.\n", msDivergent / msUniform);
    std::printf("     A branch is free when the whole warp agrees -- `if (x < width)` costs\n"
                "     nothing in the interior and only bites at the edge. A branch keyed on\n"
                "     pixel parity splits every warp. This is why a fixed-kernel demosaic like\n"
                "     Malvar-He-Cutler maps better to a GPU than an adaptive one that picks a\n"
                "     direction per pixel, even though the adaptive method scores higher.\n");
}

// ---------------------------------------------------------------------------
// 3. PRECISION AS A PERFORMANCE FEATURE
//
// Identical arithmetic on identical pixel counts. Only the storage format
// changes: RGBA fp32 is 16 bytes per pixel, RGBA fp16 is 8.
//
// On a bandwidth-bound kernel, runtime tracks bytes moved -- so halving the
// format should nearly halve the time. This is WHY an imaging framework exposes
// pixel formats at all: not to save disk, to go faster.
// ---------------------------------------------------------------------------
struct __align__(8) half4 { __half x, y, z, w; };

__global__ void kScaleF32(const float4* __restrict__ src, float4* __restrict__ dst,
                          int n, float k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 v = src[i];
    v.x *= k; v.y *= k; v.z *= k;
    dst[i] = v;
}

__global__ void kScaleF16(const half4* __restrict__ src, half4* __restrict__ dst,
                          int n, float k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    half4 v = src[i];
    v.x = __float2half(__half2float(v.x) * k);
    v.y = __float2half(__half2float(v.y) * k);
    v.z = __float2half(__half2float(v.z) * k);
    dst[i] = v;
}

void experimentPrecision() {
    header("3", "PRECISION IS A PERFORMANCE FEATURE",
           "  Same pixels, same math. RGBA fp32 = 16 B/px, RGBA fp16 = 8 B/px.\n"
           "  PREDICTION: on a bandwidth-bound kernel, half the bytes is close to half\n"
           "  the time -- the arithmetic was never the limit.");

    const std::size_t n = 24u << 20;   // ~24 M pixels
    CudaTimer t;
    float ms32 = 0.0f, ms16 = 0.0f;
    double bytes32 = 2.0 * n * sizeof(float4);
    double bytes16 = 2.0 * n * sizeof(half4);

    {
        DeviceBuffer<float4> src(n), dst(n);
        CUDA_CHECK(cudaMemset(src.get(), 0, src.bytes()));
        const int block = 256, grid = gridFor(n, block);
        kScaleF32<<<grid, block>>>(src.get(), dst.get(), (int)n, 1.5f);
        CUDA_CHECK_KERNEL("warm-up f32");
        t.start();
        for (int i = 0; i < kIters; ++i)
            kScaleF32<<<grid, block>>>(src.get(), dst.get(), (int)n, 1.5f);
        ms32 = t.stop() / kIters;
    }
    {
        DeviceBuffer<half4> src(n), dst(n);
        CUDA_CHECK(cudaMemset(src.get(), 0, src.bytes()));
        const int block = 256, grid = gridFor(n, block);
        kScaleF16<<<grid, block>>>(src.get(), dst.get(), (int)n, 1.5f);
        CUDA_CHECK_KERNEL("warm-up f16");
        t.start();
        for (int i = 0; i < kIters; ++i)
            kScaleF16<<<grid, block>>>(src.get(), dst.get(), (int)n, 1.5f);
        ms16 = t.stop() / kIters;
    }

    row("RGBA fp32  (16 B/px)", ms32, bytes32);
    row("RGBA fp16  ( 8 B/px)", ms16, bytes16);
    std::printf("\n  -> fp16 is %.2fx faster. Note both reach a SIMILAR GB/s: neither kernel\n",
                ms32 / ms16);
    std::printf("     is more efficient per byte. fp16 wins purely by moving fewer bytes --\n"
                "     the same mechanism as kernel fusion, applied to format instead of passes.\n");
    std::printf("     Caveat: fp16 carries ~10 bits of mantissa. Fine display-referred, marginal\n"
                "     for deep scene-referred highlights, which is a real pipeline decision.\n");
}

}  // namespace

int main() try {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s  sm_%d%d  %d SMs  |  measured peak from bwtest: 377.5 GB/s\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    experimentCoalescing();
    experimentDivergence();
    experimentPrecision();

    std::printf("\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
