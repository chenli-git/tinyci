// ===========================================================================
// THE FUSION EXPERIMENT
//
// This is the central measurement of the whole project, and it needs no graph
// machinery: run three pointwise stages as three separate kernels, then again as
// one fused kernel, and time both.
//
// The prediction, from first principles, before any code runs:
//
//   A pointwise stage on RGBA fp32 reads 16 B and writes 16 B per pixel. The
//   arithmetic is a handful of FLOPs -- well under 1 FLOP/byte, an order of
//   magnitude below the roofline ridge point. So every stage costs exactly one
//   read + one write of the image, no matter what math it does.
//
//   3 separate stages = 3 round-trips to DRAM.
//   1 fused stage     = 1 round-trip.
//   Expected speedup  ~3x, and the fused kernel should approach peak bandwidth.
//
// If that prediction holds, it is the empirical argument for why Core Image
// concatenates kernels rather than merely optimising them.
//
// Build:  ./build.sh   then   ./build/fusion
// ===========================================================================

#include <cstdio>
#include <vector>

#include "tinyci/cuda_util.h"

using namespace tinyci;

// ---------------------------------------------------------------------------
// Stage math, shared by both paths.
//
// __device__ = callable from GPU code only. Marking these inline and reusing
// them in both the separate and fused kernels guarantees the two paths compute
// *identical* arithmetic -- otherwise the comparison measures the wrong thing.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float4 stageColorMatrix(float4 p, const float* __restrict__ M) {
    float4 o;
    o.x = M[0] * p.x + M[1] * p.y + M[2] * p.z;
    o.y = M[3] * p.x + M[4] * p.y + M[5] * p.z;
    o.z = M[6] * p.x + M[7] * p.y + M[8] * p.z;
    o.w = p.w;
    return o;
}

__device__ __forceinline__ float4 stageToneMap(float4 p, float exposure, float invW2) {
    p.x *= exposure; p.y *= exposure; p.z *= exposure;
    const float L = 0.2126f * p.x + 0.7152f * p.y + 0.0722f * p.z;
    if (L > 0.0f) {
        const float Ld = L * (1.0f + L * invW2) / (1.0f + L);
        const float s  = Ld / L;
        p.x *= s; p.y *= s; p.z *= s;
    }
    return p;
}

__device__ __forceinline__ float srgbEncode(float v) {
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    return (v <= 0.0031308f) ? v * 12.92f : 1.055f * __powf(v, 1.0f / 2.4f) - 0.055f;
}

__device__ __forceinline__ float4 stageEncode(float4 p) {
    p.x = srgbEncode(p.x); p.y = srgbEncode(p.y); p.z = srgbEncode(p.z);
    return p;
}

// ---------------------------------------------------------------------------
// PATH A -- three separate kernels. Each one reads the whole image from DRAM and
// writes the whole image back. The intermediates are never looked at by anyone
// except the next kernel, yet they make a full round trip to memory.
// ---------------------------------------------------------------------------

__global__ void kColorMatrix(const float4* __restrict__ src, float4* __restrict__ dst,
                             int n, const float* __restrict__ M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = stageColorMatrix(src[i], M);
}

__global__ void kToneMap(const float4* __restrict__ src, float4* __restrict__ dst,
                         int n, float exposure, float invW2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = stageToneMap(src[i], exposure, invW2);
}

__global__ void kEncode(const float4* __restrict__ src, float4* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = stageEncode(src[i]);
}

// ---------------------------------------------------------------------------
// PATH B -- one fused kernel. Identical arithmetic. The difference is entirely
// that the intermediates stay in REGISTERS instead of travelling to DRAM and
// back twice.
// ---------------------------------------------------------------------------

__global__ void kFused(const float4* __restrict__ src, float4* __restrict__ dst,
                       int n, const float* __restrict__ M, float exposure, float invW2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float4 p = src[i];
    p = stageColorMatrix(p, M);
    p = stageToneMap(p, exposure, invW2);
    p = stageEncode(p);
    dst[i] = p;
}

// ===========================================================================

int main() try {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // 24 MP, the size of a real camera frame. Large enough that caches cannot
    // hide the traffic, which is the entire point of the measurement.
    const int width = 6000, height = 4000;
    const std::size_t n = static_cast<std::size_t>(width) * height;
    const double imageBytes = static_cast<double>(n) * sizeof(float4);

    std::printf("GPU            : %s (sm_%d%d, %d SMs)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    std::printf("Image          : %dx%d = %.1f MP, RGBA fp32 = %.1f MB\n",
                width, height, n / 1e6, imageBytes / 1e6);

    std::vector<float4> host(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float t = static_cast<float>(i % 1024) / 1024.0f;
        host[i] = make_float4(t, 1.0f - t, 0.5f * t, 1.0f);
    }

    DeviceBuffer<float4> a(n), b(n), c(n);
    a.upload(host);

    const float hM[9] = { 1.05f, -0.03f, -0.02f,
                         -0.01f,  1.02f, -0.01f,
                          0.00f, -0.04f,  1.04f };
    DeviceBuffer<float> dM(9);
    dM.upload(hM, 9);

    const float exposure = 1.2f;
    const float invW2    = 1.0f / (2.0f * 2.0f);

    const int block = 256;
    const int grid  = gridFor(n, block);
    const int iters = 20;

    CudaTimer timer;

    // Warm-up. The first launch pays JIT, context setup and page faults, and can
    // be 10-100x slower than steady state. Timing it would be meaningless.
    kFused<<<grid, block>>>(a.get(), b.get(), (int)n, dM.get(), exposure, invW2);
    CUDA_CHECK_KERNEL("warm-up");

    // ---- PATH A: three kernels, three round-trips --------------------------
    timer.start();
    for (int it = 0; it < iters; ++it) {
        kColorMatrix<<<grid, block>>>(a.get(), b.get(), (int)n, dM.get());
        kToneMap    <<<grid, block>>>(b.get(), c.get(), (int)n, exposure, invW2);
        kEncode     <<<grid, block>>>(c.get(), b.get(), (int)n);
    }
    const float msSeparate = timer.stop() / iters;
    CUDA_CHECK_KERNEL("separate");

    // ---- PATH B: one kernel, one round-trip --------------------------------
    timer.start();
    for (int it = 0; it < iters; ++it)
        kFused<<<grid, block>>>(a.get(), c.get(), (int)n, dM.get(), exposure, invW2);
    const float msFused = timer.stop() / iters;
    CUDA_CHECK_KERNEL("fused");

    // ---- Verify the two paths agree ----------------------------------------
    // A faster kernel that computes something different is not an optimisation.
    std::vector<float4> outSep(n), outFus(n);
    b.download(outSep);
    c.download(outFus);
    double maxDiff = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        maxDiff = fmax(maxDiff, fabs((double)outSep[i].x - outFus[i].x));
        maxDiff = fmax(maxDiff, fabs((double)outSep[i].y - outFus[i].y));
        maxDiff = fmax(maxDiff, fabs((double)outSep[i].z - outFus[i].z));
    }

    // Traffic: each stage reads the image once and writes it once.
    const double bytesSeparate = 3.0 * 2.0 * imageBytes;
    const double bytesFused    = 1.0 * 2.0 * imageBytes;

    std::printf("\n%-22s %10s %12s %10s\n", "path", "ms", "GB/s", "vs peak");
    std::printf("%-22s %10.3f %12.1f %9.0f%%\n", "3 separate kernels",
                msSeparate, gbPerSec(bytesSeparate, msSeparate),
                100.0 * gbPerSec(bytesSeparate, msSeparate) / 377.5);
    std::printf("%-22s %10.3f %12.1f %9.0f%%\n", "1 fused kernel",
                msFused, gbPerSec(bytesFused, msFused),
                100.0 * gbPerSec(bytesFused, msFused) / 377.5);

    std::printf("\nspeedup        : %.2fx   (predicted ~3x from traffic alone)\n",
                msSeparate / msFused);
    std::printf("DRAM traffic   : %.0f MB -> %.0f MB\n",
                bytesSeparate / 1e6, bytesFused / 1e6);
    std::printf("max |diff|     : %.3g   (paths must agree)\n", maxDiff);
    std::printf("\nPeak reference : 377.5 GB/s measured by bwtest. Quote against this,\n"
                "                 not the ~448 GB/s spec figure, which is unreachable.\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
