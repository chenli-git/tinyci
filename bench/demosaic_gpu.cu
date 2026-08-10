// ===========================================================================
// YOUR FIRST CUDA KERNEL -- port demosaicBilinear to the GPU.
//
// Everything except the two functions marked TODO is written for you: the test
// data, the host reference, the upload/download, the timing, and an exact
// correctness check. Fill in the two TODOs and run it.
//
//   ./build.sh && ./build/demosaic_gpu
//
// The target is BIT-IDENTICAL output to your host implementation (max diff 0),
// plus a measured speedup. Same algorithm, same arithmetic, different machine.
//
// Why this kernel and not something easier: you already own the algorithm
// completely, so the only new thing here is CUDA. It is also genuinely 2-D --
// the CFA colour depends on x&1 and y&1 -- so you have to think about thread
// indexing rather than treating the image as a flat array like fusion.cu does.
// ===========================================================================

#include <chrono>
#include <cstdio>
#include <cmath>
#include <vector>

#include "tinyci/cuda_util.h"
#include "tinyci/image.h"
#include "tinyci/pipeline.h"

using namespace tinyci;

// ---------------------------------------------------------------------------
// PROVIDED. Device copies of the two helpers your host code uses.
//
// __device__ means "compiled for the GPU, callable only from GPU code".
// __forceinline__ matters here: these are called 3x per pixel, and a real
// function call on a GPU costs far more than the handful of instructions
// inside. Inlining them is not a micro-optimisation, it is the difference
// between this being fast and being pointless.
// ---------------------------------------------------------------------------

// Mirrors tinyci::cfaColorAt. CFAPattern is passed as an int because enums are
// clumsier to hand to a kernel: 0=RGGB, 1=BGGR, 2=GRBG, 3=GBRG.
// Returns 0=R, 1=G, 2=B. Safe at negative coordinates -- it only reads parity.
__device__ __forceinline__ int cfaColorAtDev(int cfa, int x, int y) {
    const int i = (y & 1) * 2 + (x & 1);   // 0=TL 1=TR 2=BL 3=BR
    switch (cfa) {
        case 0: { const int m[4] = {0, 1, 1, 2}; return m[i]; }   // RGGB
        case 1: { const int m[4] = {2, 1, 1, 0}; return m[i]; }   // BGGR
        case 2: { const int m[4] = {1, 0, 2, 1}; return m[i]; }   // GRBG
        default:{ const int m[4] = {1, 2, 0, 1}; return m[i]; }   // GBRG
    }
}

// Mirrors Image<T>::atClamped for a 1-channel mosaic.
__device__ __forceinline__ float sampleClamped(const float* __restrict__ m,
                                               int W, int H, int x, int y) {
    x = x < 0 ? 0 : (x >= W ? W - 1 : x);
    y = y < 0 ? 0 : (y >= H ? H - 1 : y);
    return m[static_cast<long long>(y) * W + x];
}

// ---------------------------------------------------------------------------
// TODO 1 of 2. Estimate colour c at (x,y) from same-coloured neighbours.
//
// This is a DIRECT PORT of your host interpolatedAt(). Same four cases, same
// order. Two mechanical differences:
//   - use sampleClamped(m, W, H, ...) instead of img.atClamped(...)
//   - use cfaColorAtDev(cfa, ...) instead of cfaColorAt(...)
//
// Remember the shape:
//   this site already measures c            -> return it
//   c lies left and right  (horiz)          -> average 2
//   c lies above and below (vert)           -> average 2
//   BOTH axes                               -> green at a red/blue site, average 4
//   neither                                 -> diagonals only, average 4
//
// And remember why the both-axes case has to be tested FIRST: green occupies a
// quincunx, so at any non-green site it lies on both axes at once. Falling into
// the plain `horiz` branch would average 2 greens instead of 4.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float interpolateAtDev(const float* __restrict__ m,
                                                  int W, int H, int cfa,
                                                  int x, int y, int c) {
    // TODO: implement
    (void)m; (void)W; (void)H; (void)cfa; (void)x; (void)y; (void)c;
    return 0.0f;
}

// ---------------------------------------------------------------------------
// TODO 2 of 2. The kernel itself.
//
// Steps:
//   1. Compute this thread's pixel coordinates. Unlike fusion.cu, which used a
//      flat 1-D index, you need BOTH:
//          const int x = blockIdx.x * blockDim.x + threadIdx.x;
//          const int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//   2. Guard. The grid is launched in whole blocks and almost never divides the
//      image evenly, so surplus threads MUST return before touching memory:
//          if (x >= W || y >= H) return;
//      Without this you write out of bounds. It will not always crash -- it will
//      sometimes just corrupt a neighbouring allocation, which is worse.
//      compute-sanitizer catches it instantly.
//
//   3. Write all three channels for this pixel:
//          const long long o = (static_cast<long long>(y) * W + x) * 3;
//          out[o + 0] = interpolateAtDev(mosaic, W, H, cfa, x, y, 0);
//          out[o + 1] = ...  1 ...
//          out[o + 2] = ...  2 ...
//
// Note on coalescing: consecutive threads differ in x, so consecutive threads
// write consecutive pixels -- 3 interleaved floats apart. Not perfectly
// coalesced (12-byte stride, not 4 or 16), but close enough that the reads
// dominate. This is exactly the float3-vs-float4 alignment issue from the
// primer, and it is a fair thing to raise about your own code.
// ---------------------------------------------------------------------------
__global__ void kDemosaicBilinear(const float* __restrict__ mosaic,
                                  float* __restrict__ out,
                                  int W, int H, int cfa) {
    // TODO: implement
    (void)mosaic; (void)out; (void)W; (void)H; (void)cfa;
}

// ===========================================================================
// Everything below is the harness. You should not need to change it.
// ===========================================================================

namespace {

// Deterministic synthetic mosaic. Structured enough that a wrong case in the
// dispatch produces a visibly wrong number rather than plausible noise.
ImageF32 makeMosaic(int W, int H) {
    ImageF32 m(W, H, 1);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const float fx = static_cast<float>(x) / W;
            const float fy = static_cast<float>(y) / H;
            m.at(x, y) = 0.15f + 0.7f * (0.5f + 0.5f * std::sin(fx * 37.0f) *
                                                     std::cos(fy * 29.0f));
        }
    return m;
}

}  // namespace

int main() try {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const int W = 4096, H = 3072;                   // 12.6 MP
    const int cfa = static_cast<int>(CFAPattern::RGGB);
    const std::size_t n = static_cast<std::size_t>(W) * H;

    std::printf("GPU    : %s  sm_%d%d  %d SMs\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    std::printf("Image  : %dx%d = %.1f MP, CFA RGGB\n\n", W, H, n / 1e6);

    const ImageF32 mosaic = makeMosaic(W, H);

    // ---- host reference ----------------------------------------------------
    auto t0 = std::chrono::steady_clock::now();
    const ImageF32 hostOut = demosaicBilinear(mosaic, CFAPattern::RGGB);
    auto t1 = std::chrono::steady_clock::now();
    const double msHost = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---- device ------------------------------------------------------------
    DeviceBuffer<float> dMosaic(n), dOut(n * 3);
    dMosaic.upload(mosaic.data);

    // 32x8 = 256 threads. 32 wide so a warp spans 32 consecutive x, which keeps
    // the mosaic reads contiguous along a row.
    const dim3 block(32, 8);
    const dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    kDemosaicBilinear<<<grid, block>>>(dMosaic.get(), dOut.get(), W, H, cfa);
    CUDA_CHECK_KERNEL("warm-up");

    CudaTimer timer;
    const int iters = 20;
    timer.start();
    for (int i = 0; i < iters; ++i)
        kDemosaicBilinear<<<grid, block>>>(dMosaic.get(), dOut.get(), W, H, cfa);
    const float msDevice = timer.stop() / iters;
    CUDA_CHECK_KERNEL("demosaic");

    std::vector<float> gpuOut(n * 3);
    dOut.download(gpuOut);

    // ---- verify ------------------------------------------------------------
    double maxDiff = 0.0;
    std::size_t firstBad = static_cast<std::size_t>(-1);
    for (std::size_t i = 0; i < gpuOut.size(); ++i) {
        const double d = std::fabs(static_cast<double>(gpuOut[i]) - hostOut.data[i]);
        if (d > maxDiff) { maxDiff = d; if (firstBad == static_cast<std::size_t>(-1) && d > 1e-6) firstBad = i; }
    }

    // Logical traffic: read the mosaic once, write three channels.
    const double bytes = static_cast<double>(n) * sizeof(float) * 4.0;

    std::printf("%-26s %10s %12s\n", "path", "ms", "GB/s");
    std::printf("%-26s %10.2f %12s\n", "host (single-threaded)", msHost, "-");
    std::printf("%-26s %10.3f %12.1f\n", "cuda", msDevice, gbPerSec(bytes, msDevice));
    std::printf("\nspeedup    : %.0fx\n", msHost / msDevice);
    std::printf("max |diff| : %.3g%s\n", maxDiff,
                maxDiff == 0.0 ? "   <- bit-identical, correct"
                               : "   <- must be 0; see the pixel below");

    if (firstBad != static_cast<std::size_t>(-1)) {
        const std::size_t px = firstBad / 3, c = firstBad % 3;
        std::printf("\nfirst mismatch at pixel (%d, %d) channel %d\n",
                    static_cast<int>(px % W), static_cast<int>(px / W), static_cast<int>(c));
        std::printf("  host %.9g   gpu %.9g\n", hostOut.data[firstBad], gpuOut[firstBad]);
        std::printf("  site colour there is %d (0=R 1=G 2=B)\n",
                    cfaColorAt(CFAPattern::RGGB, static_cast<int>(px % W),
                               static_cast<int>(px / W)));
        std::printf("\nIf max|diff| is huge and every pixel is 0, the kernel is not running --\n"
                    "check that both TODOs are filled in. If only green is wrong, you probably\n"
                    "tested `horiz` before the both-axes case.\n");
    }

    std::printf("\nNote: %.0f%% of peak (377.5 GB/s) on LOGICAL traffic. Real DRAM traffic is\n"
                "lower -- neighbouring threads reuse the same mosaic samples through cache,\n"
                "which is why a neighbourhood kernel can appear to exceed a copy kernel.\n",
                100.0 * gbPerSec(bytes, msDevice) / 377.5);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
