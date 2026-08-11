# cuda_smooth

A standalone 3×3 box blur in CUDA, with OpenCV for image I/O. Written as a first-principles CUDA
exercise, independent of the main TinyCI build.

```bash
nvcc -O3 -arch=sm_120 smooth_png.cu -o smooth_png $(pkg-config --cflags --libs opencv4)
./smooth_png input.png output.png
```

## What it demonstrates

- **2-D thread indexing.** `blockIdx * blockDim + threadIdx` in both dimensions, because a
  neighbourhood operator needs its coordinates.
- **The bounds guard.** `16×16` blocks do not divide 1024×768 evenly, so surplus threads exist and must
  return before touching memory.
- **Clamped borders.** `max(0, min(n, size-1))` replicates the edge pixel rather than reading out of
  bounds — the cheapest boundary policy, and the one Core Image calls `clampedToExtent`.
- **`float4` rather than `float3`.** 16 bytes is naturally aligned with a single hardware load
  instruction; 12 bytes puts thread N at byte 12N and straddles sector boundaries. The unused alpha
  channel costs 25% of memory and the kernel is faster for it.
- **Launch error checking.** `cudaGetLastError()` after the launch, then `cudaDeviceSynchronize()` —
  the first catches a bad configuration, the second surfaces errors raised during execution.

## Known limitations

These are deliberate notes rather than oversights, and the first one matters.

**1. The blur runs in gamma-encoded space, which is wrong.**

`imageBGRA.convertTo(imageFloat, CV_32FC4, 1.0/255.0)` rescales 0–255 to 0–1. It does **not**
linearise. A PNG is sRGB-encoded, so those values are not proportional to light.

Averaging encoded values is the classic image-processing error: the arithmetic mean of two encoded
values is not the encoding of their mean radiance. The result comes out **too dark**, most visibly
across high-contrast edges, where averaging a bright and a dark pixel should land near the bright one
in perceptual terms but instead lands near the midpoint.

The fix is to apply the sRGB EOTF before blurring and the OETF afterwards — which is exactly what
`linearizeAndWhiteBalance` and `encodeSRGB8` do in the main pipeline (`../src/pipeline.cpp`). This
program predates that habit.

**2. No timing.** No `cudaEvent` measurement and no warm-up launch, so there is no performance number
here. See `../bench/` for benchmarks that measure properly.

**3. No shared memory, so every pixel is read 9 times from global memory.** This is precisely the reuse
case where staging a tile plus a 1-pixel halo into shared memory pays: 9 global loads per pixel become
1 global load plus 9 shared loads. Cache absorbs much of it in practice, but relying on that is luck
rather than design.

**4. The alpha read is redundant.** `src[outputIndex].w` re-loads the centre pixel, which the
neighbourhood loop already visited at `dx == dy == 0`. Also moot here: `cv::IMREAD_COLOR` discards any
alpha and `COLOR_BGR2BGRA` synthesises an opaque one, so it is always 1.0.

**5. BGRA, not RGBA.** OpenCV's byte order. Harmless for a symmetric box blur, which treats all three
colour channels identically, but it would matter for any operation that is not channel-symmetric — a
colour matrix, or luminance weights.
