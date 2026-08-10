# CUDA code walkthrough

Every CUDA construct in this repository, explained where it appears. Read alongside
[gpu-architecture.md](gpu-architecture.md) — that document is the model, this one is the code.

Files, in the order to read them:

1. `bench/bwtest.cu` — the simplest possible kernel
2. `include/tinyci/cuda_util.h` — the infrastructure
3. `bench/fusion.cu` — the central measurement
4. `bench/experiments.cu` — three isolated effects
5. `bench/demosaic_gpu.cu` — a real 2-D kernel

---

## 1. `bwtest.cu` — anatomy of a kernel

```cuda
__global__ void copyK(const float4* __restrict__ src, float4* __restrict__ dst, size_t n){
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
```

### `__global__`

Marks a function as a **kernel**: compiled for the GPU, called from the CPU, launched across a grid of
threads. Must return `void` — there is no single return value when a million threads run the function.

The three function qualifiers:

| Qualifier | Runs on | Called from |
|---|---|---|
| `__global__` | GPU | CPU (a kernel launch) |
| `__device__` | GPU | GPU only |
| `__host__` | CPU | CPU (the default) |

### `blockIdx.x * blockDim.x + threadIdx.x`

The most important line in CUDA. It is how *one thread's code* becomes *all the pixels*.

- `threadIdx.x` — this thread's index **within its block** (0 … blockDim.x−1)
- `blockDim.x` — threads per block (e.g. 256)
- `blockIdx.x` — this block's index within the grid
- `gridDim.x` — blocks in the grid

Block 0 covers elements 0–255, block 1 covers 256–511, and so on. Every thread computes a unique `i`
and does exactly one element's work.

The `(size_t)` cast matters: `blockIdx.x * blockDim.x` is `unsigned int` arithmetic and overflows past
4.29 billion. On large images that silently wraps and corrupts memory. Cast before the multiply, not
after.

### `if (i < n)` — the bounds guard

**Mandatory, not defensive.** The grid is launched in whole blocks, so `gridDim.x * blockDim.x` is
almost always larger than `n`:

```
n = 1000, block = 256  ->  grid = ceil(1000/256) = 4  ->  1024 threads for 1000 elements
```

Those 24 surplus threads must return before touching memory. Without the guard they write out of
bounds — which frequently does not crash, it just corrupts a neighbouring allocation. That is worse
than a crash, and it is what `compute-sanitizer` exists to catch.

### `__restrict__`

A promise that `src` and `dst` do not alias — no byte is reachable through both. Without it the
compiler must assume a write through `dst` could change what `src` points at, so it cannot keep loaded
values in registers or reorder loads across stores. With it, it can.

Free performance, and safe here because the buffers are separate allocations. Lying about it produces
wrong results with no diagnostic.

### `float4`

A built-in 16-byte vector type (`.x .y .z .w`). Used rather than `float3` because 16 bytes is naturally
aligned and has a single hardware load instruction, while `float3` puts thread N at byte 12N and
straddles sector boundaries. Wasting 25% of memory on an unused alpha channel makes the kernel faster.

### The launch

```cuda
const int block = 256;
const int grid  = (int)((n + block - 1) / block);   // ceiling division
copyK<<<grid, block>>>(d_src, d_dst, n);
```

`<<<grid, block>>>` is the execution configuration: how many blocks, and how many threads per block.
`(n + block - 1) / block` is integer ceiling division — it guarantees full coverage, at the cost of the
surplus threads the guard handles.

Why 256: a multiple of the 32-wide warp (so no partial warps), large enough to keep an SM busy, small
enough that several blocks fit per SM for latency hiding. 128, 256 and 512 are all reasonable; 256 is
the usual default.

### Warm-up and timing

```cuda
copyK<<<grid, block>>>(d_src, d_dst, n);      // warm-up, untimed
CHECK(cudaGetLastError());
CHECK(cudaDeviceSynchronize());

cudaEventRecord(t0);
for (int i = 0; i < iters; ++i) copyK<<<grid, block>>>(d_src, d_dst, n);
cudaEventRecord(t1);
cudaEventSynchronize(t1);
```

The warm-up absorbs JIT compilation, context creation and first-touch page faults, any of which can
make the first launch 10–100× slower.

The loop is not synchronised internally. Launches queue asynchronously and execute back to back, so the
two events bracket the whole batch and dividing by `iters` gives a clean per-launch time.

### The bandwidth calculation

```cuda
const double moved = 2.0 * bytes * iters;    // one read + one write per element
printf("%.1f GB/s", moved / (ms * 1e-3) / 1e9);
```

The factor of 2 is the point: a copy reads every byte and writes every byte. This number — **377.5
GB/s** on this machine — is the denominator for every later performance claim.

---

## 2. `cuda_util.h` — the infrastructure

### `CUDA_CHECK`

```cuda
#define CUDA_CHECK(expr) ::tinyci::cudaCheckImpl((expr), #expr, __FILE__, __LINE__)
```

Every CUDA API call returns a status, and nearly every one can fail. `#expr` is the preprocessor
stringify operator, so the error message contains the failing call verbatim.

The reason to check *every* call: CUDA errors are **sticky and asynchronous**. An error raised inside a
kernel is not reported at the launch — the launch already returned. It surfaces at whatever API call
happens to synchronise next, which is usually somewhere innocent. Checking everything keeps the blame
near the crime.

### `CUDA_CHECK_KERNEL`

```cuda
inline void cudaCheckKernel(...) {
    cudaCheckImpl(cudaGetLastError(), ...);        // launch configuration errors
    cudaCheckImpl(cudaDeviceSynchronize(), ...);   // execution errors
}
```

Two calls because they catch different failures. A kernel launch returns `void`, so `cudaGetLastError`
is the only way to learn the configuration was invalid (too many threads per block, too much shared
memory). `cudaDeviceSynchronize` then waits for execution and surfaces anything raised while running.

### `DeviceBuffer<T>`

RAII for `cudaMalloc`/`cudaFree` — the GPU's `malloc`/`free`, with the same leak consequences on an
8 GB card.

```cuda
DeviceBuffer(const DeviceBuffer&)            = delete;
DeviceBuffer& operator=(const DeviceBuffer&) = delete;
DeviceBuffer(DeviceBuffer&&) noexcept;
```

**Copying is deleted deliberately.** A default copy would duplicate the pointer, giving two objects
that both free it — a double free, and a use-after-free for whichever survives. Moves are fine: the
source is left null so its destructor does nothing.

```cuda
void release() {
    if (ptr_) cudaFree(ptr_);   // no CUDA_CHECK here
    ...
}
```

No `CUDA_CHECK` in the destructor path, because `cudaCheckImpl` throws and **throwing from a destructor
during stack unwinding calls `std::terminate`**. Destructors must not throw.

### `CudaTimer`

```cuda
void start() { CUDA_CHECK(cudaEventRecord(start_)); }
float stop() {
    CUDA_CHECK(cudaEventRecord(stop_));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
}
```

`std::chrono` cannot time a kernel. Launches are asynchronous and return before the GPU has begun, so a
host timer measures the launch call and reports microseconds for work that takes milliseconds.

CUDA events are timestamped **by the GPU, in stream order**. `cudaEventRecord` enqueues a timestamp
into the stream; `cudaEventSynchronize` blocks the host until the GPU reaches it.

---

## 3. `fusion.cu` — the central measurement

### `__device__ __forceinline__` stage functions

```cuda
__device__ __forceinline__ float4 stageColorMatrix(float4 p, const float* __restrict__ M) { ... }
```

`__device__` — GPU-only, callable from kernels.

`__forceinline__` matters here for a reason beyond speed: **both the separate and the fused paths call
the same functions**, which guarantees they compute bit-identical arithmetic. If the two paths had
duplicated math, the benchmark could be measuring a difference in what is computed rather than in how
it is scheduled. The run confirms this — `max |diff| : 0`.

### The two paths

```cuda
// Path A: three kernels, each reading and writing the whole image.
kColorMatrix<<<grid, block>>>(a, b, n, M);
kToneMap    <<<grid, block>>>(b, c, n, exposure, invW2);
kEncode     <<<grid, block>>>(c, b, n);

// Path B: one kernel. Intermediates live in registers.
kFused<<<grid, block>>>(a, c, n, M, exposure, invW2);
```

Identical arithmetic. The only difference is where the intermediates live: DRAM in path A, registers in
path B.

### The result

```
3 separate kernels    6.002 ms    383.9 GB/s
1 fused kernel        1.979 ms    388.1 GB/s
speedup 3.03x         2304 MB -> 768 MB DRAM traffic
```

**Both paths achieve the same bandwidth.** The fused kernel is not more efficient per byte. It is 3×
faster purely because it moves 3× fewer bytes.

That single observation is the project's thesis, and it is why an imaging framework concatenates
kernels rather than merely optimising them.

### 1-D versus 2-D indexing

`fusion.cu` treats the image as a flat array. Deliberate: every stage is pointwise, so no kernel needs
to know where its pixel sits. 1-D is cheaper (one multiply-add and one guard instead of two), wastes no
threads rounding rows up to block boundaries, and is perfectly coalesced.

Use 2-D when the kernel needs coordinates — demosaic (CFA colour depends on `x&1`, `y&1`), separable
blur, and anything doing shared-memory tiling where the block shape is the tile shape.

**Caveat:** flat indexing assumes rows are stored back to back. Valid here because `Image<T>` has
`stride == width * channels`. Production imaging buffers — `MTLTexture`, most Core Image backings —
align each row to a 64- or 256-byte **row pitch**, so row `y` starts at `y * pitch`. With padding, flat
indexing walks into the pad bytes.

---

## 4. `experiments.cu` — three isolated effects

Each changes exactly one thing and measures.

### Coalescing

```cuda
__global__ void kRowMajor(...) { const int i = y * W + x; dst[i] = src[i]; }
__global__ void kColMajor(...) { const int i = x * H + y; dst[i] = src[i]; }
__global__ void kPermuted(..., unsigned n) {
    const unsigned j = (i * 2654435761u) & (n - 1);
    dst[j] = src[j];
}
```

All three touch every element exactly once, so the byte count is identical. Only the thread→address
mapping differs.

The permutation works because multiplying by an **odd** constant modulo a **power of two** is a
bijection. Every element is still visited once, but with no locality and no reuse.

Results: transposed **1.14×**, permuted **13.5×**. The gap is the lesson — L2 rescues a scattered
pattern that still has aggregate density, and cannot rescue one with no reuse.

### Divergence

```cuda
v = (blockIdx.x  & 1) ? pathA(v) : pathB(v);   // whole warp agrees -> free
v = (threadIdx.x & 1) ? pathA(v) : pathB(v);   // lanes alternate  -> both paths run
```

Result: **1.90×**.

The comment in the source records a failed first attempt worth understanding. The original `pathA` and
`pathB` were `fmaf()` loops differing only in their constants, and the compiler hoisted the constants
into a select and ran a single loop — removing the divergence and making the experiment measure
nothing (1.23×, which is noise). The paths had to become structurally different (`__sinf` vs `__cosf`)
to survive optimisation.

Compute-heavy paths are also necessary: on a memory-bound kernel the extra ALU work hides behind memory
latency and the effect does not appear at all.

### Precision

```cuda
struct __align__(8) half4 { __half x, y, z, w; };
```

`__align__(8)` is required. Without it the compiler may not guarantee 8-byte alignment, and unaligned
vector loads are slower or illegal. CUDA has no built-in `half4`, so it is declared explicitly.

Result: **1.96×**, both at ~335 GB/s. Same mechanism as fusion — fewer bytes — applied to format rather
than passes.

The caveat is real: fp16 carries ~10 bits of mantissa. Adequate for display-referred values, marginal
for deep scene-referred highlights. That is a genuine pipeline decision, not a free win.

---

## 5. `demosaic_gpu.cu` — a real 2-D kernel

Two functions are left to implement; everything else is provided.

### Provided helpers

```cuda
__device__ __forceinline__ int cfaColorAtDev(int cfa, int x, int y);
__device__ __forceinline__ float sampleClamped(const float* m, int W, int H, int x, int y);
```

`__forceinline__` is not cosmetic here: these are called three times per pixel, and a real function
call on a GPU (stack frame, register spill) costs far more than the handful of instructions inside.

`CFAPattern` is passed as `int` because enum class values are clumsier to hand to a kernel. 0=RGGB,
1=BGGR, 2=GRBG, 3=GBRG.

`cfaColorAtDev` is safe at negative coordinates because it only reads parity, and `x & 1` for negative
`x` yields the correct parity in two's complement. The *samples* need clamping; the colour lookup does
not.

### What to write

```cuda
__global__ void kDemosaicBilinear(const float* __restrict__ mosaic, float* __restrict__ out,
                                  int W, int H, int cfa) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    // write out[(y*W + x)*3 + c] for c = 0,1,2
}
```

Two dimensions now, so **two** index computations and **two** bounds checks.

The launch uses `dim3 block(32, 8)` — 256 threads shaped so a warp spans 32 consecutive `x`, which
keeps each warp's mosaic reads contiguous along one row. `dim3 grid(...)` covers both dimensions with
ceiling division.

### Coalescing in this kernel

Consecutive threads write consecutive pixels, but the output is 3 interleaved floats per pixel — a
12-byte stride, neither 4 nor 16. Not perfectly coalesced. It is a fair thing to raise about your own
code, and the fix is the `float4` argument from §4 of the architecture notes: pad to RGBA and pay 25%
more memory for aligned 16-byte stores.

### Verification

The harness compares against `demosaicBilinear` from the host library and requires **`max |diff| : 0`**
— bit-identical. Same algorithm and same arithmetic on a different machine should agree exactly; both
compute in fp32 and do the same operations in the same order.

The two failure modes it diagnoses explicitly:

- **Every pixel zero** — the kernel is not running, or the TODOs are unfilled.
- **Only green wrong** — the `horiz` case was tested before the both-axes case. Green occupies a
  quincunx, so at any non-green site it lies on both axes at once; falling into the plain horizontal
  branch averages 2 greens instead of 4.

---

## 6. Constructs index

| Construct | Meaning |
|---|---|
| `__global__` | Kernel: runs on GPU, launched from CPU, returns `void` |
| `__device__` | GPU function, callable only from GPU code |
| `__forceinline__` | Force inlining; matters for small functions called per pixel |
| `__restrict__` | Pointers do not alias; enables register caching and reordering |
| `__align__(n)` | Force type alignment; required for custom vector types |
| `<<<grid, block>>>` | Execution configuration |
| `threadIdx` / `blockIdx` | Position within block / grid |
| `blockDim` / `gridDim` | Block size / grid size |
| `dim3` | 1-, 2- or 3-component launch dimensions |
| `__syncthreads()` | Barrier across a block. Every thread must reach it |
| `cudaMalloc` / `cudaFree` | Device allocation |
| `cudaMemcpy` | Host↔device transfer; synchronous by default |
| `cudaEventRecord` / `ElapsedTime` | GPU-side timing |
| `cudaGetLastError` | Catches launch configuration errors |
| `cudaDeviceSynchronize` | Blocks until the GPU is idle; surfaces execution errors |
| `float4`, `__half` | Vector and half-precision types |
| `__sinf`, `__powf` | Fast-math intrinsics: fewer instructions, lower accuracy |
