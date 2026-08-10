# Measurements

Every number produced by this project, with the reasoning that explains it. All measured on:

```
RTX 5060 (Blackwell, sm_120), 30 SMs, 8 GB GDDR7, 48 KB shared/block
CUDA 13.0, GCC 13.3, Ubuntu 24.04 under WSL2
Host: single-threaded C++, -O3
```

Reproduce with `./build.sh` then `./build/bwtest`, `./build/fusion`, `./build/experiments`, `./smoke.sh`.

---

## 1. The reference number

```
$ ./build/bwtest
MEASURED PEAK BW : 377.5 GB/s  (2.844 ms/iter)
```

A pure copy kernel over 512 MiB buffers. **377.5 GB/s is the denominator for every performance claim
here** — roughly 84% of the ~448 GB/s theoretical figure (128-bit GDDR7 at 28 Gbps).

The missing 16% is DRAM refresh, read/write bus turnaround, and imperfect access scheduling. It is
real and unreachable, which is why quoting against the spec sheet overstates every result.

---

## 2. Kernel fusion — 3.03×

```
path                     ms      GB/s
3 separate kernels    6.002     383.9
1 fused kernel        1.979     388.1

DRAM traffic: 2304 MB -> 768 MB      max |diff|: 0
```

24 MP RGBA fp32. Three pointwise stages (colour matrix, tone map, sRGB encode) run as separate kernels,
then as one fused kernel computing identical arithmetic via shared `__device__` functions.

### The prediction, made before writing the code

A pointwise stage on RGBA fp32 reads 16 B and writes 16 B per pixel, against a handful of FLOPs —
under 1 FLOP/byte, an order of magnitude below the roofline ridge. So every stage costs one read plus
one write of the image regardless of its math.

```
384 MB image  ->  768 MB traffic per stage  ->  768e6 / 377.5e9 = 2.0 ms floor
3 stages unfused: ~6 ms        1 stage fused: ~2 ms        predicted 3x
```

Measured 3.03×.

### The observation that matters

**Both paths achieve the same bandwidth**, ~385 GB/s. The fused kernel is not more efficient per byte.
It is faster purely because it moves 3× fewer bytes — the intermediates stay in registers instead of
round-tripping to DRAM.

This is the empirical argument for why an imaging framework concatenates kernels rather than optimising
them individually. Making the arithmetic faster would gain nothing; the ALUs are already idle.

*(Both figures land slightly above the 377.5 GB/s reference, which just means the copy kernel was not a
hard ceiling — a copy over 512 MiB is a slightly different access pattern. Read both as "at the memory
system's limit".)*

---

## 3. Coalescing — 1.14× and 13.5×

```
pattern                        ms      GB/s    vs contiguous
contiguous  [y*W + x]       1.561     343.9        1.00x
transposed  [x*H + y]       1.774     302.7        1.14x
permuted    (no reuse)     21.126      25.4       13.5x
```

4096×4096 float4. All three touch every element exactly once — identical byte counts. Only the
thread→address mapping differs.

### Why the transpose costs almost nothing

A transposed access **is** scattered within every warp: 32 lanes land 4096 elements apart, needing 32
separate sectors. The textbook prediction is a collapse.

It costs 14%, because the sectors a warp touches are reused almost immediately by neighbouring warps,
and L2 is large enough to hold them. The pattern is scattered in the small and dense in the aggregate.

### Why the permutation does not survive

Multiplying by an odd constant modulo a power of two is a bijection, so every element is still visited
once — but with no locality *and no reuse*. Each 32-byte sector fetched delivers one useful 16-byte
pixel and is then evicted before anyone else needs it.

### The conclusion

Coalescing still matters, but on current hardware **losing reuse costs far more than losing
contiguity**. The "transpose is 10× slower" rule dates from an era of much smaller caches. Both rules
are real; their relative weight has shifted, and the only way to know is to measure on the hardware in
front of you.

---

## 4. Branch divergence — 1.90×

```
uniform branch   (blockIdx & 1)    0.948 ms
divergent branch (threadIdx & 1)   1.804 ms
```

Same two code paths, same instruction count. Only *who* branches changes: keyed on `blockIdx` an entire
block agrees, so every warp is uniform; keyed on `threadIdx` lanes alternate and every warp splits.

A warp's 32 threads share one instruction pointer, so a split warp executes both sides serially with
the non-participating lanes disabled. The near-2× is the whole of both paths being run.

### The first version of this benchmark measured nothing

`pathA` and `pathB` were originally `fmaf()` loops differing only in their constants. The compiler
hoisted the constants into a select and ran a single loop — removing the divergence entirely. It
reported 1.23×, which is noise.

The paths had to be made **structurally** different (`__sinf` vs `__cosf`) before the branch survived
optimisation.

**Microbenchmarks frequently measure the compiler rather than the hardware.** Any result that looks too
small deserves a look at the generated SASS before it is believed.

### Why this shapes algorithm choice

A fixed 5×5 filter like Malvar-He-Cutler executes the same instructions for every pixel. An adaptive
demosaic that picks an interpolation direction per pixel splits every warp. The adaptive method scores
higher on PSNR and can still lose on a GPU — choosing the algorithm that maps to the hardware is a
routine tradeoff in this domain.

---

## 5. Precision — 1.96×

```
RGBA fp32  (16 B/px)   2.368 ms   340.1 GB/s
RGBA fp16  ( 8 B/px)   1.210 ms   332.8 GB/s
```

Same pixel count, same arithmetic. Only the storage format changes.

Both reach the same GB/s — neither kernel is more efficient per byte. fp16 wins purely by moving half
as many. This is the same mechanism as kernel fusion, applied to format instead of passes, and it is
why an imaging framework exposes pixel formats at all: for speed, not storage.

**The caveat is real.** fp16 carries ~10 bits of mantissa. That is adequate for display-referred values
after tone mapping, and marginal for deep scene-referred highlights where a single stop can span the
available precision. Choosing a format per stage is a genuine pipeline decision.

---

## 6. Host pipeline baseline

1024×768 (0.8 MP), single-threaded, σ=1.5 amount=0.8:

```
linearize + WB       2.19 ms
demosaic (bilinear) 11.01 ms        (MHC: 22.62 ms)
camera -> sRGB       0.47 ms
tone map             1.39 ms
unsharp             48.54 ms   <- bottleneck
encode sRGB8         9.81 ms
--------------------------------
total               73.42 ms
```

### `cameraToSRGB` is already memory-bound on the CPU

```
0.79 MP x 3ch x 4 B = 9.4 MB, read and written = 18.9 MB in 0.47 ms  =  ~40 GB/s
9 MACs x 0.79 M = 7 MFLOP in 0.47 ms                                 =  ~15 GFLOP/s
```

40 GB/s is single-core memory bandwidth. 15 GFLOP/s is far below what one core delivers with SIMD. The
project's central claim appears before the GPU is involved at all.

### Separable convolution scales as N^1.4, not N

```
sigma   N    ms
    1   7    29.8
    2  13    70.0
    4  25   229.7
    8  49   490.3
```

Separable convolution costs 2N taps per pixel instead of N², so time should be **linear** in N. Seven
times the taps costs 16.5× the time.

The vertical pass is responsible: consecutive taps are `width × 3` floats apart — about 12 KB here — so
every tap is its own cache line and the working set leaves L1 as the radius grows. The horizontal pass
reads adjacent memory and does not suffer.

This is precisely the problem shared-memory tiling solves on a GPU: stage a tile plus its halo into
fast memory once, then let both passes read from there.

---

## 7. Demosaic quality

PSNR against ground truth, border excluded. Generate the set with `python3 tools/make_eval_set.py`.

| Image | bilinear | MHC | Δ |
|---|---|---|---|
| flat field (4 CFA layouts) | identical | identical | — |
| smooth gradient | 62.37 | 58.86 | −3.51 |
| `gray_lo` | 56.28 | 64.11 | **+7.83** |
| `gray_mid` | 47.18 | 52.20 | **+5.02** |
| `gray_hi` | 31.93 | 40.30 | **+8.37** |
| `corr` (constant channel ratios) | 33.02 | 35.17 | **+2.15** |
| `photo_mid` | 47.94 | 46.47 | −1.47 |
| `anti` (anti-correlated) | 33.85 | 28.59 | **−5.26** |
| `scene` (adversarial chart) | 18.97 | 18.28 | −0.69 |

### MHC's assumption, and where it fails

Malvar-He-Cutler corrects the bilinear estimate of one channel using the **Laplacian of the channel
actually measured at that site**. That is valid only when luminance detail is shared across R, G and B
— which holds in real photographs, where luminance dominates.

Feed it channels that vary independently and the "correction" injects error. On anti-correlated content
it loses 5.3 dB. Bilinear assumes nothing about inter-channel structure and degrades gracefully.

Where the assumption holds exactly (`gray_*`), MHC wins by 5–8.4 dB.

### How the implementation was verified

Three independent checks, because each catches a different class of bug:

1. **Flat field exact across all four CFA layouts.** Every kernel sums to 8 and is applied with a 1/8
   scale, so DC gain is exactly 1. Catches normalisation errors.
2. **Deliberate H/V kernel swap collapses `corr` from 35.17 to 22.86 dB.** Confirms the orientation as
   written is correct — the classic MHC bug is swapping the kernels used at green sites on red rows
   versus blue rows.
3. **`gray_*` gains of +5 to +8.4 dB.** The design case works.

A methodological note worth keeping: **a grey image cannot detect the H/V swap.** The swap merely
exchanges the R and B estimates, and when `R_true == B_true` the aggregate error is unchanged. `corr`
— constant but *unequal* channel ratios — is what catches it. A test that cannot fail proves nothing.

---

## 8. Summary

| Effect | Measured | Mechanism |
|---|---|---|
| Kernel fusion | 3.03× | 3× fewer DRAM round-trips; same GB/s |
| fp16 vs fp32 | 1.96× | Half the bytes; same GB/s |
| Warp divergence | 1.90× | Both branch paths executed serially |
| Lost reuse (permuted) | 13.5× | Sectors fetched, one pixel used, evicted |
| Lost contiguity (transpose) | 1.14× | L2 absorbs it — scattered but dense in aggregate |
| MHC vs bilinear | +8.4 / −5.3 dB | Depends entirely on inter-channel correlation |

Three of these contradict the answer you would give from a textbook: the transpose penalty is small,
the divergence benchmark initially measured the compiler, and a better demosaic algorithm can be worse.
Each was found by measuring rather than assuming.
