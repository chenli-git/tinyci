#include "tinyci/pipeline.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace tinyci {

// ===========================================================================
// STAGE 1 -- IMPLEMENTED. Read this one first; the rest follow the same shape.
// ===========================================================================
ImageF32 linearizeAndWhiteBalance(const ImageU16& mosaic, const RawParams& p) {
    if (mosaic.channels != 1)
        throw std::runtime_error("linearizeAndWhiteBalance: expected a 1-channel mosaic");

    ImageF32 out(mosaic.width, mosaic.height, 1);
    const float scale = 1.0f / std::max(1.0f, p.whiteLevel - p.blackLevel);

    for (int y = 0; y < mosaic.height; ++y) {
        for (int x = 0; x < mosaic.width; ++x) {
            // Sensor codes are linear in photon count but offset: "black" is a
            // nonzero code. Subtract it before any arithmetic that assumes
            // 0 == no light -- otherwise every later ratio is wrong.
            float v = (static_cast<float>(mosaic.at(x, y)) - p.blackLevel) * scale;
            if (v < 0.0f) v = 0.0f;

            // White balance is applied HERE, on the mosaic, while each site still
            // holds exactly one known colour. Do it after demosaic and you have
            // already interpolated across channels carrying mismatched gains --
            // which bakes a colour cast into every edge. This ordering constraint
            // is worth being able to explain out loud.
            v *= p.wbGain[cfaColorAt(p.cfa, x, y)];

            out.at(x, y) = v;
        }
    }
    return out;
}
float interpolatedAt(const ImageF32& img, CFAPattern cfa, int x, int y, int c) 
    {
        
        if (cfaColorAt(cfa, x, y) == c)  // already the right colour;
            return img.atClamped(x, y);
        const bool horiz =  cfaColorAt(cfa, x - 1, y) == c;
        const bool vert  =  cfaColorAt(cfa, x, y - 1) == c;
        if (horiz && vert) {
            return 0.25f * (img.atClamped(x - 1, y) +
                            img.atClamped(x + 1, y) +
                            img.atClamped(x, y + 1) +
                            img.atClamped(x, y - 1));
        } else if (horiz) {
            return 0.5f * (img.atClamped(x - 1, y) +
                           img.atClamped(x + 1, y));
        } else if (vert) {
            return 0.5f * (img.atClamped(x, y - 1) +
                           img.atClamped(x, y + 1));
        } else {
            return 0.25f * (img.atClamped(x - 1, y - 1) +
                            img.atClamped(x + 1, y - 1) +
                            img.atClamped(x - 1, y + 1) +
                            img.atClamped(x + 1, y + 1));
        }
    }

// ===========================================================================
// STAGE 2a -- Bilinear demosaic.
//
// Every site measures exactly one of R/G/B; the other two are interpolated from
// same-coloured neighbours. Four cases, dispatched in interpolateAt() above:
//
//   GREEN occupies half the sites, in a quincunx, so at any non-green site it
//   lies N, S, E AND W -- both axes at once. Average all 4.
//
//   RED and BLUE each occupy a quarter, on square lattices, so at a site missing
//   one of them exactly one case holds:
//     - it lies left and right   -> average those 2
//     - above and below          -> average those 2
//     - on the 4 diagonals       -> average those 4
//
// The two green sites are NOT interchangeable: a green on a red row has red
// horizontally and blue vertically; a green on a blue row is the reverse.
// Deriving the case from cfaColorAt() rather than hard-coded parity handles that
// automatically and keeps this correct for all four Bayer layouts.
//
// BORDER: cfaColorAt(-1,y) reports the colour that *would* sit outside the image,
// but atClamped(-1,y) reads x=0, a different colour. The outermost row and column
// are therefore contaminated. Mirroring by 2 (x=-1 -> x=+1) would preserve CFA
// parity; clamping is kept here for simplicity, and the PSNR harness excludes a
// border. Note that a downstream blur SPREADS this error by its own radius.
//
// Expect zipper artifacts and colour fringing on high-contrast detail near
// Nyquist -- that is inherent to interpolating each channel in isolation, and is
// exactly what demosaicMHC() below addresses.
// ===========================================================================
ImageF32 demosaicBilinear(const ImageF32& mosaic, CFAPattern cfa) {
    if (mosaic.channels != 1)
        throw std::runtime_error("demosaicBilinear: expected a 1-channel mosaic");

    ImageF32 out(mosaic.width, mosaic.height, 3);

    for (int y = 0; y < mosaic.height; ++y) {
        for (int x = 0; x < mosaic.width; ++x) {
            out.at(x, y, 0) = interpolatedAt(mosaic, cfa, x, y, 0);
            out.at(x, y, 1) = interpolatedAt(mosaic, cfa, x, y, 1);
            out.at(x, y, 2) = interpolatedAt(mosaic, cfa, x, y, 2);
        }
    }

    return out;
}

// ===========================================================================
// STAGE 2b -- Malvar-He-Cutler gradient-corrected demosaic, ICASSP 2004.
//
// The insight: bilinear interpolates each channel in isolation, but luminance
// detail is shared across R, G and B -- a sharp edge appears in all three. So
// correct the bilinear estimate of one channel using the *Laplacian of the
// channel actually measured at that site*:
//
//     G_at_R  =  bilinear_G(x,y)  +  alpha * laplacian_R(x,y)
//
// It stays a fixed 5x5 LINEAR filter -- no branching on image content, no
// data-dependent work -- which is exactly why it maps well to a GPU. Adaptive
// and directional methods score higher but diverge per pixel, and a warp whose
// threads take different paths runs those paths serially.
//
// Four distinct kernels are needed (below); the paper's gains alpha = 1/2,
// beta = 5/8, gamma = 3/4 are already folded into their coefficients.
//
// Every kernel sums to 8 and is applied with a 1/8 scale, so DC gain is exactly
// 1 -- a flat field passes through untouched. That is the cheapest correctness
// check available here, and worth verifying before trusting any output.
// ===========================================================================
namespace {

// Malvar-He-Cutler 5x5 kernels, row-major, to be scaled by 1/8.
// Derivation, for the G-at-R case:
//     G = bilinear_G + (1/2) * laplacian_R
//       = (1/4)*sum(G_adjacent) + (1/2)*[ R(x,y) - (1/4)*sum(R_at_offset_2) ]
//   x8:  2*sum(G_adjacent) + 4*R(x,y) - sum(R_at_offset_2)
// which is precisely the coefficient pattern below.

// Green at a red or blue site.
const float K_G_AT_RB[25] = {
     0,  0, -1,  0,  0,
     0,  0,  2,  0,  0,
    -1,  2,  4,  2, -1,
     0,  0,  2,  0,  0,
     0,  0, -1,  0,  0,
};

// Red (or blue) at a green site, with the target colour lying HORIZONTALLY.
// The 4s sit at (x+-1, y), where the measured samples of that colour are.
const float K_C_AT_G_H[25] = {
     0,     0,  0.5f,  0,     0,
     0,    -1,  0,    -1,     0,
    -1,     4,  5,     4,    -1,
     0,    -1,  0,    -1,     0,
     0,     0,  0.5f,  0,     0,
};

// Red (or blue) at a green site, with the target colour lying VERTICALLY.
// The transpose of the above -- the green sites on a red row and those on a
// blue row are NOT interchangeable, and swapping these two is the classic bug.
const float K_C_AT_G_V[25] = {
     0,     0,    -1,  0,     0,
     0,    -1,     4, -1,     0,
     0.5f,  0,     5,  0,     0.5f,
     0,    -1,     4, -1,     0,
     0,     0,    -1,  0,     0,
};

// Red at a blue site, or blue at a red site: only the diagonals carry it.
const float K_C_AT_OPP[25] = {
     0,     0,    -1.5f,  0,     0,
     0,     2,     0,     2,     0,
    -1.5f,  0,     6,     0,    -1.5f,
     0,     2,     0,     2,     0,
     0,     0,    -1.5f,  0,     0,
};

// Branchless 5x5 convolution. Zero taps are multiplied rather than skipped:
// the branch would cost more than the multiply, and the GPU port wants a
// uniform instruction stream across the warp anyway.
float applyKernel5(const ImageF32& m, int x, int y, const float k[25]) {
    float sum = 0.0f;
    for (int j = -2; j <= 2; ++j)
        for (int i = -2; i <= 2; ++i)
            sum += k[(j + 2) * 5 + (i + 2)] * m.atClamped(x + i, y + j);
    return sum * 0.125f;
}

}  // namespace

ImageF32 demosaicMHC(const ImageF32& mosaic, CFAPattern cfa) {
    if (mosaic.channels != 1)
        throw std::runtime_error("demosaicMHC: expected a 1-channel mosaic");

    ImageF32 out(mosaic.width, mosaic.height, 3);

    for (int y = 0; y < mosaic.height; ++y) {
        for (int x = 0; x < mosaic.width; ++x) {
            const int site = cfaColorAt(cfa, x, y);   // colour measured here

            for (int c = 0; c < 3; ++c) {
                float v;
                if (c == site) {
                    // Measured directly -- never estimate what you sampled.
                    v = mosaic.at(x, y);
                } else if (c == 1) {
                    // Green missing at a red/blue site.
                    v = applyKernel5(mosaic, x, y, K_G_AT_RB);
                } else if (site == 1) {
                    // Red or blue missing at a green site. Which axis carries
                    // it depends on whether this green sits on a red row or a
                    // blue row -- ask the CFA rather than assuming.
                    const bool horiz = cfaColorAt(cfa, x - 1, y) == c;
                    v = applyKernel5(mosaic, x, y, horiz ? K_C_AT_G_H : K_C_AT_G_V);
                } else {
                    // Red at a blue site, or blue at a red site.
                    v = applyKernel5(mosaic, x, y, K_C_AT_OPP);
                }

                // The negative kernel taps can overshoot below zero at a hard
                // edge. Negative light is unphysical, so clamp the floor; leave
                // the ceiling alone, since linear values above 1 are legitimate
                // headroom for the tone curve downstream.
                out.at(x, y, c) = v < 0.0f ? 0.0f : v;
            }
        }
    }
    return out;
}

// ===========================================================================
// STAGE 3 -- Camera RGB -> linear sRGB.
//
// Raw camera RGB is DEVICE-DEPENDENT: the sensor's red filter does not pass the
// same band as the eye's L cones, so two cameras report different triples for
// one physical colour. The numbers are meaningless until you say whose red.
//
// The fix is to pivot through a space defined by human vision rather than by a
// sensor -- CIE XYZ, grounded in the 1931 standard observer:
//
//     XYZ         = camToXYZ    * cameraRGB     (measured, per illuminant)
//     linear sRGB = XYZ_TO_SRGB * XYZ           (defined by sRGB's primaries)
//
// Both operate in LINEAR light. A matrix assumes light adds linearly; applying
// one to gamma-encoded values is a classic bug producing wrong saturation
// everywhere, and it is very hard to spot by eye.
//
// Pointwise stage: 9 multiply-adds against 12 bytes read + 12 written per RGB
// pixel -- well under 1 FLOP/byte, an order of magnitude below the roofline
// ridge. On the GPU this is purely bandwidth-bound, which is the whole argument
// for fusing it with its pointwise neighbours in M4 rather than optimising it.
// ===========================================================================
void cameraToSRGB(ImageF32& rgb, const RawParams& p) {
    if (rgb.channels != 3)
        throw std::runtime_error("cameraToSRGB: expected 3 channels");

    // CIE XYZ -> linear sRGB, D65. Not measured but DEFINED: it follows entirely
    // from sRGB's primaries and white point. Display P3 has its own.
    static const float XYZ_TO_SRGB[9] = {
         3.2404542f, -1.5371385f, -0.4985314f,
        -0.9692660f,  1.8760108f,  0.0415560f,
         0.0556434f, -0.2040259f,  1.0572252f,
    };

    // Collapse both transforms into one 3x3 before the loop. Matrix multiplication
    // is associative -- (S*C)*v == S*(C*v) -- so this is exact, not an
    // approximation, and it halves the per-pixel work from 18 MACs to 9.
    float M[9];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            float s = 0.0f;
            for (int k = 0; k < 3; ++k)
                s += XYZ_TO_SRGB[r * 3 + k] * p.camToXYZ[k * 3 + c];
            M[r * 3 + c] = s;
        }

    for (std::size_t i = 0; i < rgb.pixelCount(); ++i) {
        float* px = &rgb.data[i * 3];

        // Read all three inputs BEFORE writing any output. Writing px[0] first
        // would corrupt the input to px[1] and px[2] -- the in-place hazard that
        // makes this kind of loop subtly wrong rather than obviously broken.
        const float r = px[0], g = px[1], b = px[2];

        px[0] = M[0] * r + M[1] * g + M[2] * b;
        px[1] = M[3] * r + M[4] * g + M[5] * b;
        px[2] = M[6] * r + M[7] * g + M[8] * b;

        // A negative channel means the colour falls OUTSIDE the sRGB gamut --
        // sRGB is a triangle inside the horseshoe of visible colour, and a camera
        // can capture beyond it. Clipping is the naive policy: it desaturates the
        // offending colour and loses the relationship between neighbouring
        // out-of-gamut values. Real gamut mapping compresses toward the gamut
        // boundary instead, preserving hue and relative saturation.
        for (int c = 0; c < 3; ++c)
            if (px[c] < 0.0f) px[c] = 0.0f;
    }
}

// ===========================================================================
// STAGE 4 -- Exposure and tone curve.
//
// Scene-referred linear values can exceed 1.0; a display cannot. Something must
// map one range onto the other, and a hard clip throws away every highlight.
//
// NOT IMPLEMENTED, and worth knowing as gaps:
//   * Highlight recovery. When one channel clips at the sensor and the others do
//     not, hue shifts -- blown skies drift cyan. Recovering it needs the
//     per-channel clip levels tracked through the white-balance gains, which this
//     pipeline does not carry.
//   * Local tone mapping. A single global curve cannot preserve local contrast
//     the way a local operator can; the cost is halo artifacts that then have to
//     be managed. Global is the right starting point, not the end state.
// ===========================================================================
void toneMap(ImageF32& rgb, float exposure, float whitePoint) {
    if (rgb.channels != 3)
        throw std::runtime_error("toneMap: expected 3 channels");

    const float W     = whitePoint > 0.0f ? whitePoint : 1.0f;
    const float invW2 = 1.0f / (W * W);

    for (std::size_t i = 0; i < rgb.pixelCount(); ++i) {
        float* px = &rgb.data[i * 3];

        // Exposure multiplies BEFORE the curve. It is a scene-referred scaling --
        // it simulates a different shutter or aperture, changing how much light
        // reached the sensor. Applied after the curve it would merely brighten
        // already-compressed values and crush the highlights flat.
        const float r = px[0] * exposure;
        const float g = px[1] * exposure;
        const float b = px[2] * exposure;

        // Rec.709 / sRGB luminance weights. Green dominates because human
        // sensitivity peaks near 555 nm; blue contributes least.
        const float L = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        if (L <= 0.0f) {
            px[0] = px[1] = px[2] = 0.0f;
            continue;
        }

        // Extended Reinhard.
        //
        // Plain Reinhard, L/(1+L), maps 1.0 -> 0.5 and darkens the whole image.
        // The white-point form maps W -> exactly 1.0, so W means "the scene
        // luminance that should read as display white".
        //
        // With W == 1 it is algebraically the identity: L*(1+L)/(1+L) == L. That
        // is the correct transform for display-referred input, which is why the
        // default leaves already-in-range images untouched instead of dimming
        // them. A tone curve is only meaningful when there is range to compress.
        const float Ld = L * (1.0f + L * invW2) / (1.0f + L);

        // Scale all three channels by ONE luminance ratio rather than curving
        // each independently. Per-channel tone mapping compresses the channels by
        // different amounts, which desaturates and SHIFTS HUE on saturated
        // colours -- a blown sky drifting cyan is the familiar symptom. Scaling
        // by a shared ratio preserves chromaticity exactly.
        const float s = Ld / L;
        px[0] = r * s;
        px[1] = g * s;
        px[2] = b * s;

        // Caveat of the hue-preserving form: an individual channel can still land
        // above 1.0 even when Ld <= 1.0 (a saturated primary carries little
        // luminance). encodeSRGB8 clips it. The principled fix is a desaturation
        // pass that pulls such pixels toward the achromatic axis until every
        // channel is in range -- trading saturation for detail rather than
        // trading hue for it.
    }
}

// ===========================================================================
// STAGE 5 -- Unsharp mask.
//
//     out = in + amount * (in - blur(in))
//
// (in - blur) is a high-pass; amount sets its gain. Runs on DISPLAY-REFERRED
// linear values -- after the tone curve, before the OETF. Sharpening
// scene-referred linear lets highlight overshoot scale with the raw magnitude and
// bloom after tone mapping; sharpening encoded values makes the halos nonlinear.
// Between the two is the right place.
//
// Fusion barrier: output(x,y) depends on a neighbourhood, so its ROI is the
// output rect inset by the kernel radius. It also CONTAMINATES a halo -- upstream
// border error is spread outward by exactly r pixels and amplified by `amount`.
// Stack two neighbourhood stages and the invalid margin grows by the sum of their
// radii; that arithmetic is what a tile scheduler has to get right.
//
// MEASURED, 0.8 MP: sigma 1 -> 29.8 ms, 2 -> 70.0, 4 -> 229.7, 8 -> 490.3.
// Separable convolution should scale as 2N, but 7x the taps costs 16.5x the time.
// The vertical pass is why: consecutive taps are width*3 floats apart -- roughly
// 12 KB here -- so every tap is its own cache line and the working set leaves L1
// as the radius grows. The horizontal pass reads adjacent memory and does not
// suffer. This is precisely what shared-memory tiling fixes on the GPU.
// ===========================================================================
void unsharpMask(ImageF32& rgb, float sigma, float amount) {
    if (rgb.channels != 3)
        throw std::runtime_error("unsharpMask: expected 3 channels");
    if (amount == 0.0f || sigma <= 0.0f) return;

    // Radius at 3 sigma. A Gaussian holds ~99.7% of its mass inside 3 sigma, so
    // further taps contribute less than float rounding error -- paying for them
    // buys nothing.
    const int r = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
    const int n = 2 * r + 1;

    std::vector<float> k(static_cast<std::size_t>(n));
    float ksum = 0.0f;
    const float inv2s2 = 1.0f / (2.0f * sigma * sigma);
    for (int i = -r; i <= r; ++i) {
        k[static_cast<std::size_t>(i + r)] = std::exp(-static_cast<float>(i * i) * inv2s2);
        ksum += k[static_cast<std::size_t>(i + r)];
    }
    // Normalise so the kernel sums to 1. Without this the blur would also scale
    // brightness, and the unsharp difference would carry a DC offset.
    for (float& v : k) v /= ksum;

    const int W = rgb.width, H = rgb.height;

    // SEPARABLE. A 2-D Gaussian factors into two 1-D Gaussians, so an NxN
    // convolution costing N^2 taps per pixel becomes two passes costing 2N.
    // At sigma=2, N=13: 169 taps versus 26, and the gap widens quadratically
    // with radius. The same factorisation is what makes the GPU version
    // shared-memory friendly -- each pass needs only a 1-D halo.
    ImageF32 tmp(W, H, 3);

    // Pass 1: horizontal blur into tmp.
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < 3; ++c) {
                float s = 0.0f;
                for (int i = -r; i <= r; ++i)
                    s += k[static_cast<std::size_t>(i + r)] * rgb.atClamped(x + i, y, c);
                tmp.at(x, y, c) = s;
            }

    // Pass 2: vertical blur, fused with the unsharp add so the fully blurred
    // image is never materialised.
    //
    // Writing into rgb while reading it is safe here only because the sole rgb
    // read is at (x,y,c) -- the very sample being overwritten. The neighbourhood
    // comes from tmp. Reading any neighbour of rgb would consume already-updated
    // values and silently corrupt the result.
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < 3; ++c) {
                float blur = 0.0f;
                for (int j = -r; j <= r; ++j)
                    blur += k[static_cast<std::size_t>(j + r)] * tmp.atClamped(x, y + j, c);

                // The classic formulation: add back a scaled copy of what the
                // blur removed. (in - blur) is a high-pass; amount sets its gain.
                const float in = rgb.at(x, y, c);
                const float v  = in + amount * (in - blur);

                // Overshoot at a hard edge can undershoot below zero. Negative
                // light is unphysical; the ceiling is left alone because values
                // above 1 are legitimate until encode clips them.
                rgb.at(x, y, c) = v < 0.0f ? 0.0f : v;
            }
}

// ===========================================================================
// STAGE 6 -- IMPLEMENTED. The other worked example.
// ===========================================================================
ImageU8 encodeSRGB8(const ImageF32& rgb) {
    if (rgb.channels != 3) throw std::runtime_error("encodeSRGB8: expected 3 channels");

    ImageU8 out(rgb.width, rgb.height, 3);
    for (std::size_t i = 0; i < rgb.sampleCount(); ++i) {
        float v = rgb.data[i];
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);

        // The sRGB OETF is NOT gamma 1/2.2. It has a linear segment near black --
        // a pure power function has infinite slope at 0, which would waste code
        // values and amplify sensor noise in the shadows -- and the curved part
        // uses 1/2.4 with an offset. The composite approximates 2.2 overall,
        // which is why the two get conflated. Knowing the difference is the kind
        // of detail a colour-pipeline interviewer notices.
        const float e = (v <= 0.0031308f) ? v * 12.92f
                                          : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;

        out.data[i] = static_cast<std::uint8_t>(e * 255.0f + 0.5f);
    }
    return out;
}

}  // namespace tinyci
