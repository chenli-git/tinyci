#include "tinyci/pipeline.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

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

// ===========================================================================
// STAGE 2a -- TODO. Bilinear demosaic.
//
// Every site holds exactly one of R/G/B; the other two must be interpolated
// from same-coloured neighbours.
//
//   GREEN occupies half the sites, in a quincunx. For a non-green pixel, average
//   the 4 edge-adjacent greens (N, S, E, W).
//
//   RED and BLUE each occupy a quarter, on a square lattice. For a pixel missing
//   red, exactly one of three cases applies:
//     - it sits between two reds horizontally  -> average those 2
//     - between two reds vertically            -> average those 2
//     - diagonally surrounded by four reds     -> average those 4
//   Blue is the same with the roles swapped. Use cfaColorAt() to decide which.
//
// Use atClamped() so border pixels do not read out of bounds.
//
// EXPECT IT TO LOOK BAD: zipper artifacts on horizontal edges and coloured
// fringes on high-contrast detail. That is the point -- keep the output, because
// the bilinear-vs-MHC comparison is the most persuasive image in your writeup.
// ===========================================================================
ImageF32 demosaicBilinear(const ImageF32& mosaic, CFAPattern cfa) {
    if (mosaic.channels != 1)
        throw std::runtime_error("demosaicBilinear: expected a 1-channel mosaic");

    ImageF32 out(mosaic.width, mosaic.height, 3);

    // Placeholder: replicate the mosaic sample to all three channels. This is
    // grayscale, not demosaic -- it exists only so the pipeline runs end to end
    // on day one. Expect a low PSNR; that number is your starting line.
    for (int y = 0; y < mosaic.height; ++y) {
        for (int x = 0; x < mosaic.width; ++x) {
            const float v = mosaic.at(x, y);
            out.at(x, y, 0) = v;
            out.at(x, y, 1) = v;
            out.at(x, y, 2) = v;
        }
    }
    (void)cfa;
    return out;
}

// ===========================================================================
// STAGE 2b -- TODO (stretch). Malvar-He-Cutler, ICASSP 2004.
//
// The insight: bilinear interpolates each channel in isolation, but luminance
// detail is shared across R, G and B -- a sharp edge appears in all three. So
// correct the bilinear estimate of one channel using the *Laplacian of the
// channel actually measured at that site*:
//
//     G_at_R  =  bilinear_G(x,y)  +  alpha * laplacian_R(x,y)
//
// It stays a fixed 5x5 linear filter (cheap, no branching -- which is exactly
// why it maps well to a GPU), but recovers most of the detail adaptive methods
// get. Gains from the paper: alpha = 1/2, beta = 5/8, gamma = 3/4, over eight
// 5x5 kernels covering the R-at-G / G-at-R / B-at-G / R-at-B ... cases.
//
// Look the kernels up in the paper rather than deriving them.
// ===========================================================================
ImageF32 demosaicMHC(const ImageF32& mosaic, CFAPattern cfa) {
    return demosaicBilinear(mosaic, cfa);  // TODO: replace
}

// ===========================================================================
// STAGE 3 -- TODO. Camera RGB -> linear sRGB.
//
// Two chained 3x3 matrices, per pixel:
//     XYZ        = camToXYZ   * cameraRGB
//     linear_sRGB = XYZ_TO_SRGB * XYZ
//
// Both operate in LINEAR light -- there must be no gamma anywhere in this
// function. Applying a matrix to gamma-encoded values is a classic bug that
// produces subtly wrong saturation everywhere and is very hard to spot by eye.
//
// XYZ_TO_SRGB (D65) is a standard constant, provided below.
//
// This is a pointwise stage: 9 multiply-adds against 16 bytes read + 16 written
// per RGBA pixel. Roughly 0.28 FLOP/byte -- an order of magnitude below the
// roofline ridge, so on the GPU it will be purely bandwidth-bound. That is the
// core argument for fusing it with its neighbours in M4.
// ===========================================================================
void cameraToSRGB(ImageF32& rgb, const RawParams& p) {
    static const float XYZ_TO_SRGB[9] = {
         3.2404542f, -1.5371385f, -0.4985314f,
        -0.9692660f,  1.8760108f,  0.0415560f,
         0.0556434f, -0.2040259f,  1.0572252f,
    };
    (void)rgb; (void)p; (void)XYZ_TO_SRGB;  // TODO
}

// ===========================================================================
// STAGE 4 -- TODO. Exposure, highlight handling, tone curve.
//
// Scene-referred linear values can exceed 1.0; a display cannot. Something has
// to map one range onto the other, and a hard clip throws away every highlight.
//
// Start simple -- Reinhard, v' = v / (1 + v) -- then consider:
//   * exposure: multiply before the curve, not after
//   * highlight recovery: when one channel clips and the others do not, the hue
//     shifts (blown skies go cyan). Even a crude fix is worth discussing.
//   * global vs local: a single curve cannot preserve local contrast the way a
//     local operator can. Know the tradeoff even if you only ship the global one.
// ===========================================================================
void toneMap(ImageF32& rgb, float exposure) {
    (void)rgb; (void)exposure;  // TODO
}

// ===========================================================================
// STAGE 5 -- TODO. Unsharp mask.
//
//     out = in + amount * (in - blur(in))
//
// Use a SEPARABLE Gaussian: an NxN convolution costs N^2 taps per pixel, but two
// 1-D passes cost 2N. At sigma=2 (N=13) that is 169 taps versus 26 -- and the
// same decomposition is what makes the GPU version shared-memory friendly in M5.
//
// Work in linear light. Sharpening gamma-encoded values produces dark halos.
//
// This stage is a fusion barrier: output(x,y) depends on a neighbourhood, so its
// ROI is the output rect inset by the kernel radius. When you get to M3, this is
// the stage whose tiles must overlap.
// ===========================================================================
void unsharpMask(ImageF32& rgb, float sigma, float amount) {
    (void)rgb; (void)sigma; (void)amount;  // TODO
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
