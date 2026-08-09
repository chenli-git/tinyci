#pragma once

#include "tinyci/image.h"

namespace tinyci {

struct RawParams {
    CFAPattern cfa = CFAPattern::RGGB;

    // Sensor code for "no light". Never zero on real hardware: dark current and
    // amplifier bias put the floor somewhere above 0, and subtracting it is the
    // first thing any raw pipeline does.
    float blackLevel = 0.0f;

    // Sensor code at saturation. Above this the photosite has clipped and the
    // value carries no information -- which is where highlight recovery starts.
    float whiteLevel = 65535.0f;

    // Per-channel white-balance gains (R, G, B).
    float wbGain[3] = {1.0f, 1.0f, 1.0f};

    // 3x3 row-major: camera-native RGB -> CIE XYZ (D65).
    //
    // A real camera ships a MEASURED matrix, and one per illuminant, because the
    // sensor's spectral response is not the eye's. It is obtained by shooting a
    // colour target under known light and solving for the fit.
    //
    // The default below is sRGB -> XYZ (D65), which is the honest profile for our
    // *synthetic* camera: tools/bayer.py builds the mosaic by linearising an sRGB
    // PNG, so the simulated sensor has exactly sRGB primaries. Chaining it with
    // XYZ_TO_SRGB therefore collapses to identity and the test images round-trip.
    // Substitute a real camera's matrix and the stage does real work.
    float camToXYZ[9] = {
        0.4124564f, 0.3575761f, 0.1804375f,
        0.2126729f, 0.7151522f, 0.0721750f,
        0.0193339f, 0.1191920f, 0.9503041f,
    };

    float exposure = 1.0f;
};

// ---------------------------------------------------------------------------
// Pipeline stages, in order.
//
// Note the calling convention, which is not arbitrary: stages that change the
// pixel *shape* return a new image, while stages that only transform values
// operate in place. The in-place ones are exactly the pointwise stages -- the
// ones you will fuse into a single GPU pass in M4. Keeping that distinction
// visible in the signatures is the whole reason the fusion boundary is obvious
// later.
// ---------------------------------------------------------------------------

// 1. u16 mosaic -> f32 mosaic. Black-level subtract, normalise to [0,1], apply WB.
ImageF32 linearizeAndWhiteBalance(const ImageU16& mosaic, const RawParams& p);


// 2. f32 mosaic (1ch) -> f32 linear camera RGB (3ch).  [NEIGHBOURHOOD -- fusion barrier]
ImageF32 demosaicBilinear(const ImageF32& mosaic, CFAPattern cfa);
ImageF32 demosaicMHC(const ImageF32& mosaic, CFAPattern cfa);

// 3. camera RGB -> linear sRGB, in place.                          [pointwise]
void cameraToSRGB(ImageF32& rgb, const RawParams& p);

// 4. Exposure and tone curve, in place.                            [pointwise]
//
// whitePoint is the scene luminance that should read as display white. 1.0 makes
// the curve algebraically the identity -- correct when the input is already
// display-referred, as our synthetic test data is. Values above 1 compress real
// highlight headroom into range.
void toneMap(ImageF32& rgb, float exposure, float whitePoint = 1.0f);

// 5. Unsharp mask, in place.        [NEIGHBOURHOOD, separable -- fusion barrier]
void unsharpMask(ImageF32& rgb, float sigma, float amount);

// 6. linear sRGB -> sRGB-encoded 8-bit.                            [pointwise]
ImageU8 encodeSRGB8(const ImageF32& rgb);

}  // namespace tinyci
