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
    // Identity is a placeholder. Real cameras ship a measured matrix per
    // illuminant, because the sensor's spectral response is not the eye's.
    float camToXYZ[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

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

// 4. Exposure, highlight handling, tone curve, in place.           [pointwise]
void toneMap(ImageF32& rgb, float exposure);

// 5. Unsharp mask, in place.        [NEIGHBOURHOOD, separable -- fusion barrier]
void unsharpMask(ImageF32& rgb, float sigma, float amount);

// 6. linear sRGB -> sRGB-encoded 8-bit.                            [pointwise]
ImageU8 encodeSRGB8(const ImageF32& rgb);

}  // namespace tinyci
