#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tinyci {

// Bayer colour-filter-array layouts, named by the 2x2 tile read left-to-right,
// top-to-bottom starting at pixel (0,0).
enum class CFAPattern { RGGB, BGGR, GRBG, GBRG };

// Which colour sits at (x,y) under a given CFA:  0 = R, 1 = G, 2 = B.
// Note G occupies half of all sites (the quincunx), R and B a quarter each --
// that asymmetry is why demosaic treats green differently from red and blue.
inline int cfaColorAt(CFAPattern p, int x, int y) {
    const int i = (y & 1) * 2 + (x & 1);   // 0=TL 1=TR 2=BL 3=BR
    switch (p) {
        case CFAPattern::RGGB: { static const int m[4] = {0, 1, 1, 2}; return m[i]; }
        case CFAPattern::BGGR: { static const int m[4] = {2, 1, 1, 0}; return m[i]; }
        case CFAPattern::GRBG: { static const int m[4] = {1, 0, 2, 1}; return m[i]; }
        case CFAPattern::GBRG: { static const int m[4] = {1, 2, 0, 1}; return m[i]; }
    }
    return 1;
}

// An owning 2D buffer with `channels` interleaved samples per pixel.
//
// Deliberately plain. std::vector owns the memory, so copy, move, and destruction
// are all correct without writing a destructor, copy constructor, or assignment
// operator ("Rule of Zero"). Interleaved rather than planar because that is what
// GPUs want later: one coalesced 16-byte load per thread for RGBA float.
template <typename T>
struct Image {
    int width    = 0;
    int height   = 0;
    int channels = 0;
    std::vector<T> data;

    Image() = default;
    Image(int w, int h, int c)
        : width(w), height(h), channels(c),
          data(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * c) {}

    std::size_t pixelCount()  const { return static_cast<std::size_t>(width) * height; }
    std::size_t sampleCount() const { return pixelCount() * channels; }
    bool        empty()       const { return data.empty(); }

    // Unchecked sample access. This is the inner loop; bounds checks belong in
    // your head, not in the hot path.
    T& at(int x, int y, int c = 0) {
        return data[(static_cast<std::size_t>(y) * width + x) * channels + c];
    }
    const T& at(int x, int y, int c = 0) const {
        return data[(static_cast<std::size_t>(y) * width + x) * channels + c];
    }

    // Edge-clamped access, for neighbourhood operations at the image border.
    // Clamping replicates the edge pixel; it is the cheapest boundary policy and
    // the one Core Image uses for `clampedToExtent`.
    const T& atClamped(int x, int y, int c = 0) const {
        x = x < 0 ? 0 : (x >= width  ? width  - 1 : x);
        y = y < 0 ? 0 : (y >= height ? height - 1 : y);
        return at(x, y, c);
    }
};

using ImageU8  = Image<std::uint8_t>;
using ImageU16 = Image<std::uint16_t>;
using ImageF32 = Image<float>;

}  // namespace tinyci
