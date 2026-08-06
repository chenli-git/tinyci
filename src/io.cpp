#include "tinyci/io.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

namespace tinyci {
namespace {

// Read one integer from a PGM header, skipping whitespace and '#' comments.
// Consumes exactly one delimiter character after the digits -- which is also the
// single whitespace byte the format requires between the header and the payload.
int nextHeaderInt(std::istream& s) {
    int c = 0;
    for (;;) {
        c = s.get();
        if (c == EOF) throw std::runtime_error("truncated PGM header");
        if (c == '#') {
            while (c != '\n' && c != EOF) c = s.get();
            continue;
        }
        if (!std::isspace(c)) break;
    }
    int v = 0;
    while (c != EOF && std::isdigit(c)) {
        v = v * 10 + (c - '0');
        c = s.get();
    }
    return v;
}

}  // namespace

ImageU8 loadPng8(const std::string& path) {
    int w = 0, h = 0, comp = 0;
    unsigned char* p = stbi_load(path.c_str(), &w, &h, &comp, 3);
    if (!p) throw std::runtime_error("cannot read image: " + path);
    ImageU8 img(w, h, 3);
    std::copy(p, p + static_cast<std::size_t>(w) * h * 3, img.data.begin());
    stbi_image_free(p);
    return img;
}

void savePng8(const std::string& path, const ImageU8& img) {
    const int stride = img.width * img.channels;
    if (!stbi_write_png(path.c_str(), img.width, img.height, img.channels,
                        img.data.data(), stride))
        throw std::runtime_error("cannot write image: " + path);
}

ImageU16 loadPgm16(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);

    char m0 = 0, m1 = 0;
    f.get(m0).get(m1);
    if (m0 != 'P' || m1 != '5') throw std::runtime_error(path + ": not a binary PGM (P5)");

    const int w      = nextHeaderInt(f);
    const int h      = nextHeaderInt(f);
    const int maxval = nextHeaderInt(f);
    if (w <= 0 || h <= 0) throw std::runtime_error(path + ": bad dimensions");
    if (maxval != 65535) throw std::runtime_error(path + ": expected 16-bit PGM (maxval 65535)");

    ImageU16 img(w, h, 1);
    std::vector<unsigned char> raw(static_cast<std::size_t>(w) * h * 2);
    f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
    if (f.gcount() != static_cast<std::streamsize>(raw.size()))
        throw std::runtime_error(path + ": short read");

    // PGM stores 16-bit samples big-endian regardless of host byte order, so
    // assemble explicitly rather than memcpy-ing and hoping.
    for (std::size_t i = 0; i < img.data.size(); ++i)
        img.data[i] = static_cast<std::uint16_t>((raw[2 * i] << 8) | raw[2 * i + 1]);
    return img;
}

void savePgm16(const std::string& path, const ImageU16& img) {
    if (img.channels != 1) throw std::runtime_error("savePgm16: expected 1 channel");
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    f << "P5\n" << img.width << " " << img.height << "\n65535\n";
    std::vector<unsigned char> raw(img.data.size() * 2);
    for (std::size_t i = 0; i < img.data.size(); ++i) {
        raw[2 * i]     = static_cast<unsigned char>(img.data[i] >> 8);
        raw[2 * i + 1] = static_cast<unsigned char>(img.data[i] & 0xFF);
    }
    f.write(reinterpret_cast<const char*>(raw.data()),
            static_cast<std::streamsize>(raw.size()));
}

}  // namespace tinyci
