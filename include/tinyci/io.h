#pragma once

#include <string>

#include "tinyci/image.h"

namespace tinyci {

// 8-bit RGB PNG (via stb_image). Always returns 3 channels.
ImageU8 loadPng8(const std::string& path);
void    savePng8(const std::string& path, const ImageU8& img);

// 16-bit binary PGM (P5), single channel -- the container for our Bayer mosaics.
// PGM is used rather than PNG because it is trivially parseable and stores 16-bit
// samples without any colour-management metadata to misinterpret. Sensor data is
// raw counts, not colour, so a format with no colour semantics is the right choice.
ImageU16 loadPgm16(const std::string& path);
void     savePgm16(const std::string& path, const ImageU16& img);

}  // namespace tinyci
