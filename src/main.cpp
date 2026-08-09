#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>

#include "tinyci/image.h"
#include "tinyci/io.h"
#include "tinyci/pipeline.h"

using namespace tinyci;

namespace {

// Per-stage wall-clock timing. Every optimisation this week is justified by a
// number, so measurement is scaffolding, not an afterthought.
struct Timer {
    using clock = std::chrono::steady_clock;
    clock::time_point t0 = clock::now();
    double lapMs() {
        const auto t1 = clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        t0 = t1;
        return ms;
    }
};

CFAPattern parseCFA(const std::string& s) {
    if (s == "RGGB") return CFAPattern::RGGB;
    if (s == "BGGR") return CFAPattern::BGGR;
    if (s == "GRBG") return CFAPattern::GRBG;
    if (s == "GBRG") return CFAPattern::GBRG;
    throw std::runtime_error("unknown CFA pattern: " + s);
}

void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s <bayer.pgm> <out.png> [options]\n"
        "\n"
        "  --cfa PAT      RGGB | BGGR | GRBG | GBRG      (default RGGB)\n"
        "  --black N      black level, sensor codes      (default 0)\n"
        "  --white N      saturation level               (default 65535)\n"
        "  --wb R G B     white-balance gains            (default 1 1 1)\n"
        "  --exposure X   linear exposure multiplier     (default 1)\n"
        "  --white-point X  scene luminance mapping to display white; 1 = identity\n"
        "  --mhc          Malvar-He-Cutler demosaic      (default bilinear)\n"
        "  --sigma X      unsharp radius                 (default 1.0)\n"
        "  --amount X     unsharp strength, 0 = off      (default 0)\n",
        prog);
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc < 3) { usage(argv[0]); return 2; }

    const std::string inPath  = argv[1];
    const std::string outPath = argv[2];

    RawParams p;
    bool  useMHC = false;
    float sigma      = 1.0f;
    float amount     = 0.0f;
    float whitePoint = 1.0f;

    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](int n) {
            if (i + n >= argc) throw std::runtime_error("missing argument after " + a);
        };
        if      (a == "--cfa")      { need(1); p.cfa = parseCFA(argv[++i]); }
        else if (a == "--black")    { need(1); p.blackLevel = std::stof(argv[++i]); }
        else if (a == "--white")    { need(1); p.whiteLevel = std::stof(argv[++i]); }
        else if (a == "--exposure") { need(1); p.exposure   = std::stof(argv[++i]); }
        else if (a == "--white-point") { need(1); whitePoint = std::stof(argv[++i]); }
        else if (a == "--sigma")    { need(1); sigma        = std::stof(argv[++i]); }
        else if (a == "--amount")   { need(1); amount       = std::stof(argv[++i]); }
        else if (a == "--mhc")      { useMHC = true; }
        else if (a == "--wb")       {
            need(3);
            p.wbGain[0] = std::stof(argv[++i]);
            p.wbGain[1] = std::stof(argv[++i]);
            p.wbGain[2] = std::stof(argv[++i]);
        }
        else { usage(argv[0]); throw std::runtime_error("unknown option: " + a); }
    }

    Timer t;
    const ImageU16 mosaic = loadPgm16(inPath);
    const double tLoad = t.lapMs();

    std::printf("input   : %s  (%dx%d, %.1f MP)\n", inPath.c_str(),
                mosaic.width, mosaic.height, mosaic.pixelCount() / 1e6);

    ImageF32 lin = linearizeAndWhiteBalance(mosaic, p);
    const double tLin = t.lapMs();

    ImageF32 rgb = useMHC ? demosaicMHC(lin, p.cfa) : demosaicBilinear(lin, p.cfa);
    const double tDem = t.lapMs();

    cameraToSRGB(rgb, p);
    const double tMat = t.lapMs();

    toneMap(rgb, p.exposure, whitePoint);
    const double tTone = t.lapMs();

    if (amount > 0.0f) unsharpMask(rgb, sigma, amount);
    const double tSharp = t.lapMs();

    const ImageU8 out = encodeSRGB8(rgb);
    const double tEnc = t.lapMs();

    savePng8(outPath, out);
    const double tSave = t.lapMs();

    const double total = tLin + tDem + tMat + tTone + tSharp + tEnc;
    std::printf(
        "\n  stage                       ms\n"
        "  --------------------------------\n"
        "  load (%s)                %7.2f\n"
        "  linearize + WB           %7.2f\n"
        "  demosaic (%-9s)      %7.2f\n"
        "  camera -> sRGB           %7.2f\n"
        "  tone map                 %7.2f\n"
        "  unsharp                  %7.2f\n"
        "  encode sRGB8             %7.2f\n"
        "  --------------------------------\n"
        "  pipeline total           %7.2f\n"
        "  save                     %7.2f\n"
        "\noutput  : %s\n",
        "pgm", tLoad, tLin, useMHC ? "MHC" : "bilinear", tDem, tMat, tTone,
        tSharp, tEnc, total, tSave, outPath.c_str());

    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
