#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <cstdlib>

// ------------------------------------------------------------
// CUDA error checking
// ------------------------------------------------------------
#define CHECK(call)                                                   \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        std::cerr << "CUDA error: "                                   \
                  << cudaGetErrorString(err)                           \
                  << " at " << __FILE__ << ":" << __LINE__            \
                  << std::endl;                                       \
        std::exit(EXIT_FAILURE);                                       \
    }                                                                 \
} while (0)


// ------------------------------------------------------------
// 3x3 smoothing kernel
//
// One CUDA thread processes one output pixel.
// ------------------------------------------------------------
__global__
void smooth3x3(const float4* src,
               float4* dst,
               int width,
               int height)
{
    // --------------------------------------------------------
    // Find which pixel this thread is responsible for
    // --------------------------------------------------------
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Grid may be slightly larger than the image
    if (x >= width || y >= height)
        return;

    float sumB = 0.0f;
    float sumG = 0.0f;
    float sumR = 0.0f;

    // --------------------------------------------------------
    // Visit the 3x3 neighborhood
    //
    // (-1,-1) (0,-1) (1,-1)
    // (-1, 0) (0, 0) (1, 0)
    // (-1, 1) (0, 1) (1, 1)
    // --------------------------------------------------------
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            int nx = x + dx;
            int ny = y + dy;

            // Clamp boundary coordinates
            nx = max(0, min(nx, width  - 1));
            ny = max(0, min(ny, height - 1));

            // Convert 2D coordinate → 1D memory index
            int index = ny * width + nx;

            float4 p = src[index];

            // OpenCV uses BGRA ordering here
            sumB += p.x;
            sumG += p.y;
            sumR += p.z;
        }
    }

    // Average 9 pixels
    float inv9 = 1.0f / 9.0f;

    int outputIndex = y * width + x;

    // Keep alpha unchanged
    float alpha = src[outputIndex].w;

    dst[outputIndex] = make_float4(
        sumB * inv9,
        sumG * inv9,
        sumR * inv9,
        alpha
    );
}


// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main(int argc, char** argv)
{
    // --------------------------------------------------------
    // Read command-line arguments
    // --------------------------------------------------------
    if (argc < 3)
    {
        std::cout
            << "Usage:\n"
            << "  ./smooth_png input.png output.png\n";

        return 0;
    }

    const std::string inputPath  = argv[1];
    const std::string outputPath = argv[2];


    // ========================================================
    // STEP 1: Load PNG on CPU
    // ========================================================

    cv::Mat imageBGR =
        cv::imread(inputPath, cv::IMREAD_COLOR);

    if (imageBGR.empty())
    {
        std::cerr
            << "Failed to open image: "
            << inputPath
            << std::endl;

        return 1;
    }

    int width  = imageBGR.cols;
    int height = imageBGR.rows;

    std::cout
        << "Image size: "
        << width
        << " x "
        << height
        << std::endl;


    // ========================================================
    // STEP 2: Convert BGR → BGRA
    //
    // We use 4 channels because float4 is 16 bytes/pixel.
    // ========================================================

    cv::Mat imageBGRA;

    cv::cvtColor(
        imageBGR,
        imageBGRA,
        cv::COLOR_BGR2BGRA
    );


    // ========================================================
    // STEP 3: Convert uint8 pixels → float pixels
    //
    // Original:
    //     0 ... 255
    //
    // GPU:
    //     0.0 ... 1.0
    // ========================================================

    cv::Mat imageFloat;

    imageBGRA.convertTo(
        imageFloat,
        CV_32FC4,
        1.0 / 255.0
    );


    // Make sure memory is contiguous
    if (!imageFloat.isContinuous())
    {
        imageFloat = imageFloat.clone();
    }


    // ========================================================
    // STEP 4: Calculate GPU memory size
    // ========================================================

    size_t numPixels =
        static_cast<size_t>(width) *
        static_cast<size_t>(height);

    size_t bytes =
        numPixels * sizeof(float4);

    std::cout
        << "Pixels: "
        << numPixels
        << std::endl;

    std::cout
        << "GPU bytes per image: "
        << bytes
        << std::endl;


    // ========================================================
    // STEP 5: Create GPU pointers
    // ========================================================

    float4* d_src = nullptr;
    float4* d_dst = nullptr;


    // ========================================================
    // STEP 6: Allocate GPU memory
    // ========================================================

    CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_src),
        bytes
    ));

    CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_dst),
        bytes
    ));


    // ========================================================
    // STEP 7: Copy image CPU → GPU
    // ========================================================

    CHECK(cudaMemcpy(
        d_src,
        imageFloat.ptr<float>(),
        bytes,
        cudaMemcpyHostToDevice
    ));


    // ========================================================
    // STEP 8: Define CUDA block
    //
    // 16 × 16 = 256 threads/block
    // ========================================================

    dim3 block(
        16,
        16
    );


    // ========================================================
    // STEP 9: Calculate grid size
    // ========================================================

    dim3 grid(
        (width  + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    std::cout
        << "Block: "
        << block.x
        << " x "
        << block.y
        << std::endl;

    std::cout
        << "Grid: "
        << grid.x
        << " x "
        << grid.y
        << std::endl;


    // ========================================================
    // STEP 10: Launch CUDA kernel
    // ========================================================

    smooth3x3<<<grid, block>>>(
        d_src,
        d_dst,
        width,
        height
    );


    // Check launch error
    CHECK(cudaGetLastError());


    // Wait until GPU finishes
    CHECK(cudaDeviceSynchronize());


    // ========================================================
    // STEP 11: Create CPU output image
    // ========================================================

    cv::Mat outputFloat(
        height,
        width,
        CV_32FC4
    );


    // ========================================================
    // STEP 12: Copy GPU → CPU
    // ========================================================

    CHECK(cudaMemcpy(
        outputFloat.ptr<float>(),
        d_dst,
        bytes,
        cudaMemcpyDeviceToHost
    ));


    // ========================================================
    // STEP 13: Convert float [0,1] → uint8 [0,255]
    // ========================================================

    cv::Mat outputBGRA;

    outputFloat.convertTo(
        outputBGRA,
        CV_8UC4,
        255.0
    );


    // ========================================================
    // STEP 14: BGRA → BGR
    // ========================================================

    cv::Mat outputBGR;

    cv::cvtColor(
        outputBGRA,
        outputBGR,
        cv::COLOR_BGRA2BGR
    );


    // ========================================================
    // STEP 15: Save PNG
    // ========================================================

    bool success =
        cv::imwrite(
            outputPath,
            outputBGR
        );

    if (!success)
    {
        std::cerr
            << "Failed to save output image."
            << std::endl;

        return 1;
    }


    // ========================================================
    // STEP 16: Free GPU memory
    // ========================================================

    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));


    std::cout
        << "Saved smoothed image to: "
        << outputPath
        << std::endl;

    return 0;
}
