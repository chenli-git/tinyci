cmake_minimum_required(VERSION 3.22)
project(tinyci LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# RTX 5060 / Blackwell. A binary built for another arch will fail at launch with
# "no kernel image is available for execution on the device".
set(CMAKE_CUDA_ARCHITECTURES 120)

# Emit build/compile_commands.json so the editor knows your include paths.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---- M0: host reference pipeline -------------------------------------------
add_library(tinyci_core STATIC
    src/io.cpp
    src/pipeline.cpp
)
target_include_directories(tinyci_core PUBLIC include)
# SYSTEM: stb is third-party single-header code that does not compile clean under
# -Wextra. Marking it system suppresses its warnings without hiding yours.
target_include_directories(tinyci_core SYSTEM PUBLIC third_party)
target_compile_options(tinyci_core PRIVATE -O3 -Wall -Wextra)

add_executable(tinyci src/main.cpp)
target_link_libraries(tinyci PRIVATE tinyci_core)
target_compile_options(tinyci PRIVATE -O3 -Wall -Wextra)

# ---- benchmarks -------------------------------------------------------------
add_executable(bwtest bench/bwtest.cu)
# -lineinfo lets Nsight Compute attribute stalls to specific source lines.
target_compile_options(bwtest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)

# M1: the fusion experiment -- the project's central measurement.
add_executable(fusion bench/fusion.cu)
target_include_directories(fusion PRIVATE include)
target_compile_options(fusion PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)

# Teaching benchmarks: coalescing, warp divergence, precision.
add_executable(experiments bench/experiments.cu)
target_include_directories(experiments PRIVATE include)
target_compile_options(experiments PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)
