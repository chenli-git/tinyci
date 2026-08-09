#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace tinyci {

// ---------------------------------------------------------------------------
// Error checking.
//
// Every CUDA API call returns a status and almost every one can fail. Worse,
// failures are ASYNCHRONOUS: an error raised inside a kernel surfaces at some
// later API call, so the line the runtime blames is usually innocent. Checking
// every call is the only way to keep the blame close to the crime.
// ---------------------------------------------------------------------------
inline void cudaCheckImpl(cudaError_t e, const char* expr, const char* file, int line) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e) +
                                 "\n  at " + file + ":" + std::to_string(line) +
                                 "\n  in " + expr);
}

#define CUDA_CHECK(expr) ::tinyci::cudaCheckImpl((expr), #expr, __FILE__, __LINE__)

// Call immediately after a kernel launch. Launches return void, so this is the
// only way to catch a bad launch configuration; the sync then surfaces errors
// raised during execution.
inline void cudaCheckKernel(const char* what, const char* file, int line) {
    cudaCheckImpl(cudaGetLastError(), what, file, line);
    cudaCheckImpl(cudaDeviceSynchronize(), what, file, line);
}

#define CUDA_CHECK_KERNEL(what) ::tinyci::cudaCheckKernel(what, __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// Owning device allocation. Same Rule-of-Five discipline as any C++ resource:
// cudaMalloc/cudaFree are malloc/free, and leaking device memory on an 8 GB card
// is noticed quickly.
//
// Copyable would be a trap (two owners, one pointer, double free), so copying is
// deleted and only moves are allowed.
// ---------------------------------------------------------------------------
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { allocate(count); }

    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), count_(o.count_) {
        o.ptr_ = nullptr;
        o.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            release();
            ptr_ = o.ptr_; count_ = o.count_;
            o.ptr_ = nullptr; o.count_ = 0;
        }
        return *this;
    }

    void allocate(std::size_t count) {
        release();
        if (count) CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        count_ = count;
    }

    void release() {
        if (ptr_) cudaFree(ptr_);   // no CUDA_CHECK: destructors must not throw
        ptr_ = nullptr;
        count_ = 0;
    }

    void upload(const T* host, std::size_t count) {
        CUDA_CHECK(cudaMemcpy(ptr_, host, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void upload(const std::vector<T>& host) { upload(host.data(), host.size()); }

    void download(T* host, std::size_t count) const {
        CUDA_CHECK(cudaMemcpy(host, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void download(std::vector<T>& host) const { download(host.data(), host.size()); }

    T*          get()        { return ptr_; }
    const T*    get()  const { return ptr_; }
    std::size_t size() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

private:
    T*          ptr_   = nullptr;
    std::size_t count_ = 0;
};

// ---------------------------------------------------------------------------
// GPU-side timing.
//
// std::chrono cannot time a kernel: launches are ASYNCHRONOUS and return before
// the GPU has done anything, so a host timer measures the launch call and reports
// microseconds. CUDA events are timestamped by the GPU itself, in its own stream
// order, which is what you actually want.
// ---------------------------------------------------------------------------
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    CudaTimer(const CudaTimer&)            = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    void start() { CUDA_CHECK(cudaEventRecord(start_)); }

    // Returns milliseconds. Blocks until the GPU reaches the stop event.
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{}, stop_{};
};

// Bytes moved / seconds elapsed, in GB/s. Quote this as a percentage of the
// measured peak from bwtest (377.5 GB/s on the reference machine) -- never as a
// percentage of the spec sheet, which is unreachable.
inline double gbPerSec(double bytes, float ms) {
    return bytes / (static_cast<double>(ms) * 1e-3) / 1e9;
}

// Ceiling division: the grid must cover every element, so it usually overshoots
// and surplus threads must guard themselves out.
inline int gridFor(std::size_t n, int block) {
    return static_cast<int>((n + static_cast<std::size_t>(block) - 1) / block);
}

}  // namespace tinyci
