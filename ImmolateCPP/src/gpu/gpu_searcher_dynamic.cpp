// GPU Searcher implementation with dynamic CUDA loading
// Allows cross-compilation from Linux to Windows

#include "gpu_searcher.hpp"
#include "cuda_wrapper.hpp"
#include <iostream>
#include <cstring>
#include <chrono>
#include <fstream>
#include <vector>

// Global CUDA wrapper instance
CudaWrapper g_cuda;

#ifdef GPU_ENABLED

// Load PTX file at runtime
static std::string load_ptx_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return "";
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::string content(size, '\0');
    file.read(&content[0], size);
    
    return content;
}

GPUSearcher::GPUSearcher() : initialized(false), device_id(0) {
    // Try to initialize CUDA dynamically
    if (!g_cuda.init()) {
        std::cerr << "[GPU] CUDA runtime not available" << std::endl;
        return;
    }
    
    // Check for CUDA devices
    int device_count = 0;
    cudaError_t err = g_cuda.cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "[GPU] No CUDA devices found" << std::endl;
        return;
    }
    
    // Set device
    err = g_cuda.cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "[GPU] Failed to set device: " << g_cuda.cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Allocate device memory
    err = g_cuda.cudaMalloc(&d_params, sizeof(FilterParams));
    if (err != cudaSuccess) {
        std::cerr << "[GPU] Failed to allocate params: " << g_cuda.cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = g_cuda.cudaMalloc(&d_result, sizeof(uint64_t));
    if (err != cudaSuccess) {
        std::cerr << "[GPU] Failed to allocate result: " << g_cuda.cudaGetErrorString(err) << std::endl;
        g_cuda.cudaFree(d_params);
        return;
    }
    
    err = g_cuda.cudaMalloc(&d_found, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "[GPU] Failed to allocate found flag: " << g_cuda.cudaGetErrorString(err) << std::endl;
        g_cuda.cudaFree(d_params);
        g_cuda.cudaFree(d_result);
        return;
    }
    
    initialized = true;
    std::cout << "[GPU] Initialized successfully with device " << device_id << std::endl;
}

GPUSearcher::~GPUSearcher() {
    if (initialized && g_cuda.is_available()) {
        g_cuda.cudaFree(d_params);
        g_cuda.cudaFree(d_result);
        g_cuda.cudaFree(d_found);
    }
    g_cuda.cleanup();
}

std::string GPUSearcher::search(const std::string& start_seed, const FilterParams& params) {
    if (!initialized || !g_cuda.is_available()) {
        return "";  // Return empty string to indicate no match found
    }
    
    // Reset found flag
    int zero = 0;
    cudaError_t err = g_cuda.cudaMemcpy(d_found, &zero, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "[GPU] Failed to reset found flag" << std::endl;
        return "";
    }
    
    // Copy params to device
    err = g_cuda.cudaMemcpy(d_params, &params, sizeof(FilterParams), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "[GPU] Failed to copy params" << std::endl;
        return "";
    }
    
    // Note: In a real implementation, we would load and launch the PTX kernel here
    // For now, we'll return empty to fall back to CPU
    std::cerr << "[GPU] PTX kernel launching not yet implemented - falling back to CPU" << std::endl;
    return "";
}

int GPUSearcher::get_compute_capability() const {
    if (!initialized || !g_cuda.is_available()) {
        return 0;
    }
    
    cudaDeviceProp prop;
    cudaError_t err = g_cuda.cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        return 0;
    }
    
    return prop.major * 10 + prop.minor;
}

size_t GPUSearcher::get_memory_size() const {
    if (!initialized || !g_cuda.is_available()) {
        return 0;
    }
    
    cudaDeviceProp prop;
    cudaError_t err = g_cuda.cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        return 0;
    }
    
    return prop.totalGlobalMem;
}

int GPUSearcher::get_sm_count() const {
    if (!initialized || !g_cuda.is_available()) {
        return 0;
    }
    
    cudaDeviceProp prop;
    cudaError_t err = g_cuda.cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        return 0;
    }
    
    return prop.multiProcessorCount;
}

#else

// CPU-only stub implementation
GPUSearcher::GPUSearcher() : initialized(false), device_id(0) {
    std::cout << "[GPU] GPU support not compiled in" << std::endl;
}

GPUSearcher::~GPUSearcher() {}

std::string GPUSearcher::search(const std::string& start_seed, const FilterParams& params) {
    return "";  // No GPU support
}

int GPUSearcher::get_compute_capability() const {
    return 0;
}

size_t GPUSearcher::get_memory_size() const {
    return 0;
}

int GPUSearcher::get_sm_count() const {
    return 0;
}

#endif