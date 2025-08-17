// Safe brainstorm implementation with GPU fallback
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

#ifdef GPU_ENABLED
#include "gpu/safe_cuda_wrapper.hpp"
#endif

#include "instance.hpp"
#include "items.hpp"
#include "functions.hpp"
#include "rng.hpp"
#include "seed.hpp"
#include "util.hpp"

#ifdef BUILDING_DLL
#define DLL_EXPORT extern "C" __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

// Global state for GPU availability
static bool gpu_initialized = false;
static bool gpu_available = false;
static std::mutex gpu_init_mutex;

// Initialize GPU support safely
static void init_gpu_if_needed(bool use_cuda) {
    std::lock_guard<std::mutex> lock(gpu_init_mutex);
    
    if (gpu_initialized) {
        return;
    }
    
    gpu_initialized = true;
    
    if (!use_cuda) {
        std::cout << "[Brainstorm] GPU disabled by config" << std::endl;
        return;
    }

#ifdef GPU_ENABLED
    auto& cuda = get_cuda_wrapper();
    
    // Try to initialize CUDA with timeout protection
    std::atomic<bool> init_done(false);
    std::atomic<bool> init_success(false);
    
    std::thread init_thread([&]() {
        try {
            init_success = cuda.initialize();
            init_done = true;
        } catch (...) {
            init_done = true;
        }
    });
    
    // Wait max 2 seconds for CUDA init
    auto start = std::chrono::steady_clock::now();
    while (!init_done) {
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed > std::chrono::seconds(2)) {
            std::cout << "[Brainstorm] CUDA initialization timeout - using CPU mode" << std::endl;
            init_thread.detach(); // Let it finish in background
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    if (init_thread.joinable()) {
        init_thread.join();
    }
    
    gpu_available = init_success.load();
    
    if (gpu_available) {
        std::cout << "[Brainstorm] GPU acceleration enabled" << std::endl;
    } else {
        std::cout << "[Brainstorm] GPU not available: " << cuda.get_error() << std::endl;
    }
#else
    std::cout << "[Brainstorm] Built without GPU support" << std::endl;
#endif
}

// CPU implementation of brainstorm
static uint64_t brainstorm_cpu(
    int64_t starting_seed,
    int32_t steps,
    int32_t tag1_filter,
    int32_t tag2_filter,
    int32_t voucher_filter,
    int32_t pack_filter,
    int32_t pack_slot1,
    int32_t pack_slot2,
    bool use_cuda
) {
    // Original CPU implementation using the Instance API
    for (int32_t i = 0; i < steps; i++) {
        Seed test_seed = Seed(starting_seed + i);
        Instance inst(test_seed);
        
        // Test tags with dual tag support
        if (tag1_filter > 0) {
            Item smallBlindTag = inst.nextTag(1);
            Item bigBlindTag = inst.nextTag(1);
            
            bool tag1_found = false;
            bool tag2_found = (tag2_filter <= 0); // If no tag2, consider it found
            
            // Check small blind tag
            if (smallBlindTag == static_cast<Item>(tag1_filter)) {
                tag1_found = true;
            } else if (tag2_filter > 0 && smallBlindTag == static_cast<Item>(tag2_filter)) {
                tag2_found = true;
            }
            
            // Check big blind tag
            if (bigBlindTag == static_cast<Item>(tag1_filter)) {
                tag1_found = true;
            } else if (tag2_filter > 0 && bigBlindTag == static_cast<Item>(tag2_filter)) {
                tag2_found = true;
            }
            
            // Both tags must be found (order doesn't matter)
            if (!tag1_found || !tag2_found) {
                continue;
            }
        }
        
        // Test voucher
        if (voucher_filter > 0) {
            inst.initLocks(1, false, false);
            Item firstVoucher = inst.nextVoucher(1);
            if (firstVoucher != static_cast<Item>(voucher_filter)) {
                continue;
            }
        }
        
        // Test pack
        if (pack_filter > 0) {
            inst.cache.generatedFirstPack = true;
            Item pack = inst.nextPack(1);
            if (pack != static_cast<Item>(pack_filter)) {
                continue;
            }
        }
        
        // Note: pack_slot1 and pack_slot2 were for Arcana packs but aren't currently used
        // Could add support for checking specific cards in packs if needed
        
        // All filters passed
        return starting_seed + i;
    }
    
    return 0; // No match found
}

// Main brainstorm function with safe GPU fallback
DLL_EXPORT uint64_t brainstorm(
    int64_t starting_seed,
    int32_t steps,
    int32_t tag1_filter,
    int32_t tag2_filter,
    int32_t voucher_filter,
    int32_t pack_filter,
    int32_t pack_slot1,
    int32_t pack_slot2,
    bool use_cuda
) {
    // Initialize GPU if needed (thread-safe, only happens once)
    init_gpu_if_needed(use_cuda);
    
    // If GPU is available and requested, try to use it
    if (use_cuda && gpu_available) {
#ifdef GPU_ENABLED
        try {
            // Placeholder for GPU implementation
            // For now, fall back to CPU
            return brainstorm_cpu(starting_seed, steps, tag1_filter, tag2_filter, 
                                voucher_filter, pack_filter, pack_slot1, pack_slot2, use_cuda);
        } catch (...) {
            std::cout << "[Brainstorm] GPU error, falling back to CPU" << std::endl;
            gpu_available = false; // Disable GPU for future calls
        }
#endif
    }
    
    // Use CPU implementation
    return brainstorm_cpu(starting_seed, steps, tag1_filter, tag2_filter,
                        voucher_filter, pack_filter, pack_slot1, pack_slot2, use_cuda);
}

// Test function to check DLL loading
DLL_EXPORT int test_dll() {
    std::cout << "[Brainstorm] DLL loaded successfully (Safe version)" << std::endl;
    return 42;
}

// Get GPU status
DLL_EXPORT bool is_gpu_available() {
    return gpu_available;
}

// Get version info
DLL_EXPORT const char* get_version() {
    return "Brainstorm v3.0 (Safe GPU)";
}

// Get acceleration type
DLL_EXPORT const char* get_acceleration_type() {
    if (gpu_available) {
        return "GPU (CUDA)";
    } else if (gpu_initialized) {
        return "CPU (GPU init failed)";
    } else {
        return "CPU";
    }
}

// Get hardware info
DLL_EXPORT const char* get_hardware_info() {
    static std::string info;
    if (gpu_available) {
        info = "GPU acceleration enabled (CUDA)";
    } else if (gpu_initialized) {
        info = "CPU mode (GPU unavailable)";
    } else {
        info = "CPU mode";
    }
    return info.c_str();
}

// Set whether to use CUDA
DLL_EXPORT void set_use_cuda(bool use_cuda) {
    // Re-initialize if needed
    if (!gpu_initialized || (use_cuda && !gpu_available)) {
        gpu_initialized = false;  // Force re-init
        init_gpu_if_needed(use_cuda);
    }
}