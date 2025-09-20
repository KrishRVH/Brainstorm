/*
 * GPU Kernel Driver - Debug-Instrumented Version
 * Complete lifecycle tracing and validation
 */

#include "cuda_driver_loader.h"
#include "gpu_types.h"
#include "../seed.hpp"
#include "../debug.hpp"
#include "seed_conversion.hpp"
#include <chrono>
#include <cstdio>
#include <string>
#include <atomic>
#include <mutex>

// Embedded PTX kernel code
#include "seed_filter_ptx.h"

// ============================================================================
// GLOBAL STATE
// ============================================================================

static CudaDrv drv;                      // CUDA driver API functions
static CUmodule mod = nullptr;           // PTX module
static CUfunction fn = nullptr;          // Kernel function  
static CUcontext ctx = nullptr;          // CUDA context
static std::atomic<bool> ready(false);   // Initialization state
static std::mutex init_mutex;            // Thread safety

// GPU memory
static CUdeviceptr d_params = 0;         // Filter parameters
static CUdeviceptr d_result = 0;         // Result seed
static CUdeviceptr d_found = 0;          // Found flag

// Kernel config (256x256 = 65536 parallel threads)
const uint32_t GRID_SIZE = 256;
const uint32_t BLOCK_SIZE = 256;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Wrapper functions for seed conversion with debug logging
static uint64_t seed_to_int_debug(const char* seed_str) {
    DEBUG_LOG("GPU", "Converting seed '%s' to numeric (base-36)", seed_str);
    
    uint64_t result = 0;
    for (int i = 0; i < 8 && seed_str[i]; i++) {
        char c = seed_str[i];
        uint32_t digit;
        
        if (c >= '0' && c <= '9') {
            digit = c - '0';  // 0-9 maps to 0-9
            DEBUG_LOG("GPU", "  Char[%d]='%c' (digit) -> value=%u, result=%llu", 
                      i, c, digit, result * 36 + digit);
        } else if (c >= 'A' && c <= 'Z') {
            digit = c - 'A' + 10;  // A-Z maps to 10-35
            DEBUG_LOG("GPU", "  Char[%d]='%c' (letter) -> value=%u, result=%llu", 
                      i, c, digit, result * 36 + digit);
        } else {
            DEBUG_LOG("GPU", "  ERROR: Invalid character '%c' at position %d", c, i);
            digit = 0;
        }
        
        result = result * 36 + digit;
    }
    
    DEBUG_LOG("GPU", "Final numeric seed: %llu", result);
    return result;
}

static void int_to_seed_debug(uint64_t seed_num, char* seed_str) {
    DEBUG_LOG("GPU", "Converting numeric %llu to seed string (base-36)", seed_num);
    int_to_seed(seed_num, seed_str);
    DEBUG_LOG("GPU", "Final seed string: '%s'", seed_str);
}

// ============================================================================
// CUDA CONTEXT MANAGEMENT
// ============================================================================

class ScopedContext {
    CudaDrv& drv;
    CUcontext ctx;
    bool pushed = false;
    
public:
    ScopedContext(CudaDrv& d, CUcontext c) : drv(d), ctx(c) {
        DEBUG_LOG("GPU", "ScopedContext: Ensuring context %p is current", ctx);
        
        if (!ctx) {
            DEBUG_LOG("GPU", "ScopedContext: No context provided, skipping");
            return;
        }
        
        // Make sure our context is current
        CUcontext current = nullptr;
        if (drv.cuCtxGetCurrent) {
            CUresult res = drv.cuCtxGetCurrent(&current);
            DEBUG_LOG("GPU", "cuCtxGetCurrent returned %d, current=%p", res, current);
        }
        
        if (current != ctx) {
            DEBUG_LOG("GPU", "Context switch needed: %p -> %p", current, ctx);
            
            if (drv.cuCtxPushCurrent) {
                CUresult res = drv.cuCtxPushCurrent(ctx);
                pushed = (res == CUDA_SUCCESS);
                DEBUG_LOG("GPU", "cuCtxPushCurrent returned %d, pushed=%d", res, pushed);
            }
            
            if (!pushed && drv.cuCtxSetCurrent) {
                CUresult res = drv.cuCtxSetCurrent(ctx);
                DEBUG_LOG("GPU", "cuCtxSetCurrent returned %d", res);
            }
        } else {
            DEBUG_LOG("GPU", "Context already current, no switch needed");
        }
    }
    
    ~ScopedContext() {
        if (pushed && drv.cuCtxPopCurrent) {
            DEBUG_LOG("GPU", "ScopedContext: Popping context");
            CUcontext dummy;
            drv.cuCtxPopCurrent(&dummy);
        }
    }
};

// ============================================================================
// INITIALIZATION WITH FULL INSTRUMENTATION
// ============================================================================

static bool initialize_cuda() {
    DebugTimer timer("GPU", "initialize_cuda");
    std::lock_guard<std::mutex> lock(init_mutex);
    
    DEBUG_LOG("GPU", "=== CUDA INITIALIZATION BEGIN ===");
    
    // Already initialized?
    if (ready.load() && ctx && mod && fn) {
        DEBUG_LOG("GPU", "Already initialized, returning cached state");
        DEBUG_LOG("GPU", "  ready=%d, ctx=%p, mod=%p, fn=%p", 
                  ready.load(), ctx, mod, fn);
        return true;
    }
    
    DEBUG_LOG("GPU", "Resetting state for fresh initialization");
    
    // Reset state
    ready.store(false);
    ctx = nullptr;
    mod = nullptr;
    fn = nullptr;
    d_params = 0;
    d_result = 0;
    d_found = 0;
    
    // Load CUDA driver
    DEBUG_LOG("GPU", "Loading CUDA driver (nvcuda.dll)...");
    if (!drv.load()) {
        DEBUG_LOG("GPU", "ERROR: Failed to load nvcuda.dll - No NVIDIA driver installed?");
        return false;
    }
    DEBUG_LOG("GPU", "CUDA driver loaded successfully");
    DEBUG_LOG("GPU", "  cuInit=%p, cuDeviceGetCount=%p, cuDeviceGet=%p",
              drv.cuInit, drv.cuDeviceGetCount, drv.cuDeviceGet);
    
    // Initialize CUDA
    DEBUG_LOG("GPU", "Calling cuInit(0)...");
    CUresult res = drv.cuInit(0);
    if (res != CUDA_SUCCESS) {
        DEBUG_LOG("GPU", "ERROR: cuInit failed with code %d", res);
        return false;
    }
    DEBUG_LOG("GPU", "CUDA initialized successfully");
    
    // Get GPU device
    DEBUG_LOG("GPU", "Enumerating CUDA devices...");
    int device_count = 0;
    res = drv.cuDeviceGetCount(&device_count);
    if (res != CUDA_SUCCESS) {
        DEBUG_LOG("GPU", "ERROR: cuDeviceGetCount failed with code %d", res);
        return false;
    }
    
    DEBUG_LOG("GPU", "Found %d CUDA device(s)", device_count);
    if (device_count == 0) {
        DEBUG_LOG("GPU", "ERROR: No CUDA devices available");
        return false;
    }
    
    CUdevice dev = 0;
    DEBUG_LOG("GPU", "Getting device 0...");
    res = drv.cuDeviceGet(&dev, 0);
    if (res != CUDA_SUCCESS) {
        DEBUG_LOG("GPU", "ERROR: cuDeviceGet failed with code %d", res);
        return false;
    }
    DEBUG_LOG("GPU", "Got device handle: %d", dev);
    
    // Get device properties for logging
    if (drv.cuDeviceGetName) {
        char name[256] = {0};
        drv.cuDeviceGetName(name, sizeof(name), dev);
        DEBUG_LOG("GPU", "Device name: %s", name);
    } else {
        DEBUG_LOG("GPU", "cuDeviceGetName not available");
    }
    
    // Note: cuDeviceGetAttribute may not be available in our minimal driver
    DEBUG_LOG("GPU", "Device handle acquired, continuing initialization");
    
    // Create context
    DEBUG_LOG("GPU", "Creating primary context...");
    res = drv.cuDevicePrimaryCtxRetain(&ctx, dev);
    if (res != CUDA_SUCCESS) {
        DEBUG_LOG("GPU", "ERROR: cuDevicePrimaryCtxRetain failed with code %d", res);
        return false;
    }
    DEBUG_LOG("GPU", "Primary context retained: %p", ctx);
    
    DEBUG_LOG("GPU", "Setting context as current...");
    res = drv.cuCtxSetCurrent(ctx);
    if (res != CUDA_SUCCESS) {
        DEBUG_LOG("GPU", "ERROR: cuCtxSetCurrent failed with code %d", res);
        drv.cuDevicePrimaryCtxRelease(dev);
        return false;
    }
    DEBUG_LOG("GPU", "Context set as current");
    
    // Load PTX module
    DEBUG_LOG("GPU", "Loading PTX module...");
    DEBUG_LOG("GPU", "PTX data pointer: %p", seed_filter_kernel_ptx);
    if (seed_filter_kernel_ptx) {
        // Log first few bytes of PTX to verify it's valid
        const char* ptx_str = (const char*)seed_filter_kernel_ptx;
        char preview[65] = {0};
        strncpy(preview, ptx_str, 64);
        DEBUG_LOG("GPU", "PTX preview: '%.64s'", preview);
    }
    
    res = drv.cuModuleLoadDataEx(&mod, seed_filter_kernel_ptx, 0, nullptr, nullptr);
    if (res != CUDA_SUCCESS) {
        DEBUG_LOG("GPU", "ERROR: cuModuleLoadDataEx failed with code %d", res);
        DEBUG_LOG("GPU", "Common causes: Invalid PTX, version mismatch, corrupt data");
        drv.cuDevicePrimaryCtxRelease(dev);
        return false;
    }
    DEBUG_LOG("GPU", "PTX module loaded: %p", mod);
    
    // Get kernel function
    DEBUG_LOG("GPU", "Looking for kernel function 'find_seeds_kernel_optimized'...");
    res = drv.cuModuleGetFunction(&fn, mod, "find_seeds_kernel_optimized");
    if (res != CUDA_SUCCESS) {
        DEBUG_LOG("GPU", "Not found (code %d), trying 'find_seeds_kernel'...", res);
        res = drv.cuModuleGetFunction(&fn, mod, "find_seeds_kernel");
        if (res != CUDA_SUCCESS) {
            DEBUG_LOG("GPU", "ERROR: No kernel function found (code %d)", res);
            DEBUG_LOG("GPU", "PTX may not contain expected kernel names");
            drv.cuModuleUnload(mod);
            drv.cuDevicePrimaryCtxRelease(dev);
            return false;
        }
        DEBUG_LOG("GPU", "Found legacy kernel 'find_seeds_kernel'");
    } else {
        DEBUG_LOG("GPU", "Found optimized kernel");
    }
    DEBUG_LOG("GPU", "Kernel function handle: %p", fn);
    
    // Allocate GPU memory
    DEBUG_LOG("GPU", "Allocating GPU memory...");
    
    size_t params_size = sizeof(FilterParams);
    size_t result_size = sizeof(uint64_t);
    size_t found_size = sizeof(int);
    
    DEBUG_LOG("GPU", "  FilterParams: %zu bytes", params_size);
    DEBUG_LOG("GPU", "  Result buffer: %zu bytes", result_size);
    DEBUG_LOG("GPU", "  Found flag: %zu bytes", found_size);
    
    res = drv.cuMemAlloc(&d_params, params_size);
    if (res != CUDA_SUCCESS) {
        DEBUG_LOG("GPU", "ERROR: Failed to allocate params memory (code %d)", res);
        drv.cuModuleUnload(mod);
        drv.cuDevicePrimaryCtxRelease(dev);
        return false;
    }
    DEBUG_LOG("GPU", "  d_params allocated at %llu", d_params);
    
    res = drv.cuMemAlloc(&d_result, result_size);
    if (res != CUDA_SUCCESS) {
        DEBUG_LOG("GPU", "ERROR: Failed to allocate result memory (code %d)", res);
        drv.cuMemFree(d_params);
        drv.cuModuleUnload(mod);
        drv.cuDevicePrimaryCtxRelease(dev);
        return false;
    }
    DEBUG_LOG("GPU", "  d_result allocated at %llu", d_result);
    
    res = drv.cuMemAlloc(&d_found, found_size);
    if (res != CUDA_SUCCESS) {
        DEBUG_LOG("GPU", "ERROR: Failed to allocate found flag memory (code %d)", res);
        drv.cuMemFree(d_params);
        drv.cuMemFree(d_result);
        drv.cuModuleUnload(mod);
        drv.cuDevicePrimaryCtxRelease(dev);
        return false;
    }
    DEBUG_LOG("GPU", "  d_found allocated at %llu", d_found);
    
    DEBUG_LOG("GPU", "All GPU memory allocated successfully");
    
    ready.store(true);
    DEBUG_LOG("GPU", "=== CUDA INITIALIZATION COMPLETE ===");
    DEBUG_LOG("GPU", "System ready for GPU operations");
    
    return true;
}

// ============================================================================
// MAIN SEARCH FUNCTION WITH FULL INSTRUMENTATION
// ============================================================================

extern "C" std::string gpu_search_with_driver(
    const std::string& start_seed_str,
    const FilterParams& params
) {
    DebugTimer timer("GPU", "gpu_search_with_driver");
    
    DEBUG_LOG("GPU", "=== GPU SEARCH BEGIN ===");
    DEBUG_LOG("GPU", "Start seed: '%s'", start_seed_str.c_str());
    
    // Log filter parameters in detail
    DEBUG_LOG("GPU", "Filter parameters:");
    DEBUG_LOG("GPU", "  tag1: %u (%s)", params.tag1,
              params.tag1 == 0xFFFFFFFF ? "no filter" : "active");
    DEBUG_LOG("GPU", "  tag2: %u (%s)", params.tag2,
              params.tag2 == 0xFFFFFFFF ? "no filter" : "active");
    DEBUG_LOG("GPU", "  voucher: %u (%s)", params.voucher,
              params.voucher == 0xFFFFFFFF ? "no filter" : "active");
    DEBUG_LOG("GPU", "  pack: %u (%s)", params.pack,
              params.pack == 0xFFFFFFFF ? "no filter" : "active");
    DEBUG_LOG("GPU", "  require_souls: %u", params.require_souls);
    DEBUG_LOG("GPU", "  require_observatory: %u", params.require_observatory);
    DEBUG_LOG("GPU", "  require_perkeo: %u", params.require_perkeo);
    
    // Hex dump for debugging
    DEBUG_HEX("GPU", "FilterParams raw bytes", &params, sizeof(FilterParams));
    
    // Initialize CUDA on first use
    if (!initialize_cuda()) {
        DEBUG_LOG("GPU", "ERROR: CUDA initialization failed");
        DEBUG_LOG("GPU", "Returning empty string (GPU not available)");
        return "";
    }
    
    // Ensure context is current
    DEBUG_LOG("GPU", "Setting up context scope...");
    ScopedContext scope(drv, ctx);
    
    // Convert seed to numeric (base-36)
    uint64_t start_seed = seed_to_int_debug(start_seed_str.c_str());
    // Max valid seed in base-36 is 36^8 - 1 = 2,821,109,907,455
    DEBUG_ASSERT("GPU", "seed_numeric_valid", 1, start_seed <= 2821109907455ULL ? 1 : 0);
    
    // Upload filter parameters
    DEBUG_LOG("GPU", "Uploading FilterParams to GPU...");
    CUresult res = drv.cuMemcpyHtoD(d_params, &params, sizeof(FilterParams));
    if (res != CUDA_SUCCESS) {
        DEBUG_LOG("GPU", "ERROR: cuMemcpyHtoD(params) failed with code %d", res);
        return "";
    }
    DEBUG_LOG("GPU", "FilterParams uploaded successfully");
    
    // Clear found flag
    DEBUG_LOG("GPU", "Clearing found flag...");
    int found_flag = 0;
    res = drv.cuMemcpyHtoD(d_found, &found_flag, sizeof(int));
    if (res != CUDA_SUCCESS) {
        DEBUG_LOG("GPU", "ERROR: cuMemcpyHtoD(found) failed with code %d", res);
        return "";
    }
    DEBUG_LOG("GPU", "Found flag cleared");
    
    // Search parameters
    const uint32_t SEEDS_PER_LAUNCH = 10000000;  // 10M seeds per kernel
    const uint64_t MAX_SEEDS = 1000000000;       // 1B total search space
    
    DEBUG_LOG("GPU", "Search configuration:");
    DEBUG_LOG("GPU", "  Seeds per launch: %u", SEEDS_PER_LAUNCH);
    DEBUG_LOG("GPU", "  Max search space: %llu", MAX_SEEDS);
    DEBUG_LOG("GPU", "  Grid size: %u x 1 x 1", GRID_SIZE);
    DEBUG_LOG("GPU", "  Block size: %u x 1 x 1", BLOCK_SIZE);
    DEBUG_LOG("GPU", "  Total threads: %u", GRID_SIZE * BLOCK_SIZE);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    uint32_t launches = 0;
    
    // Search loop - launch kernel multiple times
    for (uint64_t offset = 0; offset < MAX_SEEDS; offset += SEEDS_PER_LAUNCH) {
        uint32_t count = (offset + SEEDS_PER_LAUNCH > MAX_SEEDS) 
                        ? (MAX_SEEDS - offset) 
                        : SEEDS_PER_LAUNCH;
        
        uint64_t current_start = start_seed + offset;
        
        if (launches < 3 || (launches % 10) == 0) {  // Log first few and every 10th
            DEBUG_LOG("GPU", "Launch #%u: start=%llu, count=%u", 
                      launches, current_start, count);
        }
        
        // Prepare kernel arguments
        void* args[] = {
            &current_start,
            &count,
            &d_params,
            &d_result,
            &d_found,
            nullptr  // debug_stats (unused)
        };
        
        DEBUG_LOG("GPU", "Kernel arguments:");
        DEBUG_LOG("GPU", "  arg[0] start_seed: %llu", current_start);
        DEBUG_LOG("GPU", "  arg[1] count: %u", count);
        DEBUG_LOG("GPU", "  arg[2] d_params: %llu", d_params);
        DEBUG_LOG("GPU", "  arg[3] d_result: %llu", d_result);
        DEBUG_LOG("GPU", "  arg[4] d_found: %llu", d_found);
        DEBUG_LOG("GPU", "  arg[5] debug_stats: nullptr");
        
        // Launch kernel
        res = drv.cuLaunchKernel(
            fn,
            GRID_SIZE, 1, 1,    // Grid dimensions
            BLOCK_SIZE, 1, 1,   // Block dimensions
            0,                  // Shared memory
            0,                  // Stream
            args,
            nullptr
        );
        
        if (res != CUDA_SUCCESS) {
            DEBUG_LOG("GPU", "ERROR: cuLaunchKernel failed with code %d", res);
            return "";
        }
        
        // Wait for kernel completion
        res = drv.cuCtxSynchronize();
        if (res != CUDA_SUCCESS) {
            DEBUG_LOG("GPU", "ERROR: cuCtxSynchronize failed with code %d", res);
            return "";
        }
        
        // Check if seed was found
        res = drv.cuMemcpyDtoH(&found_flag, d_found, sizeof(int));
        if (res != CUDA_SUCCESS) {
            DEBUG_LOG("GPU", "ERROR: cuMemcpyDtoH(found) failed with code %d", res);
            return "";
        }
        
        DEBUG_LOG("GPU", "Kernel completed, found_flag=%d", found_flag);
        
        if (found_flag) {
            // Get the matching seed
            uint64_t found_seed_num;
            res = drv.cuMemcpyDtoH(&found_seed_num, d_result, sizeof(uint64_t));
            if (res != CUDA_SUCCESS) {
                DEBUG_LOG("GPU", "ERROR: cuMemcpyDtoH(result) failed with code %d", res);
                return "";
            }
            
            DEBUG_LOG("GPU", "Found seed numeric: %llu", found_seed_num);
            
            // Convert to string
            char result_str[9];
            int_to_seed_debug(found_seed_num, result_str);
            
            DEBUG_LOG("GPU", "=== SEED FOUND: %s ===", result_str);
            DEBUG_LOG("GPU", "Total launches: %u", launches + 1);
            
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time
            );
            DEBUG_LOG("GPU", "Search time: %lld ms", elapsed.count());
            
            return std::string(result_str);
        }
        
        launches++;
        
        // Check for timeout (5 seconds to avoid Windows TDR)
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::high_resolution_clock::now() - start_time
        ).count();
        
        if (elapsed > 5) {
            DEBUG_LOG("GPU", "Search timeout after %lld seconds", elapsed);
            DEBUG_LOG("GPU", "Searched %llu seeds across %u launches", 
                      offset + count, launches);
            break;
        }
    }
    
    DEBUG_LOG("GPU", "=== NO MATCH FOUND ===");
    DEBUG_LOG("GPU", "Total launches: %u", launches);
    DEBUG_LOG("GPU", "Seeds searched: ~%llu", (uint64_t)launches * SEEDS_PER_LAUNCH);
    
    return ""; // Not found
}

// ============================================================================
// CLEANUP
// ============================================================================

extern "C" void cleanup_gpu_driver() {
    DebugTimer timer("GPU", "cleanup_gpu_driver");
    std::lock_guard<std::mutex> lock(init_mutex);
    
    DEBUG_LOG("GPU", "=== GPU CLEANUP BEGIN ===");
    
    if (!ready.load()) {
        DEBUG_LOG("GPU", "Not initialized, nothing to clean up");
        return;
    }
    
    ready.store(false);
    
    // Set context and free resources
    if (ctx && drv.cuCtxSetCurrent) {
        DEBUG_LOG("GPU", "Setting context for cleanup...");
        drv.cuCtxSetCurrent(ctx);
        
        if (d_found) {
            DEBUG_LOG("GPU", "Freeing d_found (%llu)...", d_found);
            drv.cuMemFree(d_found);
        }
        if (d_result) {
            DEBUG_LOG("GPU", "Freeing d_result (%llu)...", d_result);
            drv.cuMemFree(d_result);
        }
        if (d_params) {
            DEBUG_LOG("GPU", "Freeing d_params (%llu)...", d_params);
            drv.cuMemFree(d_params);
        }
        
        if (mod) {
            DEBUG_LOG("GPU", "Unloading module (%p)...", mod);
            drv.cuModuleUnload(mod);
        }
        
        DEBUG_LOG("GPU", "Releasing primary context...");
        drv.cuDevicePrimaryCtxRelease(0);
    }
    
    // Reset pointers
    ctx = nullptr;
    mod = nullptr;
    fn = nullptr;
    d_params = 0;
    d_result = 0;
    d_found = 0;
    
    DEBUG_LOG("GPU", "Unloading CUDA driver...");
    drv.unload();
    
    DEBUG_LOG("GPU", "=== GPU CLEANUP COMPLETE ===");
}