# Brainstorm Mod - Technical Documentation for Engineering Review

## Executive Summary

**Project**: Brainstorm - A high-performance seed filtering mod for Balatro (a roguelike card game)
**Core Challenge**: Achieving GPU acceleration for seed searching via Windows DLL from cross-compiled Linux environment
**Current Status**: GPU initialization succeeds but CUDA context management fails during memory allocation

## Architecture Overview

### System Components
1. **Lua Frontend** (Balatro mod layer) - Handles UI, game hooks, state management
2. **C++ DLL** (Immolate.dll) - Native acceleration for seed searching
3. **CUDA GPU Kernel** - Massively parallel seed evaluation (attempting to implement)
4. **FFI Bridge** - LuaJIT Foreign Function Interface connecting Lua to DLL

### Development Environment
- **Host OS**: WSL2 Ubuntu 24.04.2 LTS
- **Target OS**: Windows 10/11 (Balatro runs on Windows)
- **GPU**: NVIDIA RTX 4090 (Compute Capability 8.9, 24GB VRAM)
- **CUDA**: 12.6.85
- **Compiler**: MinGW-w64 for cross-compilation, NVCC for CUDA
- **Target Runtime**: LuaJIT 2.1 (used by Balatro)

## Current Technical Challenge

### The Core Problem
We're attempting to use CUDA GPU acceleration in a Windows DLL that's:
1. Cross-compiled from Linux (WSL2) using MinGW
2. Loaded by LuaJIT FFI in a Lua game mod
3. Using CUDA Driver API for runtime PTX loading (to avoid CUDA runtime dependencies)

### Specific Issue
The GPU initialization succeeds, but memory allocation fails with "invalid device context" (error 201):

```
[CUDA Driver] Creating new context for device 0
[CUDA Driver] Context created/retained: 0x00000207df97d180
[CUDA Driver] PTX module loaded successfully: 0x00000207e0182400
[CUDA Driver] Kernel function found: 0x00000207de29fc30
[CUDA Driver] Attempting to allocate 28 bytes for params...
[CUDA Driver] cuMemAlloc failed: invalid device context (code 201)
```

### What We've Tried
1. **Primary Context**: Using `cuDevicePrimaryCtxRetain()` - same error
2. **Created Context**: Using `cuCtxCreate()` - same error
3. **Context Synchronization**: Adding `cuCtxSynchronize()` after creation
4. **Context Verification**: Checking current context before/after operations
5. **Driver API Instead of Runtime**: To avoid CUDA runtime dependencies on target

## Critical Source Files

### 1. Main DLL Entry Point
**File: ImmolateCPP/src/brainstorm.cpp**
```cpp
#include "brainstorm.hpp"
#include "dll_exports.hpp"
#include "functions.hpp"
#include "instance.hpp"
#include "items.hpp"
#include "rng.hpp"
#include "util.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#ifdef GPU_ENABLED
#include "gpu/cuda_wrapper.hpp"
#include "gpu/gpu_searcher.hpp"
#endif

// Global filter parameters set by DLL calls
Item BRAINSTORM_VOUCHER = Item::RETRY;
Item BRAINSTORM_PACK = Item::RETRY;
Item BRAINSTORM_TAG1 = Item::RETRY;
Item BRAINSTORM_TAG2 = Item::RETRY;
double BRAINSTORM_SOULS = 0.0;
bool BRAINSTORM_OBSERVATORY = false;
bool BRAINSTORM_PERKEO = false;

// Performance tracking
static uint64_t total_seeds_checked = 0;
static uint64_t session_start_time = 0;
static uint64_t last_report_time = 0;

// Filter function that checks if a seed matches criteria
long filter(Instance inst) {
    if (inst.cache.generatedAnte1Tags) {
        // Check for tags first (cheapest check)
        if (BRAINSTORM_TAG1 != Item::RETRY) {
            Item smallBlindTag = inst.cache.smallBlindTag;
            Item bigBlindTag = inst.cache.bigBlindTag;
            
            if (BRAINSTORM_TAG2 == Item::RETRY) {
                // Single tag mode - must appear on either blind
                if (smallBlindTag != BRAINSTORM_TAG1 && bigBlindTag != BRAINSTORM_TAG1) {
                    return 0;
                }
            } else if (BRAINSTORM_TAG1 == BRAINSTORM_TAG2) {
                // Same tag twice - must appear on BOTH blinds
                if (smallBlindTag != BRAINSTORM_TAG1 || bigBlindTag != BRAINSTORM_TAG1) {
                    return 0;
                }
            } else {
                // Two different tags - both must appear (order doesn't matter)
                bool hasTag1 = (smallBlindTag == BRAINSTORM_TAG1 || bigBlindTag == BRAINSTORM_TAG1);
                bool hasTag2 = (smallBlindTag == BRAINSTORM_TAG2 || bigBlindTag == BRAINSTORM_TAG2);
                if (!hasTag1 || !hasTag2) {
                    return 0;
                }
            }
        }
    }
    
    // Check voucher if specified
    if (BRAINSTORM_VOUCHER != Item::RETRY) {
        inst.initLocks(1, BRAINSTORM_OBSERVATORY, BRAINSTORM_PERKEO);
        Item voucher = inst.nextVoucher(1);
        if (voucher != BRAINSTORM_VOUCHER) {
            return 0;
        }
    }
    
    // Check pack if specified  
    if (BRAINSTORM_PACK != Item::RETRY) {
        inst.cache.generatedFirstPack = true;
        Item pack = inst.nextPack(1);
        if (pack != BRAINSTORM_PACK) {
            return 0;
        }
    }
    
    // Check souls count if specified
    if (BRAINSTORM_SOULS > 0) {
        double soulRate = inst.soulRate();
        if (soulRate < BRAINSTORM_SOULS) {
            return 0;
        }
    }
    
    return 1; // All checks passed
}

// GPU search wrapper
#ifdef GPU_ENABLED
extern "C" std::string gpu_search_with_driver(
    const std::string& start_seed,
    const FilterParams& params
);
#endif

// Main brainstorm function exported to Lua
extern "C" __declspec(dllexport) const char* brainstorm(
    const char* seed,
    const char* voucher,
    const char* pack, 
    const char* tag1,
    const char* tag2,
    double souls,
    int observatory,
    int perkeo
) {
    // Debug logging
    auto now = std::chrono::high_resolution_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
        
    FILE* debug_file = fopen("C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm\\brainstorm_dll.log", "a");
    if (debug_file) {
        fprintf(debug_file, "\n========================================\n");
        fprintf(debug_file, "[DLL] brainstorm() ENTRY at %lld\n", timestamp);
        fprintf(debug_file, "[DLL] Parameters:\n");
        fprintf(debug_file, "  seed=%s\n", seed);
        fprintf(debug_file, "  tag1=%s\n", tag1 ? tag1 : "null");
        fprintf(debug_file, "  tag2=%s\n", tag2 ? tag2 : "null"); 
        fprintf(debug_file, "  souls=%f\n", souls);
        fprintf(debug_file, "  observatory=%d\n", observatory);
        fprintf(debug_file, "  perkeo=%d\n", perkeo);
        fprintf(debug_file, "[DLL] Thread ID: %lu\n", std::this_thread::get_id());
        fprintf(debug_file, "[DLL] Process ID: %d\n", _getpid());
    }
    
    // Parse filters
    BRAINSTORM_VOUCHER = (voucher && strlen(voucher) > 0) ? stringToItem(voucher) : Item::RETRY;
    BRAINSTORM_PACK = (pack && strlen(pack) > 0) ? stringToItem(pack) : Item::RETRY;
    BRAINSTORM_TAG1 = (tag1 && strlen(tag1) > 0) ? stringToItem(tag1) : Item::RETRY;
    BRAINSTORM_TAG2 = (tag2 && strlen(tag2) > 0) ? stringToItem(tag2) : Item::RETRY;
    BRAINSTORM_SOULS = souls;
    BRAINSTORM_OBSERVATORY = observatory;
    BRAINSTORM_PERKEO = perkeo;
    
    // Prepare FilterParams for GPU
    FilterParams params;
    params.tag1 = (BRAINSTORM_TAG1 != Item::RETRY) ? static_cast<uint32_t>(BRAINSTORM_TAG1) : 0xFFFFFFFF;
    params.tag2 = (BRAINSTORM_TAG2 != Item::RETRY) ? static_cast<uint32_t>(BRAINSTORM_TAG2) : 0xFFFFFFFF;
    params.voucher = (BRAINSTORM_VOUCHER != Item::RETRY) ? static_cast<uint32_t>(BRAINSTORM_VOUCHER) : 0xFFFFFFFF;
    params.pack = (BRAINSTORM_PACK != Item::RETRY) ? static_cast<uint32_t>(BRAINSTORM_PACK) : 0xFFFFFFFF;
    params.min_souls = static_cast<float>(souls);
    params.observatory = observatory ? 1 : 0;
    params.perkeo = perkeo ? 1 : 0;
    
    if (debug_file) {
        fprintf(debug_file, "[DLL] FilterParams prepared:\n");
        fprintf(debug_file, "  tag1=%u (0x%X)\n", params.tag1, params.tag1);
        fprintf(debug_file, "  tag2=%u (0x%X)\n", params.tag2, params.tag2);
        fprintf(debug_file, "  voucher=%u\n", params.voucher);
        fprintf(debug_file, "  pack=%u\n", params.pack);
    }
    
#ifdef GPU_ENABLED
    // Try GPU search first
    if (debug_file) {
        fprintf(debug_file, "[DLL] Calling gpu_search_with_driver()...\n");
        fflush(debug_file);
    }
    
    std::string gpu_result = gpu_search_with_driver(seed, params);
    
    if (debug_file) {
        fprintf(debug_file, "[DLL] gpu_search_with_driver() returned\n");
    }
    
    if (!gpu_result.empty()) {
        if (debug_file) {
            fprintf(debug_file, "[DLL] GPU found match: %s\n", gpu_result.c_str());
            fprintf(debug_file, "========================================\n\n");
            fclose(debug_file);
        }
        char* result = (char*)malloc(gpu_result.length() + 1);
        strcpy(result, gpu_result.c_str());
        return result;
    }
    
    if (debug_file) {
        fprintf(debug_file, "[DLL] GPU search failed or no match, falling back to CPU\n");
    }
#endif
    
    // CPU fallback
    std::string result = Search(seed).find(filter);
    
    if (debug_file) {
        if (!result.empty()) {
            fprintf(debug_file, "[DLL] CPU found match: %s\n", result.c_str());
        } else {
            fprintf(debug_file, "[DLL] No match found by CPU\n");
        }
        fprintf(debug_file, "[DLL] Result: %s\n", result.empty() ? "EMPTY (no match or GPU failed)" : result.c_str());
        fprintf(debug_file, "[DLL] Result length: %zu\n", result.length());
        fprintf(debug_file, "========================================\n\n");
        fclose(debug_file);
    }
    
    if (result.empty()) {
        return strdup("");
    }
    
    return strdup(result.c_str());
}

// Other exported functions...
extern "C" __declspec(dllexport) const char* get_tags() {
    return "tag_uncommon|tag_holo|tag_polychrome|tag_negative|tag_foil|tag_investment|tag_voucher|tag_boss|tag_standard|tag_charm|tag_meteor|tag_buffoon|tag_handy|tag_garbage|tag_coupon|tag_double|tag_juggle|tag_runner|tag_ice_cream|tag_free|tag_reroll|tag_rare|tag_economy|tag_override|tag_super_materialism|tag_optimist|tag_lottery|tag_transit|tag_project|tag_aquarium|tag_planet";
}

extern "C" __declspec(dllexport) void free_result(const char* str) {
    if (str) {
        free((void*)str);
    }
}
```

### 2. GPU Driver Implementation (Current Problem Area)
**File: ImmolateCPP/src/gpu/gpu_kernel_driver.cpp**
```cpp
// GPU Kernel Driver Bridge using CUDA Driver API
// This file provides the bridge between the host DLL and the CUDA kernel
// using the Driver API for runtime PTX loading

#include "cuda_driver_loader.h"
#include "gpu_types.h"
#include "../seed.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

// Include the embedded PTX code
#include "seed_filter_ptx.h"

// Global driver API instance
static CudaDrv drv;
static CUmodule mod = nullptr;
static CUfunction fn = nullptr;
static CUcontext ctx = nullptr;
static bool ready = false;

// Device memory pointers
static CUdeviceptr d_params = 0;
static CUdeviceptr d_result = 0;
static CUdeviceptr d_found = 0;
static CUdeviceptr d_debug_stats = 0;

// Initialize CUDA Driver API and load kernel
static bool ensure_module_loaded() {
    if (ready) return true;
    
    // Reset pointers in case of re-initialization
    mod = nullptr;
    fn = nullptr;
    ctx = nullptr;
    d_params = 0;
    d_result = 0;
    d_found = 0;
    d_debug_stats = 0;
    
    // Open debug log
    FILE* debug_file = fopen("C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm\\gpu_driver.log", "a");
    drv.debug_file = debug_file;
    
    if (debug_file) {
        auto now = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        fprintf(debug_file, "\n[%lld] ===== CUDA Driver API Initialization =====\n", ms);
    }
    
    // Load nvcuda.dll
    if (!drv.load()) {
        if (debug_file) {
            fprintf(debug_file, "[CUDA Driver] Failed to load nvcuda.dll or get functions\n");
            fclose(debug_file);
        }
        return false;
    }
    
    // Initialize CUDA
    CUresult res = drv.cuInit(0);
    if (res != CUDA_SUCCESS) {
        if (debug_file) {
            fprintf(debug_file, "[CUDA Driver] cuInit failed: %s\n", drv.getErrorString(res));
            fclose(debug_file);
        }
        return false;
    }
    
    // Get device count
    int device_count = 0;
    res = drv.cuDeviceGetCount(&device_count);
    if (res != CUDA_SUCCESS || device_count == 0) {
        if (debug_file) {
            fprintf(debug_file, "[CUDA Driver] No CUDA devices found\n");
            fclose(debug_file);
        }
        return false;
    }
    
    // Get first device
    CUdevice dev = 0;
    res = drv.cuDeviceGet(&dev, 0);
    if (res != CUDA_SUCCESS) {
        if (debug_file) {
            fprintf(debug_file, "[CUDA Driver] cuDeviceGet failed: %s\n", drv.getErrorString(res));
            fclose(debug_file);
        }
        return false;
    }
    
    // Get device name
    char device_name[256] = {0};
    if (drv.cuDeviceGetName) {
        drv.cuDeviceGetName(device_name, sizeof(device_name), dev);
        if (debug_file) {
            fprintf(debug_file, "[CUDA Driver] Device 0: %s\n", device_name);
        }
    }
    
    // Create a new context instead of using primary context
    // Primary context can have issues with memory allocation in some situations
    if (debug_file) {
        fprintf(debug_file, "[CUDA Driver] Creating new context for device %d\n", dev);
    }
    
    // Create context with default flags (0)
    res = drv.cuCtxCreate(&ctx, 0, dev);
    if (res != CUDA_SUCCESS) {
        if (debug_file) {
            fprintf(debug_file, "[CUDA Driver] cuCtxCreate failed: %s (code %d)\n", drv.getErrorString(res), res);
            fprintf(debug_file, "[CUDA Driver] Falling back to primary context\n");
        }
        
        // Fallback: try primary context
        res = drv.cuDevicePrimaryCtxRetain(&ctx, dev);
        if (res != CUDA_SUCCESS) {
            if (debug_file) {
                fprintf(debug_file, "[CUDA Driver] cuDevicePrimaryCtxRetain also failed: %s (code %d)\n", drv.getErrorString(res), res);
                fclose(debug_file);
            }
            return false;
        }
        
        // Make primary context current
        res = drv.cuCtxSetCurrent(ctx);
        if (res != CUDA_SUCCESS) {
            if (debug_file) {
                fprintf(debug_file, "[CUDA Driver] cuCtxSetCurrent failed: %s (code %d)\n", drv.getErrorString(res), res);
                fclose(debug_file);
            }
            return false;
        }
    }
    
    if (debug_file) {
        fprintf(debug_file, "[CUDA Driver] Context created/retained: %p\n", ctx);
        fprintf(debug_file, "[CUDA Driver] Context is now current\n");
    }
    
    // Synchronize context to ensure it's fully initialized
    res = drv.cuCtxSynchronize();
    if (res != CUDA_SUCCESS) {
        if (debug_file) {
            fprintf(debug_file, "[CUDA Driver] WARNING: cuCtxSynchronize failed: %s (code %d)\n", drv.getErrorString(res), res);
        }
    }
    
    // Load PTX module
    if (debug_file) {
        fprintf(debug_file, "[CUDA Driver] Loading PTX module (%u bytes)\n", build_seed_filter_ptx_len);
        fprintf(debug_file, "[CUDA Driver] PTX data ptr: %p\n", build_seed_filter_ptx);
        fflush(debug_file);
    }
    
    res = drv.cuModuleLoadDataEx(&mod, (const void*)build_seed_filter_ptx, 0, nullptr, nullptr);
    if (res != CUDA_SUCCESS) {
        if (debug_file) {
            fprintf(debug_file, "[CUDA Driver] cuModuleLoadDataEx failed: %s (code %d)\n", drv.getErrorString(res), res);
            fclose(debug_file);
        }
        return false;
    }
    
    if (debug_file) {
        fprintf(debug_file, "[CUDA Driver] PTX module loaded successfully: %p\n", mod);
        fflush(debug_file);
    }
    
    // Get kernel function
    if (debug_file) {
        fprintf(debug_file, "[CUDA Driver] Getting kernel function 'find_seeds_kernel'\n");
        fflush(debug_file);
    }
    res = drv.cuModuleGetFunction(&fn, mod, "find_seeds_kernel");
    if (res != CUDA_SUCCESS) {
        if (debug_file) {
            fprintf(debug_file, "[CUDA Driver] cuModuleGetFunction failed: %s (code %d)\n", drv.getErrorString(res), res);
            fclose(debug_file);
        }
        return false;
    }
    
    // Allocate device memory
    const size_t PARAMS_SIZE = sizeof(FilterParams);
    const size_t RESULT_SIZE = sizeof(uint64_t);
    const size_t FOUND_SIZE = sizeof(int);
    const size_t DEBUG_STATS_SIZE = sizeof(DebugStats);
    
    // Allocate parameters
    res = drv.cuMemAlloc(&d_params, PARAMS_SIZE);
    if (res != CUDA_SUCCESS) {
        if (debug_file) {
            fprintf(debug_file, "[CUDA Driver] Failed to allocate params: %s\n", drv.getErrorString(res));
            fclose(debug_file);
        }
        return false;
    }
    
    // ... more allocations ...
    
    ready = true;
    
    if (debug_file) {
        fprintf(debug_file, "[CUDA Driver] GPU initialization complete\n");
        fprintf(debug_file, "[CUDA Driver] Ready to process seeds\n");
        fclose(debug_file);
    }
    
    return true;
}

// Main GPU search function
extern "C" std::string gpu_search_with_driver(
    const std::string& start_seed_str,
    const FilterParams& params
) {
    FILE* debug_file = fopen("C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm\\gpu_driver.log", "a");
    if (debug_file) {
        fprintf(debug_file, "\n[gpu_search_with_driver] ENTRY\n");
        fprintf(debug_file, "[gpu_search_with_driver] Seed: %s\n", start_seed_str.c_str());
        fprintf(debug_file, "[gpu_search_with_driver] Ready flag: %s\n", ready ? "true" : "false");
        fprintf(debug_file, "[gpu_search_with_driver] Context: %p\n", ctx);
        fprintf(debug_file, "[gpu_search_with_driver] Module: %p\n", mod);
        fprintf(debug_file, "[gpu_search_with_driver] Function: %p\n", fn);
        fflush(debug_file);
    }
    
    // Try to initialize GPU if not ready
    if (!ensure_module_loaded()) {
        // GPU initialization failed - return empty string to indicate no GPU
        if (debug_file) {
            fprintf(debug_file, "[gpu_search_with_driver] ensure_module_loaded() returned false\n");
            fprintf(debug_file, "[gpu_search_with_driver] GPU initialization failed, cannot search\n");
            fclose(debug_file);
        }
        return "";  // Return empty to indicate failure
    }
    
    // ... kernel launch code ...
    
    return "";  // No match found or error
}
```

### 3. CUDA Driver Loader Header
**File: ImmolateCPP/src/gpu/cuda_driver_loader.h**
```cpp
#pragma once
#ifdef _WIN32
#include <windows.h>
#endif
#include <stdint.h>
#include <cstdio>
#include <type_traits>

// CUDA Driver API calling convention on Windows
#ifndef CUDAAPI
#ifdef _WIN32
#define CUDAAPI __stdcall
#else
#define CUDAAPI
#endif
#endif

// Minimal CUDA Driver API types
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUstream_st* CUstream;
typedef unsigned long long CUdeviceptr;

typedef enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_UNKNOWN = 999
} CUresult;

// Function pointer types for CUDA Driver API
typedef CUresult (CUDAAPI *PFN_cuInit)(unsigned int);
typedef CUresult (CUDAAPI *PFN_cuDeviceGet)(CUdevice*, int);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetCount)(int*);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetName)(char*, int, CUdevice);
typedef CUresult (CUDAAPI *PFN_cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
typedef CUresult (CUDAAPI *PFN_cuDevicePrimaryCtxRelease)(CUdevice);
typedef CUresult (CUDAAPI *PFN_cuCtxCreate)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (CUDAAPI *PFN_cuCtxDestroy)(CUcontext);
typedef CUresult (CUDAAPI *PFN_cuCtxSetCurrent)(CUcontext);
typedef CUresult (CUDAAPI *PFN_cuCtxGetCurrent)(CUcontext*);
typedef CUresult (CUDAAPI *PFN_cuCtxPushCurrent)(CUcontext);
typedef CUresult (CUDAAPI *PFN_cuCtxPopCurrent)(CUcontext*);
typedef CUresult (CUDAAPI *PFN_cuModuleLoadDataEx)(CUmodule*, const void*, unsigned int, void*, void*);
typedef CUresult (CUDAAPI *PFN_cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
typedef CUresult (CUDAAPI *PFN_cuMemAlloc)(CUdeviceptr*, size_t);
typedef CUresult (CUDAAPI *PFN_cuMemFree)(CUdeviceptr);
typedef CUresult (CUDAAPI *PFN_cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
typedef CUresult (CUDAAPI *PFN_cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
typedef CUresult (CUDAAPI *PFN_cuMemsetD32)(CUdeviceptr, unsigned int, size_t);
typedef CUresult (CUDAAPI *PFN_cuLaunchKernel)(CUfunction, 
                                               unsigned, unsigned, unsigned,  // grid
                                               unsigned, unsigned, unsigned,  // block
                                               unsigned, CUstream,            // shared mem, stream
                                               void**, void**);               // kernel params, extra
typedef CUresult (CUDAAPI *PFN_cuCtxSynchronize)(void);
typedef CUresult (CUDAAPI *PFN_cuModuleUnload)(CUmodule);
typedef CUresult (CUDAAPI *PFN_cuGetErrorString)(CUresult, const char**);
typedef CUresult (CUDAAPI *PFN_cuGetErrorName)(CUresult, const char**);

// CUDA Driver API loader structure
struct CudaDrv {
#ifdef _WIN32
    HMODULE h = nullptr;
#else
    void* h = nullptr;
#endif
    
    // Function pointers
    PFN_cuInit cuInit = nullptr;
    PFN_cuDeviceGet cuDeviceGet = nullptr;
    PFN_cuDeviceGetCount cuDeviceGetCount = nullptr;
    PFN_cuDeviceGetName cuDeviceGetName = nullptr;
    PFN_cuDevicePrimaryCtxRetain cuDevicePrimaryCtxRetain = nullptr;
    PFN_cuDevicePrimaryCtxRelease cuDevicePrimaryCtxRelease = nullptr;
    PFN_cuCtxCreate cuCtxCreate = nullptr;
    PFN_cuCtxDestroy cuCtxDestroy = nullptr;
    PFN_cuCtxSetCurrent cuCtxSetCurrent = nullptr;
    PFN_cuCtxGetCurrent cuCtxGetCurrent = nullptr;
    PFN_cuCtxPushCurrent cuCtxPushCurrent = nullptr;
    PFN_cuCtxPopCurrent cuCtxPopCurrent = nullptr;
    PFN_cuModuleLoadDataEx cuModuleLoadDataEx = nullptr;
    PFN_cuModuleGetFunction cuModuleGetFunction = nullptr;
    PFN_cuMemAlloc cuMemAlloc = nullptr;
    PFN_cuMemFree cuMemFree = nullptr;
    PFN_cuMemcpyHtoD cuMemcpyHtoD = nullptr;
    PFN_cuMemcpyDtoH cuMemcpyDtoH = nullptr;
    PFN_cuMemsetD32 cuMemsetD32 = nullptr;
    PFN_cuLaunchKernel cuLaunchKernel = nullptr;
    PFN_cuCtxSynchronize cuCtxSynchronize = nullptr;
    PFN_cuModuleUnload cuModuleUnload = nullptr;
    PFN_cuGetErrorString cuGetErrorString = nullptr;
    PFN_cuGetErrorName cuGetErrorName = nullptr;
    
    // Debug logging file
    FILE* debug_file = nullptr;
    
    bool load() {
#ifdef _WIN32
        h = LoadLibraryA("nvcuda.dll");
        if (!h) {
            if (debug_file) fprintf(debug_file, "[CUDA Driver] Failed to load nvcuda.dll\n");
            return false;
        }
        
        auto Q = [&](auto& fp, const char* name) {
            fp = reinterpret_cast<std::remove_reference_t<decltype(fp)>>(GetProcAddress(h, name));
            if (!fp && debug_file) {
                fprintf(debug_file, "[CUDA Driver] Failed to get function: %s\n", name);
            }
            return fp != nullptr;
        };
#else
        // Linux/WSL2 path (for testing)
        return false;
#endif
        
        // Load all required functions
        bool success = Q(cuInit, "cuInit") &&
                      Q(cuDeviceGet, "cuDeviceGet") &&
                      Q(cuDeviceGetCount, "cuDeviceGetCount") &&
                      Q(cuDeviceGetName, "cuDeviceGetName") &&
                      Q(cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain") &&
                      Q(cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease") &&
                      Q(cuCtxCreate, "cuCtxCreate") &&
                      Q(cuCtxDestroy, "cuCtxDestroy") &&
                      Q(cuCtxSetCurrent, "cuCtxSetCurrent") &&
                      Q(cuModuleLoadDataEx, "cuModuleLoadDataEx") &&
                      Q(cuModuleGetFunction, "cuModuleGetFunction") &&
                      Q(cuMemAlloc, "cuMemAlloc") &&
                      Q(cuMemFree, "cuMemFree") &&
                      Q(cuMemcpyHtoD, "cuMemcpyHtoD") &&
                      Q(cuMemcpyDtoH, "cuMemcpyDtoH") &&
                      Q(cuMemsetD32, "cuMemsetD32") &&
                      Q(cuLaunchKernel, "cuLaunchKernel") &&
                      Q(cuCtxSynchronize, "cuCtxSynchronize") &&
                      Q(cuModuleUnload, "cuModuleUnload");
        
        // Optional error functions
        Q(cuGetErrorString, "cuGetErrorString");
        Q(cuGetErrorName, "cuGetErrorName");
        Q(cuCtxGetCurrent, "cuCtxGetCurrent");
        Q(cuCtxPushCurrent, "cuCtxPushCurrent");
        Q(cuCtxPopCurrent, "cuCtxPopCurrent");
        
        if (debug_file) {
            fprintf(debug_file, "[CUDA Driver] Successfully loaded all required functions\n");
        }
        
        return success;
    }
    
    void unload() {
#ifdef _WIN32
        if (h) {
            FreeLibrary(h);
            h = nullptr;
        }
#endif
    }
    
    const char* getErrorString(CUresult res) {
        const char* str = nullptr;
        if (cuGetErrorString && cuGetErrorString(res, &str) == CUDA_SUCCESS && str) {
            return str;
        }
        
        // Fallback for common errors
        switch(res) {
            case CUDA_SUCCESS: return "no error";
            case CUDA_ERROR_INVALID_VALUE: return "invalid value";
            case CUDA_ERROR_OUT_OF_MEMORY: return "out of memory";
            case CUDA_ERROR_NOT_INITIALIZED: return "not initialized";
            case CUDA_ERROR_DEINITIALIZED: return "deinitialized";
            case CUDA_ERROR_NO_DEVICE: return "no device";
            case CUDA_ERROR_INVALID_DEVICE: return "invalid device";
            case CUDA_ERROR_INVALID_CONTEXT: return "invalid device context";
            case CUDA_ERROR_FILE_NOT_FOUND: return "file not found";
            case CUDA_ERROR_NOT_FOUND: return "not found";
            case CUDA_ERROR_NOT_READY: return "not ready";
            case CUDA_ERROR_LAUNCH_FAILED: return "launch failed";
            case CUDA_ERROR_LAUNCH_TIMEOUT: return "launch timeout";
            case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return "launch out of resources";
            default: return "unknown error";
        }
    }
};
```

### 4. CUDA Kernel
**File: ImmolateCPP/src/gpu/seed_filter.cu**
```cuda
// CUDA kernel for GPU-accelerated seed filtering
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// GPU-compatible RNG implementation matching Balatro's algorithm
__device__ uint32_t pseudoseed_device(const char* key, int len) {
    uint32_t hash = 2166136261u;
    for (int i = 0; i < len; i++) {
        hash ^= (uint32_t)key[i];
        hash *= 16777619u;
    }
    return hash;
}

__device__ double pseudorandom_device(uint32_t seed, const char* context) {
    // Combine seed with context string
    int ctx_len = 0;
    while (context[ctx_len]) ctx_len++;
    
    uint32_t combined = seed;
    for (int i = 0; i < ctx_len; i++) {
        combined = combined * 31u + (uint32_t)context[i];
    }
    
    // PCG-like algorithm
    uint64_t state = combined;
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t xorshifted = ((state >> 18u) ^ state) >> 27u;
    uint32_t rot = state >> 59u;
    uint32_t result = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    
    return (double)result / 4294967296.0;
}

// Filter parameters structure (must match host)
struct FilterParams {
    uint32_t tag1;
    uint32_t tag2;
    uint32_t voucher;
    uint32_t pack;
    float min_souls;
    uint32_t observatory;
    uint32_t perkeo;
};

// Debug statistics
struct DebugStats {
    uint64_t seeds_tested;
    uint64_t tag_matches;
    uint64_t voucher_matches;
    uint64_t pack_matches;
    uint64_t souls_matches;
    uint64_t total_matches;
    uint32_t thread_id;
    uint32_t block_id;
};

// Main kernel function
__global__ void find_seeds_kernel(
    uint64_t start_seed,
    uint32_t count,
    const FilterParams* params,
    uint64_t* result,
    volatile int* found,
    DebugStats* debug_stats
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    
    // Early exit if another thread found a match
    if (*found) return;
    
    // Process seeds in strided pattern
    for (uint32_t i = tid; i < count && !(*found); i += stride) {
        uint64_t seed_num = start_seed + i;
        
        // Convert seed number to string (8 chars)
        char seed_str[9];
        for (int j = 7; j >= 0; j--) {
            seed_str[j] = 'A' + (seed_num % 26);
            seed_num /= 26;
        }
        seed_str[8] = '\0';
        
        // Check tags if specified
        if (params->tag1 != 0xFFFFFFFF) {
            // Generate ante 1 tags using RNG
            uint32_t seed_hash = pseudoseed_device(seed_str, 8);
            
            // Small blind tag
            double rng1 = pseudorandom_device(seed_hash, "Tag_Small_1");
            uint32_t small_tag = (uint32_t)(rng1 * 30);  // 30 possible tags
            
            // Big blind tag  
            double rng2 = pseudorandom_device(seed_hash, "Tag_Big_1");
            uint32_t big_tag = (uint32_t)(rng2 * 30);
            
            bool match = false;
            if (params->tag2 == 0xFFFFFFFF) {
                // Single tag - must appear on either blind
                match = (small_tag == params->tag1 || big_tag == params->tag1);
            } else if (params->tag1 == params->tag2) {
                // Same tag twice - must appear on both blinds
                match = (small_tag == params->tag1 && big_tag == params->tag1);
            } else {
                // Two different tags - both must appear
                bool has_tag1 = (small_tag == params->tag1 || big_tag == params->tag1);
                bool has_tag2 = (small_tag == params->tag2 || big_tag == params->tag2);
                match = has_tag1 && has_tag2;
            }
            
            if (!match) continue;
            
            if (debug_stats && atomicAdd(&debug_stats->tag_matches, 1) == 0) {
                debug_stats->thread_id = tid;
            }
        }
        
        // Check voucher if specified
        if (params->voucher != 0xFFFFFFFF) {
            uint32_t seed_hash = pseudoseed_device(seed_str, 8);
            double rng = pseudorandom_device(seed_hash, "Voucher_1");
            uint32_t voucher = (uint32_t)(rng * 32);  // 32 possible vouchers
            
            if (voucher != params->voucher) continue;
            
            if (debug_stats) atomicAdd(&debug_stats->voucher_matches, 1);
        }
        
        // Check pack if specified
        if (params->pack != 0xFFFFFFFF) {
            uint32_t seed_hash = pseudoseed_device(seed_str, 8);
            double rng = pseudorandom_device(seed_hash, "Pack_1");
            uint32_t pack = (uint32_t)(rng * 8);  // 8 possible packs
            
            if (pack != params->pack) continue;
            
            if (debug_stats) atomicAdd(&debug_stats->pack_matches, 1);
        }
        
        // Found a match! Try to claim it
        int old = atomicCAS((int*)found, 0, 1);
        if (old == 0) {
            // We won the race, store the result
            *result = start_seed + i;
            
            if (debug_stats) {
                atomicAdd(&debug_stats->total_matches, 1);
                debug_stats->thread_id = tid;
                debug_stats->block_id = blockIdx.x;
            }
        }
    }
    
    if (debug_stats && tid == 0) {
        atomicAdd(&debug_stats->seeds_tested, count);
    }
}

// Host wrapper function
extern "C" void launch_seed_search(
    uint64_t start_seed,
    uint32_t count,
    void* d_params,
    uint64_t* d_result,
    int* d_found,
    void* d_debug_stats
) {
    // Calculate grid dimensions
    const int threads_per_block = 256;
    const int max_blocks = 65535;
    int blocks = min((count + threads_per_block - 1) / threads_per_block, max_blocks);
    
    // Clear found flag
    cudaMemset(d_found, 0, sizeof(int));
    
    // Launch kernel
    find_seeds_kernel<<<blocks, threads_per_block>>>(
        start_seed, count,
        (FilterParams*)d_params,
        d_result,
        d_found,
        (DebugStats*)d_debug_stats
    );
    
    // Wait for completion
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[CUDA] Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}
```

### 5. Build Script
**File: ImmolateCPP/build_driver.sh**
```bash
#!/bin/bash

echo "======================================="
echo "  Brainstorm Driver API Build v1.0"
echo "======================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for CUDA
CUDA_AVAILABLE=0
NVCC_PATH=""

if [ -f "/usr/local/cuda/bin/nvcc" ]; then
    NVCC_PATH="/usr/local/cuda/bin/nvcc"
    CUDA_VERSION=$($NVCC_PATH --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo -e "${GREEN}✓ CUDA found:${NC} version $CUDA_VERSION at $NVCC_PATH"
    CUDA_AVAILABLE=1
elif [ -f "/usr/local/cuda-12.6/bin/nvcc" ]; then
    NVCC_PATH="/usr/local/cuda-12.6/bin/nvcc"
    CUDA_VERSION=$($NVCC_PATH --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo -e "${GREEN}✓ CUDA found:${NC} version $CUDA_VERSION at $NVCC_PATH"
    CUDA_AVAILABLE=1
else
    echo -e "${RED}✗ CUDA not found${NC}"
    exit 1
fi

# Check for MinGW
if ! command -v x86_64-w64-mingw32-g++ &> /dev/null; then
    echo -e "${RED}✗ MinGW not found. Please install mingw-w64${NC}"
    exit 1
fi

echo -e "${GREEN}✓ MinGW found${NC}"

# Create build directory
mkdir -p build
cd build

echo -e "\n${YELLOW}Building GPU Driver API version...${NC}"

# Step 1: Compile CUDA kernel to PTX
echo "Step 1: Compiling CUDA kernel to PTX..."

$NVCC_PATH \
    -ptx \
    -O3 \
    -arch=sm_70 \
    -o seed_filter.ptx \
    ../src/gpu/seed_filter.cu \
    2>&1 | tee build_ptx.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}✗ PTX compilation failed${NC}"
    exit 1
fi

PTX_SIZE=$(du -h seed_filter.ptx | cut -f1)
echo -e "${GREEN}✓ PTX compiled successfully (${PTX_SIZE})${NC}"

# Step 2: Convert PTX to C header
echo "Step 2: Converting PTX to C header..."

xxd -i seed_filter.ptx > ../src/gpu/seed_filter_ptx.h

echo -e "${GREEN}✓ PTX embedded as C header${NC}"

# Step 3: Compile GPU driver bridge  
echo "Step 3: Compiling GPU driver bridge..."

x86_64-w64-mingw32-g++ \
    -c \
    -O3 \
    -std=c++17 \
    -DGPU_ENABLED \
    -DGPU_DRIVER_API \
    -I ../src/ \
    -o gpu_kernel_driver.o \
    ../src/gpu/gpu_kernel_driver.cpp \
    2>&1 | tee build_driver.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}✗ GPU driver compilation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ GPU driver compiled${NC}"

# Step 4: Compile main brainstorm
echo "Step 4: Compiling main brainstorm..."

x86_64-w64-mingw32-g++ \
    -c \
    -O3 \
    -std=c++17 \
    -DBUILDING_DLL \
    -DGPU_ENABLED \
    -DGPU_DRIVER_API \
    -I ../src/ \
    -o brainstorm.o \
    ../src/brainstorm.cpp \
    2>&1 | tee -a build_driver.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}✗ Brainstorm compilation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Brainstorm wrapper compiled${NC}"

# Step 5: Compile support files
echo "Step 5: Compiling support files..."

x86_64-w64-mingw32-g++ \
    -c -O3 -std=c++17 \
    -o items.o ../src/items.cpp \
    -o rng.o ../src/rng.cpp \
    -o seed.o ../src/seed.cpp \
    -o util.o ../src/util.cpp \
    -o functions.o ../src/functions.cpp

echo -e "${GREEN}✓ Support files compiled${NC}"

# Step 6: Link everything
echo "Step 6: Linking DLL with Driver API..."

x86_64-w64-mingw32-g++ \
    -shared \
    -o ../Immolate.dll \
    brainstorm.o \
    gpu_kernel_driver.o \
    items.o \
    rng.o \
    seed.o \
    util.o \
    functions.o \
    -static-libgcc \
    -static-libstdc++ \
    -Wl,--export-all-symbols \
    2>&1 | tee -a build_driver.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}✓ GPU Driver API build successful${NC}"
    DLL_SIZE=$(du -h ../Immolate.dll | cut -f1)
    echo -e "  DLL size: ${DLL_SIZE}"
else
    echo -e "${RED}✗ Linking failed${NC}"
    exit 1
fi

# Copy PTX to deployment
cp seed_filter.ptx ..

echo -e "\n${GREEN}========================================"
echo -e "  Build Complete!"
echo -e "========================================${NC}"
echo ""
echo "The DLL uses CUDA Driver API with embedded PTX."
echo "It will JIT-compile on the target system."
echo ""
echo "Requirements on target system:"
echo "  - NVIDIA GPU with driver installed"
echo "  - nvcuda.dll (comes with driver)"
echo "  - No CUDA toolkit needed!"
echo ""
echo "Debug logs will be written to:"
echo "  %AppData%\\Roaming\\Balatro\\Mods\\Brainstorm\\gpu_driver.log"
```

## Recent Error Logs

### GPU Initialization Failure Log
```
[1756248637540] ===== CUDA Driver API Initialization =====
[CUDA Driver] Successfully loaded all required functions
[CUDA Driver] Device 0: NVIDIA GeForce RTX 4090
[CUDA Driver] Creating new context for device 0
[CUDA Driver] Context created/retained: 0x00000207df97d180
[CUDA Driver] Context is now current
[CUDA Driver] WARNING: cuCtxSynchronize failed: no error (code 0)
[CUDA Driver] Context set as current successfully
[CUDA Driver] Verified current context: 0x00000207df97d180 (expected: 0x00000207df97d180)
[CUDA Driver] Loading PTX module (120081 bytes)
[CUDA Driver] PTX data ptr: 0x00007ff89d01b040
[CUDA Driver] PTX module loaded successfully: 0x00000207e0182400
[CUDA Driver] Getting kernel function 'find_seeds_kernel'
[CUDA Driver] Kernel function found: 0x00000207de29fc30
[CUDA Driver] === Memory Allocation Phase ===
[CUDA Driver] Pre-alloc context check: current=0x00000207df97d180, expected=0x00000207df97d180, result=no error
[CUDA Driver] Post-SetCurrent verification: current=0x00000207df97d180, expected=0x00000207df97d180, result=no error
[CUDA Driver] Attempting to allocate 28 bytes for params...
[CUDA Driver] cuMemAlloc failed: invalid device context (code 201)
[CUDA Driver] Failed params: ptr=0x00007ff89d06b160, size=28
[CUDA Driver] Context at failure: current=0x00000207df97d180, expected=0x00000207df97d180, result=no error
[CUDA Driver] Attempting cuCtxSynchronize...
[CUDA Driver] cuCtxSynchronize result: no error
[gpu_search_with_driver] ensure_module_loaded() returned false
[gpu_search_with_driver] GPU initialization failed, cannot search
```

## Analysis of the Problem

### What Works
1. **CUDA Driver Loading**: Successfully loads nvcuda.dll and all function pointers
2. **Device Detection**: Correctly identifies RTX 4090
3. **Context Creation**: Context is created successfully (0x00000207df97d180)
4. **PTX Loading**: PTX module loads and compiles successfully
5. **Kernel Function**: Kernel function is found in the module

### What Fails
1. **Memory Allocation**: `cuMemAlloc()` returns error 201 (invalid device context)
2. **Context appears valid**: All context checks show it's current and valid
3. **Synchronization doesn't help**: `cuCtxSynchronize()` succeeds but doesn't fix the issue

### Hypotheses
1. **Cross-compilation issue**: MinGW DLL might have ABI incompatibility with CUDA Driver API
2. **Thread-local storage**: Context might be thread-local and we're losing it somehow
3. **Module loading side effects**: Loading PTX might be corrupting the context state
4. **Windows DLL loading context**: The DLL might be loaded in a way that breaks CUDA context management
5. **LuaJIT FFI interaction**: The FFI bridge might be interfering with CUDA's internal state

## Questions for the Distinguished Engineer

1. **Is using CUDA Driver API from a MinGW-compiled DLL fundamentally flawed?** Should we consider alternatives like:
   - Creating a separate Windows-native helper process?
   - Using OpenCL instead of CUDA?
   - Building a Windows-native DLL on Windows instead of cross-compiling?

2. **Why would a valid, current context fail during memory allocation?** The context verification shows it's current, but cuMemAlloc still fails with "invalid device context".

3. **Are there known issues with CUDA Driver API in DLLs loaded via FFI?** Specifically when the DLL is loaded by LuaJIT's FFI in a game environment.

4. **Would CUDA Runtime API be more stable?** Even though it requires the CUDA runtime on the target system.

5. **Is the PTX JIT compilation approach problematic?** Should we pre-compile to cubin instead?

## Build and Test Instructions

### Building
```bash
# From WSL2 Ubuntu in the project directory
cd ImmolateCPP
./build_driver.sh  # Builds with CUDA Driver API
# or
./build_simple.sh  # CPU-only fallback
```

### Testing
1. Copy `Immolate.dll` and `seed_filter.ptx` to `%AppData%\Roaming\Balatro\Mods\Brainstorm\`
2. Launch Balatro
3. Press Ctrl+A to trigger seed search
4. Check logs at `%AppData%\Roaming\Balatro\Mods\Brainstorm\gpu_driver.log`

### Current Workaround
The mod falls back to CPU search when GPU fails, which works but is 10-100x slower.

## Additional Context

This is a hobby project to learn CUDA programming, but we've hit a fundamental issue that seems to be related to the interaction between:
- Cross-compiled Windows DLLs (MinGW from Linux)
- CUDA Driver API
- LuaJIT FFI
- Game mod environment

Any insights on the architecture or alternative approaches would be greatly appreciated. The core goal is to accelerate seed searching using GPU parallelism, but the current approach might be fundamentally flawed.

Thank you for your expertise and time in reviewing this challenging issue.