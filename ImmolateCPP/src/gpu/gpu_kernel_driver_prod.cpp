/*
 * GPU Kernel Driver - Clean Version
 * Manages CUDA Driver API for GPU-accelerated seed finding
 */

#include "cuda_driver_loader.h"
#include "gpu_types.h"
#include "../seed.hpp"
#include "seed_conversion.hpp"
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <windows.h>  // For Sleep()
#include <unordered_set>

// Embedded PTX kernel code
#include "seed_filter_ptx.h"
#include "pool_types.h"

// Forward declaration for pool manager
extern "C" const void* get_device_pools();

// ============================================================================
// GLOBAL STATE
// ============================================================================

static CudaDrv drv;                      // CUDA driver API functions
static CUmodule mod = nullptr;           // PTX module
static CUfunction fn = nullptr;          // Kernel function  
static CUcontext ctx = nullptr;          // CUDA context
static CUdevice dev = 0;                 // CUDA device (CRITICAL - needed for cleanup!)
static std::atomic<bool> ready(false);   // Initialization state
static std::mutex init_mutex;            // Thread safety

// GPU memory
static CUdeviceptr d_params = 0;         // Filter parameters
static CUdeviceptr d_result = 0;         // Result seed (single mode)
static CUdeviceptr d_found = 0;          // Found flag
static CUdeviceptr d_candidates = 0;     // Candidate buffer (multi mode)
static CUdeviceptr d_cand_count = 0;     // Candidate count
static CUdeviceptr d_debug = 0;          // Debug buffer for probe kernel

// Kernel config (256x256 = 65536 parallel threads)
const uint32_t GRID_SIZE = 256;
const uint32_t BLOCK_SIZE = 256;
const uint32_t CANDIDATE_CAPACITY = 4096; // Max candidates per launch

// Calibration and scheduling
static double g_seeds_per_ms = 0.0;      // Throughput estimate
static uint64_t resume_index = 0;        // Resume point for continued scanning
static uint32_t target_ms = 250;         // Target time per kernel (TDR safety)

// Diagnostic state
static std::atomic<int> g_last_error{0};
static std::atomic<bool> g_driver_ready{false};
static std::atomic<bool> g_driver_broken{false};
static bool using_explicit_context = false;  // Track context type
static std::mutex search_mutex;              // Serialize searches
static int soft_reset_attempts = 0;          // Track reset attempts
static int hard_reset_attempts = 0;

// Allocation registry for safety checks
static std::unordered_set<CUdeviceptr> allocation_registry;
static std::mutex registry_mutex;

// Forward declarations
static bool ensure_context_current();
static void register_allocation(CUdeviceptr ptr);
static void unregister_allocation(CUdeviceptr ptr);
static bool is_valid_device_ptr(CUdeviceptr ptr);
extern "C" __declspec(dllexport) bool brainstorm_probe_args(uint64_t start_seed_index, uint32_t total_seeds, uint32_t chunk_size, uint32_t cap);

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Seed conversion functions are now in seed_conversion.hpp
// Using base-36 (0-9, A-Z) to match Balatro's format

// ============================================================================
// CUDA CONTEXT MANAGEMENT
// ============================================================================

class ScopedContext {
    CudaDrv& drv;
    CUcontext ctx;
    bool pushed = false;
    
public:
    ScopedContext(CudaDrv& d, CUcontext c) : drv(d), ctx(c) {
        if (!ctx) return;
        
        // Make sure our context is current
        CUcontext current = nullptr;
        if (drv.cuCtxGetCurrent) {
            drv.cuCtxGetCurrent(&current);
        }
        
        if (current != ctx) {
            if (drv.cuCtxPushCurrent) {
                pushed = (drv.cuCtxPushCurrent(ctx) == CUDA_SUCCESS);
            }
            if (!pushed && drv.cuCtxSetCurrent) {
                drv.cuCtxSetCurrent(ctx);
            }
        }
    }
    
    ~ScopedContext() {
        if (pushed && drv.cuCtxPopCurrent) {
            CUcontext dummy;
            drv.cuCtxPopCurrent(&dummy);
        }
    }
};

// ============================================================================
// INITIALIZATION
// ============================================================================

static bool initialize_cuda() {
    std::lock_guard<std::mutex> lock(init_mutex);
    
    // Already initialized? Check all required resources
    if (ready.load() && ctx && mod && fn && d_params && d_result && d_found) {
        return true;
    }
    
    // Check if driver is permanently broken
    if (g_driver_broken.load()) {
        return false;
    }
    
    // Log initialization attempt
    FILE* log = fopen("C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm\\gpu_driver.log", "a");
    if (log) {
        fprintf(log, "[GPU] Starting initialization...\n");
        fflush(log);
    }
    
    // Reset state
    ready.store(false);
    ctx = nullptr;
    mod = nullptr;
    fn = nullptr;
    d_params = 0;
    d_result = 0;
    d_found = 0;
    
    // Load CUDA driver
    if (!drv.load()) {
        if (log) {
            fprintf(log, "[GPU] ERROR: Failed to load nvcuda.dll\n");
            fclose(log);
        }
        return false; // No NVIDIA driver
    }
    
    // Initialize CUDA
    if (drv.cuInit(0) != CUDA_SUCCESS) {
        if (log) {
            fprintf(log, "[GPU] ERROR: cuInit failed\n");
            fclose(log);
        }
        return false;
    }
    if (log) fprintf(log, "[GPU] CUDA initialized\n");
    
    // Get GPU device
    int device_count = 0;
    if (drv.cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count == 0) {
        if (log) {
            fprintf(log, "[GPU] ERROR: No CUDA devices found\n");
            fclose(log);
        }
        return false; // No CUDA devices
    }
    if (log) fprintf(log, "[GPU] Found %d CUDA device(s)\n", device_count);
    
    // Use global device variable
    if (drv.cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
        if (log) {
            fprintf(log, "[GPU] ERROR: cuDeviceGet failed\n");
            fclose(log);
        }
        return false;
    }
    if (log) fprintf(log, "[GPU] Got device 0\n");
    
    // Try explicit context first (more reliable)
    CUresult res = drv.cuCtxCreate(&ctx, 0, dev);
    if (res == CUDA_SUCCESS) {
        using_explicit_context = true;
        if (log) fprintf(log, "[GPU] Created explicit context successfully\n");
    } else {
        // Log the specific error for cuCtxCreate
        const char* err_name = nullptr;
        const char* err_str = nullptr;
        drv.cuGetErrorName(res, &err_name);
        drv.cuGetErrorString(res, &err_str);
        if (log) fprintf(log, "[GPU] cuCtxCreate failed: %s (%d) - %s\n", 
                err_name ? err_name : "UNKNOWN", res, err_str ? err_str : "No description");
        
        // If we get ILLEGAL_ADDRESS even on cuCtxCreate, the driver is corrupted
        if (res == 700) { // CUDA_ERROR_ILLEGAL_ADDRESS
            if (log) {
                fprintf(log, "[GPU] CRITICAL: Driver corrupted (ILLEGAL_ADDRESS on context create)\n");
                fprintf(log, "[GPU] Try: brainstorm_gpu_hard_reset() or REBOOT SYSTEM\n");
                fclose(log);
            }
            g_driver_broken.store(true);
            return false;
        }
        
        // Fallback: try primary context (without reset since function not available)
        if (log) fprintf(log, "[GPU] Trying primary context as fallback...\n", res);
        
        // Note: cuDevicePrimaryCtxReset not available in our loader
        // Try to release first if any lingering reference
        drv.cuDevicePrimaryCtxRelease(dev); // Ignore errors
        
        res = drv.cuDevicePrimaryCtxRetain(&ctx, dev);
        if (res != CUDA_SUCCESS) {
            const char* err_name = nullptr;
            const char* err_str = nullptr;
            drv.cuGetErrorName(res, &err_name);
            drv.cuGetErrorString(res, &err_str);
            if (log) {
                fprintf(log, "[GPU] ERROR: cuDevicePrimaryCtxRetain failed: %s (%d) - %s\n", 
                        err_name ? err_name : "UNKNOWN", res, err_str ? err_str : "No description");
                fclose(log);
            }
            g_last_error.store(res);
            return false;
        }
        using_explicit_context = false;
    }
    if (log) fprintf(log, "[GPU] Context retained\n");
    
    if (drv.cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
        if (log) {
            fprintf(log, "[GPU] ERROR: cuCtxSetCurrent failed\n");
            fclose(log);
        }
        drv.cuDevicePrimaryCtxRelease(dev);
        return false;
    }
    if (log) fprintf(log, "[GPU] Context set as current\n");
    
    // Load PTX module
    if (drv.cuModuleLoadDataEx(&mod, seed_filter_kernel_ptx, 0, nullptr, nullptr) != CUDA_SUCCESS) {
        if (log) {
            fprintf(log, "[GPU] ERROR: Failed to load PTX module\n");
            fclose(log);
        }
        // CRITICAL: Must destroy context on failure to prevent leak
        if (using_explicit_context) {
            drv.cuCtxDestroy(ctx);
        } else {
            drv.cuDevicePrimaryCtxRelease(dev);
        }
        ctx = nullptr;
        return false;
    }
    if (log) fprintf(log, "[GPU] PTX module loaded\n");
    
    // Get kernel function
    if (drv.cuModuleGetFunction(&fn, mod, "find_seeds_kernel_balatro") != CUDA_SUCCESS) {
        // Try legacy name
        if (drv.cuModuleGetFunction(&fn, mod, "find_seeds_kernel") != CUDA_SUCCESS) {
            if (log) {
                fprintf(log, "[GPU] ERROR: Kernel function not found\n");
                fclose(log);
            }
            drv.cuModuleUnload(mod);
            drv.cuDevicePrimaryCtxRelease(dev);
            return false;
        }
    }
    if (log) fprintf(log, "[GPU] Kernel function found\n");
    
    // Ensure context is current
    if (!ensure_context_current()) {
        if (log) {
            fprintf(log, "[GPU] ERROR: Failed to set context current\n");
            fclose(log);
        }
        if (using_explicit_context) {
            drv.cuCtxDestroy(ctx);
        } else {
            drv.cuDevicePrimaryCtxRelease(dev);
        }
        return false;
    }
    
    // Allocate GPU memory (including candidate buffer and debug buffer)
    if (drv.cuMemAlloc(&d_params, sizeof(FilterParams)) != CUDA_SUCCESS ||
        drv.cuMemAlloc(&d_result, sizeof(uint64_t)) != CUDA_SUCCESS ||
        drv.cuMemAlloc(&d_found, sizeof(int)) != CUDA_SUCCESS ||
        drv.cuMemAlloc(&d_candidates, CANDIDATE_CAPACITY * sizeof(uint64_t)) != CUDA_SUCCESS ||
        drv.cuMemAlloc(&d_cand_count, sizeof(uint32_t)) != CUDA_SUCCESS ||
        drv.cuMemAlloc(&d_debug, 16 * sizeof(uint64_t)) != CUDA_SUCCESS) {
        
        if (log) {
            fprintf(log, "[GPU] ERROR: Memory allocation failed\n");
            fclose(log);
        }
        // Cleanup on failure
        if (d_params) drv.cuMemFree(d_params);
        if (d_result) drv.cuMemFree(d_result);
        if (d_found) drv.cuMemFree(d_found);
        if (d_candidates) drv.cuMemFree(d_candidates);
        if (d_cand_count) drv.cuMemFree(d_cand_count);
        if (d_debug) drv.cuMemFree(d_debug);
        drv.cuModuleUnload(mod);
        drv.cuDevicePrimaryCtxRelease(dev);
        return false;
    }
    if (log) {
        fprintf(log, "[GPU] Memory allocated successfully\n");
        fprintf(log, "[GPU] Initialization complete!\n");
        fclose(log);
    }
    
    // Register allocations for safety
    register_allocation(d_params);
    register_allocation(d_result);
    register_allocation(d_found);
    register_allocation(d_candidates);
    register_allocation(d_cand_count);
    register_allocation(d_debug);
    
    // Log device info (cuDriverGetVersion not in our loader)
    
    ready.store(true);
    g_driver_ready.store(true);
    g_last_error.store(0);
    soft_reset_attempts = 0;
    hard_reset_attempts = 0;
    return true;
}

// Helper to ensure context is current on this thread
static bool ensure_context_current() {
    if (!ctx) return false;
    
    CUcontext current = nullptr;
    CUresult res = drv.cuCtxGetCurrent(&current);
    if (res != CUDA_SUCCESS) {
        g_last_error.store(res);
        return false;
    }
    
    if (current != ctx) {
        res = drv.cuCtxSetCurrent(ctx);
        if (res != CUDA_SUCCESS) {
            g_last_error.store(res);
            return false;
        }
        
        // Verify it's set
        res = drv.cuCtxGetCurrent(&current);
        if (res != CUDA_SUCCESS || current != ctx) {
            g_last_error.store(res);
            return false;
        }
    }
    return true;
}

// Check if an error is "sticky" and requires reset
static bool is_sticky_error(CUresult r) {
    return r == 700 ||  // CUDA_ERROR_ILLEGAL_ADDRESS
           r == 201 ||  // CUDA_ERROR_DEINITIALIZED
           r == 709 ||  // CUDA_ERROR_CONTEXT_IS_DESTROYED
           r == 716 ||  // CUDA_ERROR_INVALID_CONTEXT
           r == 719 ||  // CUDA_ERROR_NOT_PERMITTED
           r == 702;    // CUDA_ERROR_LAUNCH_TIMEOUT (TDR)
}

// Register/unregister device allocations
static void register_allocation(CUdeviceptr ptr) {
    if (ptr) {
        std::lock_guard<std::mutex> lock(registry_mutex);
        allocation_registry.insert(ptr);
    }
}

static void unregister_allocation(CUdeviceptr ptr) {
    if (ptr) {
        std::lock_guard<std::mutex> lock(registry_mutex);
        allocation_registry.erase(ptr);
    }
}

static bool is_valid_device_ptr(CUdeviceptr ptr) {
    if (!ptr) return false;
    std::lock_guard<std::mutex> lock(registry_mutex);
    return allocation_registry.count(ptr) > 0;
}

// ============================================================================
// SYNTHETIC POOLS FOR CALIBRATION
// ============================================================================

// Build a minimal DevicePools blob in host memory and upload to device.
// Returns true on success; sets out_dev_ptr to the device allocation.
static bool build_and_upload_synthetic_pools(CudaDrv& drv, CUdeviceptr& out_dev_ptr) {
    // Minimal context keys; exact strings don't matter for calibration, but lengths do.
    const char* keys[5] = {"Voucher", "PackSlot1", "PackSlot2", "Tag_small", "Tag_big"};
    
    // Build header
    DevicePools header{};
    uint32_t prefix_offs[5]{};
    uint32_t prefix_lens[5]{};
    std::vector<uint8_t> prefixes;
    prefixes.reserve(64);

    for (int i = 0; i < 5; ++i) {
        prefix_offs[i] = (uint32_t)prefixes.size();
        size_t len = strlen(keys[i]);
        prefix_lens[i] = (uint32_t)len;
        prefixes.insert(prefixes.end(), (const uint8_t*)keys[i], (const uint8_t*)keys[i] + len);

        header.ctx[i].prefix_off = prefix_offs[i];
        header.ctx[i].prefix_len = prefix_lens[i];
        header.ctx[i].pool_off   = 0;     // uniform
        header.ctx[i].pool_len   = 1;     // single item
        header.ctx[i].weighted   = 0;     // uniform
    }
    
    // 8-byte pad prefixes
    while (prefixes.size() % 8 != 0) prefixes.push_back(0);
    header.prefixes_size = (uint32_t)prefixes.size();
    header.weights_count = 0;
    header.reserved[0] = 1;   // version
    header.reserved[1] = 0;

    // Assemble final host buffer
    size_t total = sizeof(DevicePools) + prefixes.size();
    std::vector<uint8_t> buf;
    buf.resize(total);
    memcpy(buf.data(), &header, sizeof(DevicePools));
    memcpy(buf.data() + sizeof(DevicePools), prefixes.data(), prefixes.size());

    // Upload to device
    CUdeviceptr d_mem = 0;
    if (drv.cuMemAlloc(&d_mem, total) != CUDA_SUCCESS) return false;
    if (drv.cuMemcpyHtoD(d_mem, buf.data(), total) != CUDA_SUCCESS) {
        drv.cuMemFree(d_mem);
        return false;
    }
    out_dev_ptr = d_mem;
    return true;
}

// ============================================================================
// CALIBRATION
// ============================================================================

static bool calibrate_throughput() {
    if (!ready.load() || !ctx || !fn) return false;
    
    ScopedContext scope(drv, ctx);
    
    FILE* log = fopen("C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm\\gpu_driver.log", "a");
    if (log) fprintf(log, "[GPU] Starting throughput calibration...\n");
    
    // Use a small fixed scan for calibration (2 million seeds)
    uint64_t calibration_seeds = 2000000;
    uint32_t chunk_size = 1024;
    
    // Setup test parameters (no filters - just measure raw throughput)
    FilterParams test_params;
    test_params.tag1_small = 0xFFFFFFFF;
    test_params.tag1_big = 0xFFFFFFFF;
    test_params.tag2_small = 0xFFFFFFFF;
    test_params.tag2_big = 0xFFFFFFFF;
    test_params.voucher = 0xFFFFFFFF;
    test_params.pack1 = 0xFFFFFFFF;
    test_params.pack2 = 0xFFFFFFFF;
    test_params.require_souls = 0;
    test_params.require_observatory = 0;
    test_params.require_perkeo = 0;
    
    // Upload parameters
    drv.cuMemcpyHtoD(d_params, &test_params, sizeof(FilterParams));
    
    // Clear found flag and candidate count
    int zero_flag = 0;
    uint32_t zero_count = 0;
    drv.cuMemcpyHtoD(d_found, &zero_flag, sizeof(int));
    drv.cuMemcpyHtoD(d_cand_count, &zero_count, sizeof(uint32_t));
    
    // CRITICAL FIX: get_device_pools() returns HOST memory, not device!
    // Always pass nullptr for pools to avoid ILLEGAL_ADDRESS
    CUdeviceptr pools_for_calibration = 0;  // nullptr
    if (log) fprintf(log, "[GPU] Using null pools for calibration (pools disabled)\n");
    
    // Launch kernel with timing
    uint64_t start_index = 0;
    uint32_t cap = CANDIDATE_CAPACITY;
    void* args[] = {
        &start_index,
        &calibration_seeds,
        &chunk_size,
        &d_params,
        &pools_for_calibration,
        &d_candidates,
        &cap,
        &d_cand_count,
        &d_found
    };
    
    // Record start time using wall clock
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Launch kernel
    CUresult launch_result = drv.cuLaunchKernel(
        fn,
        GRID_SIZE, 1, 1,    // Grid dimensions
        BLOCK_SIZE, 1, 1,   // Block dimensions
        0,                  // Shared memory
        0,                  // Stream
        args,
        nullptr
    );
    
    if (launch_result != CUDA_SUCCESS) {
        if (log) {
            fprintf(log, "[GPU] ERROR: Calibration kernel launch failed\n");
            fclose(log);
        }
        return false;
    }
    
    // Wait for completion
    drv.cuCtxSynchronize();
    
    // Calculate elapsed time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    // Calculate throughput
    if (elapsed_ms > 0) {
        g_seeds_per_ms = (double)calibration_seeds / (double)elapsed_ms;
        if (log) {
            fprintf(log, "[GPU] Calibration complete: %.2f seeds/ms (%.2f M seeds/sec)\n",
                    g_seeds_per_ms, g_seeds_per_ms / 1000.0);
            fprintf(log, "[GPU] Time for %llu seeds: %lld ms\n", calibration_seeds, elapsed_ms);
        }
    } else {
        g_seeds_per_ms = 10000.0; // Fallback estimate
        if (log) fprintf(log, "[GPU] WARNING: Calibration timing failed, using fallback\n");
    }
    
    // No cleanup needed since we're using nullptr
    
    if (log) fclose(log);
    return true;
}

// ============================================================================
// CPU FALLBACK
// ============================================================================

// Forward declaration of CPU fallback
extern "C" std::string cpu_search_balatro(
    const std::string& start_seed_str,
    const FilterParams& params
);

// Forward declarations
static void cleanup_cuda();

// ============================================================================
// CUDA ERROR CODES
// ============================================================================

#ifndef CUDA_ERROR_ILLEGAL_ADDRESS
#define CUDA_ERROR_ILLEGAL_ADDRESS 700
#define CUDA_ERROR_INVALID_CONTEXT 201
#define CUDA_ERROR_CONTEXT_IS_DESTROYED 709
#define CUDA_ERROR_DEINITIALIZED 4
#define CUDA_ERROR_LAUNCH_TIMEOUT 702
#endif

// ============================================================================
// ERROR HANDLING MACRO
// ============================================================================

#define CHECK_CUDA_LOG(call, log) do { \
    CUresult _result = (call); \
    if (_result != CUDA_SUCCESS) { \
        const char* err_name = nullptr; \
        const char* err_str = nullptr; \
        drv.cuGetErrorName(_result, &err_name); \
        drv.cuGetErrorString(_result, &err_str); \
        if (log) { \
            fprintf(log, "[GPU] CUDA ERROR: %s (%d): %s at %s:%d\n", \
                    err_name ? err_name : "?", _result, \
                    err_str ? err_str : "?", __FILE__, __LINE__); \
            fflush(log); \
        } \
        /* Context recovery for fatal errors */ \
        if (_result == CUDA_ERROR_ILLEGAL_ADDRESS || \
            _result == CUDA_ERROR_INVALID_CONTEXT || \
            _result == CUDA_ERROR_CONTEXT_IS_DESTROYED || \
            _result == CUDA_ERROR_DEINITIALIZED || \
            _result == CUDA_ERROR_LAUNCH_TIMEOUT) { \
            if (log) fprintf(log, "[GPU] Fatal error detected, forcing re-initialization\n"); \
            cleanup_cuda(); \
            ready.store(false); \
        } \
        return ""; \
    } \
} while(0)

// ============================================================================
// MAIN SEARCH FUNCTION
// ============================================================================

extern "C" std::string gpu_search_with_driver(
    const std::string& start_seed_str,
    const FilterParams& params
) {
    // Serialize all searches to avoid concurrent CUDA operations
    std::lock_guard<std::mutex> search_lock(search_mutex);
    
    // Check if driver is broken
    if (g_driver_broken.load()) {
        return "";  // Will trigger RETRY in DLL
    }
    
    FILE* log = fopen("C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm\\gpu_driver.log", "a");
    if (log) {
        fprintf(log, "\n[GPU] gpu_search_with_driver called with seed: %s\n", start_seed_str.c_str());
        fprintf(log, "[GPU] Filter params - voucher:%u pack1:%u pack2:%u tag1(s=%u,b=%u) tag2(s=%u,b=%u) souls:%u obs:%u perkeo:%u\n",
                params.voucher, params.pack1, params.pack2, 
                params.tag1_small, params.tag1_big, params.tag2_small, params.tag2_big,
                params.require_souls, params.require_observatory, params.require_perkeo);
        fflush(log);
    }
    
    // Initialize CUDA on first use
    if (!initialize_cuda()) {
        if (log) {
            fprintf(log, "[GPU] ERROR: CUDA initialization failed, disabling GPU\n");
            fprintf(log, "[GPU] Fatal error detected, forcing re-initialization\n");
            fclose(log);
        }
        g_driver_ready.store(false);
        return ""; // Return empty to trigger RETRY, don't use buggy CPU fallback
    }
    
    if (log) fprintf(log, "[GPU] CUDA initialized, setting context\n");
    
    // Ensure context is current on this thread
    CUcontext current_ctx = nullptr;
    drv.cuCtxGetCurrent(&current_ctx);
    if (current_ctx != ctx) {
        if (log) fprintf(log, "[GPU] Context not current, setting...\n");
        CUresult res = drv.cuCtxSetCurrent(ctx);
        if (res != CUDA_SUCCESS) {
            if (log) {
                fprintf(log, "[GPU] ERROR: Failed to set context current (code %d)\n", res);
                fclose(log);
            }
            return cpu_search_balatro(start_seed_str, params);
        }
    }
    
    // Ensure context is current
    ScopedContext scope(drv, ctx);
    
    // Determine start index (use resume if seed is empty)
    uint64_t start_index;
    if (start_seed_str.empty()) {
        // Use resume index
        start_index = resume_index;
        if (log) fprintf(log, "[GPU] Using resume index: %llu\n", start_index);
    } else {
        // Convert seed to numeric and reset resume
        start_index = seed_to_int(start_seed_str.c_str());
        resume_index = start_index;
        if (log) fprintf(log, "[GPU] Seed converted to numeric: %llu\n", start_index);
    }
    
    // Ensure context is current before memory operations
    if (!ensure_context_current()) {
        if (log) {
            fprintf(log, "[GPU] ERROR: Context not current before memory operations\n");
            fclose(log);
        }
        return ""; // Context error
    }
    
    // Validate device pointers are in registry
    if (!is_valid_device_ptr(d_params) || !is_valid_device_ptr(d_found) || 
        !is_valid_device_ptr(d_result) || !is_valid_device_ptr(d_candidates) || 
        !is_valid_device_ptr(d_cand_count)) {
        if (log) {
            fprintf(log, "[GPU] ERROR: Invalid device pointers detected\n");
            fprintf(log, "[GPU] Attempting reinitialization...\n");
            fclose(log);
        }
        // Pointers are invalid, need to reinitialize
        if (!initialize_cuda()) {
            return "";
        }
    }
    
    // Log device pointer values for debugging
    if (log) {
        fprintf(log, "[GPU] Device pointers: d_params=%p d_found=%p d_cand_count=%p d_result=%p d_candidates=%p\n",
                (void*)d_params, (void*)d_found, (void*)d_cand_count, (void*)d_result, (void*)d_candidates);
    }
    
    // Upload filter parameters with proper error handling
    CHECK_CUDA_LOG(drv.cuMemcpyHtoD(d_params, &params, sizeof(FilterParams)), log);
    if (log) fprintf(log, "[GPU] Filter params uploaded to GPU\n");
    
    // Clear found flag and candidate count
    int found_flag = 0;
    uint32_t cand_count = 0;
    CHECK_CUDA_LOG(drv.cuMemcpyHtoD(d_found, &found_flag, sizeof(int)), log);
    CHECK_CUDA_LOG(drv.cuMemcpyHtoD(d_cand_count, &cand_count, sizeof(uint32_t)), log);
    if (log) fprintf(log, "[GPU] Flags cleared\n");
    
    // Calibrate if needed
    if (g_seeds_per_ms <= 0.0) {
        if (log) fprintf(log, "[GPU] No calibration data, running calibration...\n");
        if (!calibrate_throughput()) {
            g_seeds_per_ms = 10000.0; // Fallback: 10M seeds/sec
            if (log) fprintf(log, "[GPU] Calibration failed, using fallback: %.2f seeds/ms\n", g_seeds_per_ms);
        }
    }
    
    // Calculate dynamic seeds_per_launch based on target time
    uint64_t seeds_per_launch = (uint64_t)std::max(1.0, g_seeds_per_ms * target_ms);
    const uint64_t MAX_SEEDS = 1000000000;       // 1B total search space
    const uint32_t MAX_LAUNCHES = 100000;        // Support long-running sessions
    
    if (log) {
        fprintf(log, "[GPU] Using dynamic scheduling:\n");
        fprintf(log, "  - Throughput: %.2f seeds/ms\n", g_seeds_per_ms);
        fprintf(log, "  - Target time: %u ms\n", target_ms);
        fprintf(log, "  - Seeds per launch: %llu\n", seeds_per_launch);
        fprintf(log, "[GPU] Starting search loop (up to %llu seeds)\n", MAX_SEEDS);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // current_start is our working position
    uint64_t current_start = start_index;
    
    // Search loop - launch kernel multiple times
    uint32_t launch_count = 0;
    while (launch_count < MAX_LAUNCHES && !found_flag) {
        // Calculate seeds for this launch
        uint32_t count = (uint32_t)std::min(seeds_per_launch, MAX_SEEDS - (current_start - start_index));
        if (count == 0) break; // Wrapped around or finished
        
        if (log && launch_count == 0) {
            fprintf(log, "[GPU] Launch %u - start:%llu count:%u\n", launch_count, current_start, count);
        }
        
        // Clear candidate count and found flag for this launch (per DE instructions)
        uint32_t zero_count = 0;
        int zero_found = 0;
        drv.cuMemcpyHtoD(d_cand_count, &zero_count, sizeof(uint32_t));
        drv.cuMemcpyHtoD(d_found, &zero_found, sizeof(int));
        
        // Run probe kernel on first REAL launch (skip for calibration)
        static bool probe_run = false;
        static int launch_count = 0;
        launch_count++;
        
        // Force probe to run on EVERY launch for debugging
        if (!probe_run) {
            if (log) {
                fprintf(log, "[GPU] Running probe kernel before first real launch...\n");
                fflush(log);
                fclose(log);
                log = nullptr;
            }
            
            bool probe_ok = brainstorm_probe_args(current_start, count, 1024, CANDIDATE_CAPACITY);
            
            // Reopen log
            log = fopen("C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm\\gpu_driver.log", "a");
            if (log) {
                if (!probe_ok) {
                    fprintf(log, "[GPU] ERROR: Probe kernel failed - argument mismatch detected!\n");
                    fprintf(log, "[GPU] Aborting to prevent corruption\n");
                    fclose(log);
                    return "";
                }
                fprintf(log, "[GPU] Probe kernel passed - continuing with real kernel\n");
            }
            probe_run = true;
        }
        
        // CRITICAL BUG FIX: get_device_pools() returns host pointer when not properly initialized
        // This causes ILLEGAL_ADDRESS when kernel tries to access it
        // For now, always pass nullptr for pools to avoid crash
        CUdeviceptr d_pools_ptr = 0;  // nullptr - don't use pools for now
        
        if (log) {
            fprintf(log, "[GPU] NOTE: Pools disabled to prevent illegal address error\n");
        }
        
        // Launch kernel with candidate buffer support
        uint32_t chunk_size = 1024;  // Process 1024 seeds per thread chunk
        uint32_t cap = CANDIDATE_CAPACITY;
        void* args[] = {
            &current_start,
            &count,
            &chunk_size,
            &d_params,
            &d_pools_ptr,     // Dynamic pools (can be null)
            &d_candidates,    // Candidate buffer
            &cap,             // Capacity
            &d_cand_count,    // Count
            &d_found          // Single-result flag
        };
        
        // Ensure context is current before kernel launch
        if (!ensure_context_current()) {
            if (log) {
                fprintf(log, "[GPU] ERROR: Context not current before kernel launch\n");
                fclose(log);
            }
            return "";
        }
        
        CUresult launch_result = drv.cuLaunchKernel(
            fn,
            GRID_SIZE, 1, 1,    // Grid dimensions
            BLOCK_SIZE, 1, 1,   // Block dimensions
            0,                  // Shared memory
            0,                  // Stream
            args,
            nullptr
        );
        
        if (launch_result != CUDA_SUCCESS) {
            g_last_error.store(launch_result);
            const char* err_name = nullptr;
            const char* err_str = nullptr;
            drv.cuGetErrorName(launch_result, &err_name);
            drv.cuGetErrorString(launch_result, &err_str);
            
            if (log) {
                fprintf(log, "[GPU] ERROR: Kernel launch failed: %s (%d) - %s\n",
                        err_name ? err_name : "UNKNOWN", launch_result, 
                        err_str ? err_str : "No description");
                
                if (is_sticky_error(launch_result)) {
                    fprintf(log, "[GPU] Sticky error detected, GPU disabled for session\n");
                    fprintf(log, "[GPU] To recover, run: dofile(\"Mods/Brainstorm/gpu_recovery.lua\")\n");
                    g_driver_ready.store(false);
                    g_driver_broken.store(true);
                }
                fclose(log);
            }
            return cpu_search_balatro(start_seed_str, params);  // Use CPU fallback
        }
        
        // Wait for kernel completion
        if (drv.cuCtxSynchronize() != CUDA_SUCCESS) {
            if (log) {
                fprintf(log, "[GPU] ERROR: Kernel sync failed\n");
                fclose(log);
            }
            return "";
        }
        
        // Check candidate count
        uint32_t num_candidates = 0;
        drv.cuMemcpyDtoH(&num_candidates, d_cand_count, sizeof(uint32_t));
        
        if (num_candidates > 0) {
            // Copy candidates back
            uint32_t to_copy = std::min(num_candidates, CANDIDATE_CAPACITY);
            std::vector<uint64_t> candidates(to_copy);
            drv.cuMemcpyDtoH(candidates.data(), d_candidates, to_copy * sizeof(uint64_t));
            
            // For now, treat first candidate as the result (single-result mode)
            if (!candidates.empty()) {
                uint64_t found_seed_num = candidates[0];
                char result_str[9];
                int_to_seed(found_seed_num, result_str);
                
                if (log) {
                    fprintf(log, "[GPU] FOUND SEED: %s (numeric: %llu)\n", result_str, found_seed_num);
                    fprintf(log, "[GPU] Found %u candidates total\n", num_candidates);
                    fclose(log);
                }
                
                // Reset resume for next search
                resume_index = 0;
                return std::string(result_str);
            }
        }
        
        // Check single-result flag for backwards compatibility
        drv.cuMemcpyDtoH(&found_flag, d_found, sizeof(int));
        if (found_flag) {
            uint64_t found_seed_num;
            drv.cuMemcpyDtoH(&found_seed_num, d_result, sizeof(uint64_t));
            
            char result_str[9];
            int_to_seed(found_seed_num, result_str);
            
            if (log) {
                fprintf(log, "[GPU] FOUND SEED: %s (numeric: %llu)\n", result_str, found_seed_num);
                fclose(log);
            }
            
            resume_index = 0;
            return std::string(result_str);
        }
        
        // Update resume index for next launch
        current_start += count;
        resume_index = current_start;
        launch_count++;
        
        // Check for overall timeout (avoid infinite loops)
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::high_resolution_clock::now() - start_time
        ).count();
        
        if (elapsed > 30) {  // 30 second overall timeout
            if (log) {
                fprintf(log, "[GPU] Search timeout after %lld seconds\n", elapsed);
                fprintf(log, "[GPU] Resume index saved: %llu\n", resume_index);
                fclose(log);
            }
            break;
        }
    }
    
    if (log) {
        fprintf(log, "[GPU] No match found after searching\n");
        fclose(log);
    }
    
    return ""; // Not found
}

// ============================================================================
// CLEANUP
// ============================================================================

// Forward declare the actual cleanup function
extern "C" void cleanup_gpu_driver();

static void cleanup_cuda() {
    // Internal cleanup function for error recovery
    cleanup_gpu_driver();
}

extern "C" void cleanup_gpu_driver() {
    std::lock_guard<std::mutex> lock(init_mutex);
    
    if (!ready.load()) return;
    
    ready.store(false);
    g_driver_ready.store(false);
    
    // Set context and free resources
    if (ctx && drv.cuCtxSetCurrent) {
        drv.cuCtxSetCurrent(ctx);
        
        // Free device memory
        if (d_found) drv.cuMemFree(d_found);
        if (d_result) drv.cuMemFree(d_result);
        if (d_params) drv.cuMemFree(d_params);
        if (d_candidates) drv.cuMemFree(d_candidates);
        if (d_cand_count) drv.cuMemFree(d_cand_count);
        
        // Unload module
        if (mod) drv.cuModuleUnload(mod);
        
        // Destroy or release context based on type
        if (using_explicit_context) {
            drv.cuCtxDestroy(ctx);
        } else {
            drv.cuDevicePrimaryCtxRelease(dev);
        }
    }
    
    // Reset pointers
    ctx = nullptr;
    mod = nullptr;
    fn = nullptr;
    d_params = 0;
    d_result = 0;
    d_found = 0;
    d_candidates = 0;
    d_cand_count = 0;
    
    drv.unload();
}

// ============================================================================
// DIAGNOSTIC EXPORTS
// ============================================================================

extern "C" __declspec(dllexport) int brainstorm_get_last_cuda_error() {
    return g_last_error.load(std::memory_order_relaxed);
}

extern "C" __declspec(dllexport) bool brainstorm_is_driver_ready() {
    return g_driver_ready.load(std::memory_order_relaxed);
}

// Soft reset - try to recover in-process
static bool soft_reset_internal() {
    std::lock_guard<std::mutex> lock(init_mutex);
    
    FILE* log = fopen("C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm\\gpu_driver.log", "a");
    if (log) {
        fprintf(log, "\n[GPU] SOFT RESET: Attempting context recreation (attempt %d)\n", ++soft_reset_attempts);
        fflush(log);
    }
    
    // Step 1: Set driver not ready
    g_driver_ready.store(false);
    ready.store(false);
    
    // Step 2: Set context current if possible
    if (ctx && drv.cuCtxSetCurrent) {
        drv.cuCtxSetCurrent(ctx);  // Ignore errors
    }
    
    // Step 3: Unload module
    if (mod) {
        CUresult res = drv.cuModuleUnload(mod);
        if (log) fprintf(log, "[GPU] Module unload: %s\n", res == CUDA_SUCCESS ? "SUCCESS" : "FAILED");
        mod = nullptr;
    }
    
    // Step 4: Free device allocations
    if (d_params) { drv.cuMemFree(d_params); unregister_allocation(d_params); d_params = 0; }
    if (d_result) { drv.cuMemFree(d_result); unregister_allocation(d_result); d_result = 0; }
    if (d_found) { drv.cuMemFree(d_found); unregister_allocation(d_found); d_found = 0; }
    if (d_candidates) { drv.cuMemFree(d_candidates); unregister_allocation(d_candidates); d_candidates = 0; }
    if (d_cand_count) { drv.cuMemFree(d_cand_count); unregister_allocation(d_cand_count); d_cand_count = 0; }
    
    // Step 5: Destroy context
    if (ctx) {
        if (using_explicit_context) {
            CUresult res = drv.cuCtxDestroy(ctx);
            if (log) fprintf(log, "[GPU] Explicit context destroy: %s\n", res == CUDA_SUCCESS ? "SUCCESS" : "FAILED");
        } else {
            CUresult res = drv.cuDevicePrimaryCtxRelease(dev);
            if (log) fprintf(log, "[GPU] Primary context release: %s\n", res == CUDA_SUCCESS ? "SUCCESS" : "FAILED");
        }
        ctx = nullptr;
    }
    
    // Step 6: Null all handles
    fn = nullptr;
    
    // Step 7: Try to recreate explicit context first
    CUresult res = drv.cuCtxCreate(&ctx, 0, dev);
    if (res == CUDA_SUCCESS) {
        using_explicit_context = true;
        if (log) fprintf(log, "[GPU] Created new explicit context\n");
        
        res = drv.cuCtxSetCurrent(ctx);
        if (res != CUDA_SUCCESS) {
            if (log) {
                fprintf(log, "[GPU] Failed to set new context current\n");
                fclose(log);
            }
            drv.cuCtxDestroy(ctx);
            ctx = nullptr;
            return false;
        }
    } else {
        // Step 8: Fallback to primary context with reset
        if (log) fprintf(log, "[GPU] Explicit context failed, trying primary...\n");
        
        // Try to reset primary context first
        drv.cuDevicePrimaryCtxRelease(dev);  // Release any lingering reference
        
        res = drv.cuDevicePrimaryCtxRetain(&ctx, dev);
        if (res != CUDA_SUCCESS) {
            if (log) {
                fprintf(log, "[GPU] Primary context retain failed: %d\n", res);
                fclose(log);
            }
            return false;
        }
        
        using_explicit_context = false;
        res = drv.cuCtxSetCurrent(ctx);
        if (res != CUDA_SUCCESS) {
            if (log) {
                fprintf(log, "[GPU] Failed to set primary context current\n");
                fclose(log);
            }
            drv.cuDevicePrimaryCtxRelease(dev);
            ctx = nullptr;
            return false;
        }
    }
    
    // Step 9: Reload module and reallocate
    if (log) {
        fprintf(log, "[GPU] Context recreated, completing initialization...\n");
        fclose(log);
    }
    
    // Complete the initialization (module load, memory alloc)
    return initialize_cuda();
}

extern "C" __declspec(dllexport) bool brainstorm_gpu_reset() {
    return soft_reset_internal();
}

extern "C" __declspec(dllexport) bool brainstorm_gpu_hard_reset() {
    // Nuclear option: full driver unload and reload
    FILE* log = fopen("C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm\\gpu_driver.log", "a");
    if (log) {
        fprintf(log, "\n[GPU] HARD RESET: Attempting full driver unload/reload\n");
        fflush(log);
    }
    
    // Clean up everything
    cleanup_gpu_driver();
    
    // Unload the driver completely
    drv.unload();
    
    // Wait for Windows to clean up
    Sleep(2000);
    
    // Try to reload
    if (!drv.load()) {
        if (log) {
            fprintf(log, "[GPU] HARD RESET FAILED: Cannot reload nvcuda.dll\n");
            fprintf(log, "[GPU] SYSTEM REBOOT REQUIRED\n");
            fclose(log);
        }
        g_driver_broken.store(true);
        return false;
    }
    
    if (log) {
        fprintf(log, "[GPU] Driver reloaded, attempting initialization...\n");
        fclose(log);
    }
    
    // Try to initialize fresh
    bool success = initialize_cuda();
    
    if (!success) {
        g_driver_broken.store(true);
        FILE* log2 = fopen("C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm\\gpu_driver.log", "a");
        if (log2) {
            fprintf(log2, "[GPU] HARD RESET FAILED: Still getting errors after reload\n");
            fprintf(log2, "[GPU] SYSTEM REBOOT REQUIRED\n");
            fclose(log2);
        }
    }
    
    return success;
}

extern "C" __declspec(dllexport) void brainstorm_gpu_disable_for_session() {
    g_driver_ready.store(false);
    g_driver_broken.store(true);
}

// Smoke test - launch a trivial kernel to verify GPU works
extern "C" __declspec(dllexport) bool brainstorm_run_smoke() {
    if (!ready.load() || !ctx || !mod || !fn) {
        return false;
    }
    
    FILE* log = fopen("C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm\\gpu_driver.log", "a");
    if (log) {
        fprintf(log, "[GPU] Running smoke test kernel...\n");
        fflush(log);
    }
    
    // Ensure context is current
    if (!ensure_context_current()) {
        if (log) {
            fprintf(log, "[GPU] Smoke test failed: context not current\n");
            fclose(log);
        }
        return false;
    }
    
    // Set a test value in found flag
    int test_val = 0;
    CUresult res = drv.cuMemcpyHtoD(d_found, &test_val, sizeof(int));
    if (res != CUDA_SUCCESS) {
        if (log) {
            fprintf(log, "[GPU] Smoke test failed: cuMemcpyHtoD error %d\n", res);
            fclose(log);
        }
        return false;
    }
    
    // Launch a minimal kernel (1 block, 1 thread)
    // The kernel will just write 42 to d_found
    FilterParams dummy_params = {};
    void* args[] = {
        &dummy_params,
        &d_found,  // Use d_found as output
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr
    };
    
    res = drv.cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr);
    if (res != CUDA_SUCCESS) {
        if (log) {
            fprintf(log, "[GPU] Smoke test failed: kernel launch error %d\n", res);
            fclose(log);
        }
        g_last_error.store(res);
        return false;
    }
    
    // Synchronize
    res = drv.cuCtxSynchronize();
    if (res != CUDA_SUCCESS) {
        if (log) {
            fprintf(log, "[GPU] Smoke test failed: synchronize error %d\n", res);
            fclose(log);
        }
        g_last_error.store(res);
        return false;
    }
    
    // Read back result
    int result = 0;
    res = drv.cuMemcpyDtoH(&result, d_found, sizeof(int));
    if (res != CUDA_SUCCESS) {
        if (log) {
            fprintf(log, "[GPU] Smoke test failed: cuMemcpyDtoH error %d\n", res);
            fclose(log);
        }
        return false;
    }
    
    // The kernel should have written something non-zero
    bool success = (result != 0);
    
    if (log) {
        fprintf(log, "[GPU] Smoke test %s (result=%d)\n", success ? "PASSED" : "FAILED", result);
        fclose(log);
    }
    
    return success;
}

// Probe kernel - verify argument passing per Distinguished Engineer's instructions
extern "C" __declspec(dllexport) bool brainstorm_probe_args(
    uint64_t start_seed_index,
    uint32_t total_seeds,
    uint32_t chunk_size,
    uint32_t cap) {
    
    if (!ready.load() || !ctx || !mod) {
        return false;
    }
    
    FILE* log = fopen("C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm\\gpu_driver.log", "a");
    if (log) {
        fprintf(log, "[GPU] Running probe kernel to verify arguments...\n");
        fprintf(log, "[GPU] Host-side arguments:\n");
        fprintf(log, "  start_seed_index: %llu (0x%llX)\n", start_seed_index, start_seed_index);
        fprintf(log, "  total_seeds: %u\n", total_seeds);
        fprintf(log, "  chunk_size: %u\n", chunk_size);
        fprintf(log, "  d_params: 0x%016llX\n", (unsigned long long)d_params);
        fprintf(log, "  d_pools: 0x%016llX (nullptr)\n", 0ULL);
        fprintf(log, "  d_candidates: 0x%016llX\n", (unsigned long long)d_candidates);
        fprintf(log, "  cap: %u\n", cap);
        fprintf(log, "  d_cand_count: 0x%016llX\n", (unsigned long long)d_cand_count);
        fprintf(log, "  d_found: 0x%016llX\n", (unsigned long long)d_found);
        fflush(log);
    }
    
    // Get probe kernel function
    CUfunction probe_fn = nullptr;
    CUresult res = drv.cuModuleGetFunction(&probe_fn, mod, "probe_args_kernel");
    if (res != CUDA_SUCCESS) {
        const char* err_name = nullptr;
        const char* err_str = nullptr;
        drv.cuGetErrorName(res, &err_name);
        drv.cuGetErrorString(res, &err_str);
        if (log) {
            fprintf(log, "[GPU] ERROR: Probe kernel not found: %s (%d) - %s\n",
                    err_name ? err_name : "UNKNOWN", res,
                    err_str ? err_str : "No description");
            
            // Try to list what functions ARE available (for debugging)
            fprintf(log, "[GPU] Attempting to find alternate probe functions...\n");
            CUfunction test_fn = nullptr;
            if (drv.cuModuleGetFunction(&test_fn, mod, "probe_args_struct") == CUDA_SUCCESS) {
                fprintf(log, "[GPU] Found: probe_args_struct\n");
            }
            if (drv.cuModuleGetFunction(&test_fn, mod, "find_seeds_kernel_balatro") == CUDA_SUCCESS) {
                fprintf(log, "[GPU] Found: find_seeds_kernel_balatro (main kernel)\n");
            }
            
            fclose(log);
        }
        return false;
    }
    
    // Ensure context is current
    if (!ensure_context_current()) {
        if (log) {
            fprintf(log, "[GPU] ERROR: Context not current for probe\n");
            fclose(log);
        }
        return false;
    }
    
    // Clear debug buffer
    uint64_t zero_buf[16] = {0};
    res = drv.cuMemcpyHtoD(d_debug, zero_buf, sizeof(zero_buf));
    if (res != CUDA_SUCCESS) {
        if (log) {
            fprintf(log, "[GPU] ERROR: Failed to clear debug buffer: %d\n", res);
            fclose(log);
        }
        return false;
    }
    
    // Zero-initialize cand_count and found (per DE instructions)
    uint32_t zero_count = 0;
    int zero_found = 0;
    drv.cuMemcpyHtoD(d_cand_count, &zero_count, sizeof(uint32_t));
    drv.cuMemcpyHtoD(d_found, &zero_found, sizeof(int));
    
    // Setup arguments for probe kernel (same as real kernel + debug buffer)
    CUdeviceptr d_pools_ptr = 0;  // nullptr
    void* args[] = {
        &start_seed_index,
        &total_seeds,
        &chunk_size,
        &d_params,
        &d_pools_ptr,     // nullptr
        &d_candidates,
        &cap,
        &d_cand_count,
        &d_found,
        &d_debug          // Debug buffer (extra param)
    };
    
    // Launch probe kernel (1 block, 1 thread)
    res = drv.cuLaunchKernel(probe_fn, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr);
    if (res != CUDA_SUCCESS) {
        const char* err_name = nullptr;
        const char* err_str = nullptr;
        drv.cuGetErrorName(res, &err_name);
        drv.cuGetErrorString(res, &err_str);
        if (log) {
            fprintf(log, "[GPU] ERROR: Probe kernel launch failed: %s (%d) - %s\n",
                    err_name ? err_name : "UNKNOWN", res, 
                    err_str ? err_str : "No description");
            fclose(log);
        }
        g_last_error.store(res);
        return false;
    }
    
    // Synchronize
    res = drv.cuCtxSynchronize();
    if (res != CUDA_SUCCESS) {
        if (log) {
            fprintf(log, "[GPU] ERROR: Probe kernel sync failed: %d\n", res);
            fclose(log);
        }
        g_last_error.store(res);
        return false;
    }
    
    // Read back debug buffer
    uint64_t dbg[16] = {0};
    res = drv.cuMemcpyDtoH(dbg, d_debug, sizeof(dbg));
    if (res != CUDA_SUCCESS) {
        if (log) {
            fprintf(log, "[GPU] ERROR: Failed to read debug buffer: %d\n", res);
            fclose(log);
        }
        return false;
    }
    
    // Log device-side view
    if (log) {
        fprintf(log, "[GPU] Device-side arguments (from probe kernel):\n");
        fprintf(log, "  start_seed_index: %llu (0x%llX)\n", dbg[0], dbg[0]);
        fprintf(log, "  total_seeds: %u\n", (uint32_t)dbg[1]);
        fprintf(log, "  chunk_size: %u\n", (uint32_t)dbg[2]);
        fprintf(log, "  d_params: 0x%016llX\n", dbg[3]);
        fprintf(log, "  d_pools: 0x%016llX\n", dbg[4]);
        fprintf(log, "  d_candidates: 0x%016llX\n", dbg[5]);
        fprintf(log, "  cap: %u\n", (uint32_t)dbg[6]);
        fprintf(log, "  d_cand_count: 0x%016llX\n", dbg[7]);
        fprintf(log, "  d_found: 0x%016llX\n", dbg[8]);
        fprintf(log, "  Alignment checks:\n");
        fprintf(log, "    params: %s\n", dbg[9] == 1 ? "OK" : dbg[9] == 0 ? "NULL" : "MISALIGNED");
        fprintf(log, "    candidates: %s\n", dbg[10] == 1 ? "OK" : dbg[10] == 0 ? "NULL" : "MISALIGNED");
        fprintf(log, "    cand_count: %s\n", dbg[11] == 1 ? "OK" : dbg[11] == 0 ? "NULL" : "MISALIGNED");
        fprintf(log, "    found: %s\n", dbg[12] == 1 ? "OK" : dbg[12] == 0 ? "NULL" : "MISALIGNED");
        fprintf(log, "  Completion: 0x%llX (should be 0xC0FFEE)\n", dbg[13]);
        
        // Check for mismatches
        bool match = true;
        if (dbg[0] != start_seed_index) {
            fprintf(log, "  ERROR: start_seed_index mismatch!\n");
            match = false;
        }
        if ((uint32_t)dbg[1] != total_seeds) {
            fprintf(log, "  ERROR: total_seeds mismatch!\n");
            match = false;
        }
        if ((uint32_t)dbg[2] != chunk_size) {
            fprintf(log, "  ERROR: chunk_size mismatch!\n");
            match = false;
        }
        if (dbg[3] != d_params) {
            fprintf(log, "  ERROR: d_params mismatch!\n");
            match = false;
        }
        if (dbg[4] != 0) {
            fprintf(log, "  ERROR: d_pools should be 0 (nullptr)!\n");
            match = false;
        }
        if (dbg[5] != d_candidates) {
            fprintf(log, "  ERROR: d_candidates mismatch!\n");
            match = false;
        }
        if ((uint32_t)dbg[6] != cap) {
            fprintf(log, "  ERROR: cap mismatch!\n");
            match = false;
        }
        if (dbg[7] != d_cand_count) {
            fprintf(log, "  ERROR: d_cand_count mismatch!\n");
            match = false;
        }
        if (dbg[8] != d_found) {
            fprintf(log, "  ERROR: d_found mismatch!\n");
            match = false;
        }
        if (dbg[13] != 0xC0FFEE) {
            fprintf(log, "  ERROR: Kernel did not complete normally!\n");
            match = false;
        }
        
        fprintf(log, "[GPU] Probe kernel %s\n", match ? "PASSED - All arguments match!" : "FAILED - Arguments mismatch!");
        fclose(log);
        
        return match;
    }
    
    return true;
}

// ============================================================================
// POOL MANAGER SUPPORT FUNCTIONS
// ============================================================================

extern "C" void* get_cuda_context() {
    if (ready.load() && ctx) {
        return ctx;
    }
    return nullptr;
}

extern "C" void* cuda_alloc_device_mem(size_t size) {
    if (!ready.load() || !ctx || !drv.cuMemAlloc) {
        return nullptr;
    }
    
    ScopedContext scope(drv, ctx);
    
    CUdeviceptr ptr = 0;
    CUresult res = drv.cuMemAlloc(&ptr, size);
    
    if (res == CUDA_SUCCESS) {
        return reinterpret_cast<void*>(ptr);
    }
    
    return nullptr;
}

extern "C" void cuda_free_device_mem(void* ptr) {
    if (!ptr || !ready.load() || !ctx || !drv.cuMemFree) {
        return;
    }
    
    ScopedContext scope(drv, ctx);
    
    CUdeviceptr d_ptr = reinterpret_cast<CUdeviceptr>(ptr);
    drv.cuMemFree(d_ptr);
}

extern "C" bool cuda_copy_to_device(void* dst, const void* src, size_t size) {
    if (!dst || !src || size == 0 || !ready.load() || !ctx || !drv.cuMemcpyHtoD) {
        return false;
    }
    
    ScopedContext scope(drv, ctx);
    
    CUdeviceptr d_dst = reinterpret_cast<CUdeviceptr>(dst);
    CUresult res = drv.cuMemcpyHtoD(d_dst, src, size);
    
    return (res == CUDA_SUCCESS);
}

// ============================================================================
// EXPORT FUNCTIONS FOR PHASE 4
// ============================================================================

// Resume control functions
extern "C" __declspec(dllexport)
void brainstorm_set_target_ms(uint32_t ms) {
    if (ms >= 50 && ms <= 1000) {  // Sanity bounds
        target_ms = ms;
        fprintf(stderr, "[FFI] Target time set to %u ms\n", target_ms);
    }
}

extern "C" __declspec(dllexport)
double brainstorm_get_throughput() {
    return g_seeds_per_ms;
}

extern "C" __declspec(dllexport)
void brainstorm_reset_resume(const char* seed) {
    if (seed && seed[0]) {
        // Reset to specific seed
        resume_index = seed_to_int(seed);
        fprintf(stderr, "[FFI] Resume reset to seed %s (index %llu)\n", seed, resume_index);
    } else {
        // Clear resume state
        resume_index = 0;
        fprintf(stderr, "[FFI] Resume state cleared\n");
    }
}

extern "C" __declspec(dllexport)
uint64_t brainstorm_get_resume_index() {
    return resume_index;
}

extern "C" __declspec(dllexport)
void brainstorm_calibrate() {
    g_seeds_per_ms = 0.0;  // Force recalibration
    if (!calibrate_throughput()) {
        g_seeds_per_ms = 10000.0; // Fallback
    }
}