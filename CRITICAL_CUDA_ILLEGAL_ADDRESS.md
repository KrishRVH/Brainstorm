# CRITICAL: CUDA_ERROR_ILLEGAL_ADDRESS During Context Creation

## Executive Summary
**CUDA is completely broken**. We're getting `CUDA_ERROR_ILLEGAL_ADDRESS (700)` when calling `cuDevicePrimaryCtxRetain()`, which should be impossible since we haven't even allocated any memory yet. This suggests the GPU driver or device itself is in a corrupted state.

## Current Error Pattern (Infinite Loop)
```
[GPU] Starting initialization...
[GPU] CUDA initialized
[GPU] Found 1 CUDA device(s)
[GPU] Got device 0
[GPU] ERROR: cuDevicePrimaryCtxRetain failed: CUDA_ERROR_ILLEGAL_ADDRESS (700) - an illegal memory access was encountered
[GPU] ERROR: CUDA initialization failed, using CPU fallback
```

## Timeline of Corruption

### Phase 1: Working State (Earlier Today)
- CUDA was working but failing later in the pipeline
- Error occurred at `cuMemcpyHtoD` with device pointers
- GPU was still able to create contexts and load modules

### Phase 2: Added Error Recovery (Made Things Worse)
We added error recovery code with cleanup:
```cpp
#define CHECK_CUDA_LOG(call, log) do { \
    CUresult _result = (call); \
    if (_result != CUDA_SUCCESS) { \
        if (_result == CUDA_ERROR_ILLEGAL_ADDRESS || ...) { \
            cleanup_cuda(); \
            ready.store(false); \
        } \
        return ""; \
    } \
} while(0)
```

### Phase 3: Cleanup Bug Discovered and Fixed
Found that cleanup was using `cuDevicePrimaryCtxRelease(0)` instead of the device handle:
```cpp
// BEFORE (BUG):
drv.cuDevicePrimaryCtxRelease(0);  // Wrong! Passing 0 instead of device

// AFTER (FIXED):
static CUdevice dev = 0;  // Global storage
drv.cuDevicePrimaryCtxRelease(dev);  // Correct
```

### Phase 4: Current State (Still Broken)
Even after fixing the cleanup bug, we're getting `CUDA_ERROR_ILLEGAL_ADDRESS` during context creation.

## Code Analysis

### Current Initialization Code
```cpp
static bool initialize_cuda() {
    std::lock_guard<std::mutex> lock(init_mutex);
    
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
        fprintf(log, "[GPU] ERROR: Failed to load nvcuda.dll\n");
        return false;
    }
    
    // Initialize CUDA
    if (drv.cuInit(0) != CUDA_SUCCESS) {
        fprintf(log, "[GPU] ERROR: cuInit failed\n");
        return false;
    }
    fprintf(log, "[GPU] CUDA initialized\n");
    
    // Get GPU device
    int device_count = 0;
    if (drv.cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count == 0) {
        fprintf(log, "[GPU] ERROR: No CUDA devices found\n");
        return false;
    }
    fprintf(log, "[GPU] Found %d CUDA device(s)\n", device_count);
    
    // Use global device variable
    if (drv.cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
        fprintf(log, "[GPU] ERROR: cuDeviceGet failed\n");
        return false;
    }
    fprintf(log, "[GPU] Got device 0\n");
    
    // Create context with detailed error logging
    CUresult res = drv.cuDevicePrimaryCtxRetain(&ctx, dev);
    if (res != CUDA_SUCCESS) {
        const char* err_name = nullptr;
        const char* err_str = nullptr;
        drv.cuGetErrorName(res, &err_name);
        drv.cuGetErrorString(res, &err_str);
        fprintf(log, "[GPU] ERROR: cuDevicePrimaryCtxRetain failed: %s (%d) - %s\n", 
                err_name ? err_name : "UNKNOWN", res, err_str ? err_str : "No description");
        return false;
    }
    // ... rest of initialization
}
```

## Why Is This Happening?

### Theory 1: Primary Context Corruption
The primary context may be in a permanently corrupted state from the previous illegal memory access. Even though we're calling `cuDevicePrimaryCtxRelease()`, the context might not be fully cleaned up.

### Theory 2: Driver State Corruption
The NVIDIA driver itself might be in a corrupted state. The `CUDA_ERROR_ILLEGAL_ADDRESS` during context creation (before any memory operations) is highly unusual.

### Theory 3: Incomplete Cleanup
Even with our fix to use the correct device handle, there might be lingering state from the previous errors that isn't being cleaned up properly.

### Theory 4: Windows DLL Caching
Windows might be caching the DLL with corrupted state, and our cleanup isn't fully resetting everything.

## Attempted Fixes That Failed

1. ✅ Fixed `cuDevicePrimaryCtxRelease(0)` to use proper device handle
2. ✅ Added detailed error logging to see exact error codes
3. ✅ Added mutex protection for thread safety
4. ❌ Still getting `CUDA_ERROR_ILLEGAL_ADDRESS` on context creation

## Questions for Distinguished Engineer

1. **How can we get `CUDA_ERROR_ILLEGAL_ADDRESS` during `cuDevicePrimaryCtxRetain()`?**
   - This happens BEFORE any memory allocation
   - The device handle is valid (cuDeviceGet succeeds)
   - cuInit(0) succeeds

2. **Is the primary context permanently corrupted?**
   - Should we try `cuDevicePrimaryCtxReset()` instead of just release?
   - Should we use `cuCtxCreate()` instead of primary context?

3. **How to fully reset CUDA state?**
   - Is there a way to force a complete GPU reset?
   - Do we need to unload/reload the driver?
   - Should we try `cuDeviceReset()` (if it exists)?

4. **Could this be a driver bug?**
   - NVIDIA Driver: 561.09
   - CUDA appears to initialize but fails at context creation
   - Is this a known issue with this driver version?

## Potential Solutions to Try

### Solution 1: Use cuDevicePrimaryCtxReset()
```cpp
// Before trying to retain
CUresult reset_res = drv.cuDevicePrimaryCtxReset(dev);
if (reset_res == CUDA_SUCCESS) {
    fprintf(log, "[GPU] Primary context reset successfully\n");
}

// Then try to retain
CUresult res = drv.cuDevicePrimaryCtxRetain(&ctx, dev);
```

### Solution 2: Use cuCtxCreate() Instead
```cpp
// Skip primary context entirely
CUresult res = drv.cuCtxCreate(&ctx, 0, dev);
if (res != CUDA_SUCCESS) {
    // Handle error
}
```

### Solution 3: Complete Driver Reset
```cpp
// Try to completely unload and reload driver
drv.unload();
Sleep(1000);  // Give Windows time to clean up
if (!drv.load()) {
    // Handle error
}
```

### Solution 4: System Reboot (Nuclear Option)
If nothing else works, the user may need to reboot Windows to clear the corrupted GPU state.

## System Information
- Windows 11
- NVIDIA RTX 4090
- CUDA Driver 561.09
- Using cuda_driver_loader.h for dynamic loading

## Raw Error Logs (Last 20 Attempts)
All attempts show the same pattern:
```
[GPU] Starting initialization...
[GPU] CUDA initialized
[GPU] Found 1 CUDA device(s)
[GPU] Got device 0
[GPU] ERROR: cuDevicePrimaryCtxRetain failed: CUDA_ERROR_ILLEGAL_ADDRESS (700) - an illegal memory access was encountered
[GPU] ERROR: CUDA initialization failed, using CPU fallback
```

## Impact
- **GPU acceleration completely unavailable**
- **Falling back to buggy CPU implementation**
- **CPU returns wrong seeds that don't match filters**
- **Users getting incorrect results**

## URGENT: Need Expert Guidance

This is a critical production issue. The GPU acceleration is completely dead and users are getting wrong results. We need expert guidance on:

1. How to recover from `CUDA_ERROR_ILLEGAL_ADDRESS` during context creation
2. Whether to abandon primary context and use `cuCtxCreate()`
3. How to properly reset GPU state after corruption
4. Any known issues with Driver 561.09 and RTX 4090

## Code References

- `/home/krvh/personal/Brainstorm/ImmolateCPP/src/gpu/gpu_kernel_driver_prod.cpp` - Main GPU driver
- Lines 166-173: Where the error occurs
- Lines 727-758: Cleanup function with the fix applied
- `/home/krvh/personal/Brainstorm/ImmolateCPP/src/gpu/cuda_driver_loader.h` - Dynamic CUDA loader

Please help - the entire GPU acceleration feature is dead!