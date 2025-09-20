# CRITICAL: GPU Driver Kernel-Level Corruption - Need Distinguished Engineer Help

## Executive Summary
**GPU is permanently corrupted at kernel level. Even `cuCtxCreate()` fails with `CUDA_ERROR_ILLEGAL_ADDRESS`. Hard reset with full driver unload/reload doesn't fix it. System reboot appears to be the only option.**

## What We Implemented From Your Plan

### ✅ Completed Fixes (DLL checksum: f490178b11efc643ee65a5a791d7c7c0)

1. **Explicit Context Path**
```cpp
CUresult res = drv.cuCtxCreate(&ctx, 0, dev);
if (res == CUDA_SUCCESS) {
    using_explicit_context = true;
    // ... use explicit context
} else {
    // fallback to primary
}
```

2. **Serialization**
```cpp
static std::mutex search_mutex;
std::lock_guard<std::mutex> search_lock(search_mutex);
```

3. **Diagnostic Exports**
```cpp
brainstorm_get_last_cuda_error()
brainstorm_is_driver_ready()
brainstorm_gpu_reset()
brainstorm_gpu_hard_reset()  // NEW: Nuclear option
brainstorm_gpu_disable_for_session()
```

4. **Proper Cleanup Based on Context Type**
```cpp
if (using_explicit_context) {
    drv.cuCtxDestroy(ctx);
} else {
    drv.cuDevicePrimaryCtxRelease(dev);
}
```

5. **Hard Reset Implementation**
```cpp
extern "C" __declspec(dllexport) bool brainstorm_gpu_hard_reset() {
    // Clean up everything
    cleanup_gpu_driver();
    
    // Unload the driver completely
    drv.unload();
    
    // Wait for Windows to clean up
    Sleep(2000);
    
    // Try to reload
    if (!drv.load()) {
        // nvcuda.dll won't reload
        g_driver_broken.store(true);
        return false;
    }
    
    // Try to initialize fresh
    bool success = initialize_cuda();
    // ...
}
```

## Current Error Pattern (After All Fixes)

```
[GPU] Starting initialization...
[GPU] CUDA initialized
[GPU] Found 1 CUDA device(s)
[GPU] Got device 0
[GPU] cuCtxCreate failed: CUDA_ERROR_ILLEGAL_ADDRESS (700) - an illegal memory access was encountered
[GPU] CRITICAL: Driver corrupted (ILLEGAL_ADDRESS on context create)
[GPU] Try: brainstorm_gpu_hard_reset() or REBOOT SYSTEM
```

## The Core Problem

### What Should Be Impossible Is Happening
```cpp
// This sequence:
cuInit(0);                     // ✅ SUCCESS
cuDeviceGetCount(&count);      // ✅ SUCCESS (returns 1)
cuDeviceGet(&dev, 0);          // ✅ SUCCESS
cuCtxCreate(&ctx, 0, dev);     // ❌ CUDA_ERROR_ILLEGAL_ADDRESS (700)
```

**How can we get illegal memory access BEFORE allocating any memory?**

### Timeline of Corruption

1. **Initial State**: CUDA worked but had illegal address errors during cuMemcpyHtoD
2. **Added Recovery**: Added cleanup on errors, but cleanup had bug (cuDevicePrimaryCtxRelease(0))
3. **Fixed Cleanup**: Changed to use proper device handle
4. **Switched to Explicit Context**: To bypass primary context corruption
5. **Current State**: Even cuCtxCreate fails with ILLEGAL_ADDRESS

## System Configuration

- **OS**: Windows 11 (WSL2 host)
- **GPU**: NVIDIA RTX 4090
- **Driver**: 561.09
- **CUDA**: Driver API via dynamic loading (cuda_driver_loader.h)
- **Architecture**: x86_64 Windows DLL called from Lua via FFI

## Code Analysis

### Dynamic Driver Loading
```cpp
// cuda_driver_loader.h
struct CudaDrv {
    HMODULE dll = nullptr;
    
    bool load() {
        dll = LoadLibraryA("nvcuda.dll");
        if (!dll) return false;
        
        #define Q(name, str) \
            name = (PFN_##name)GetProcAddress(dll, str); \
            if (!name) return false;
        
        return Q(cuInit, "cuInit") &&
               Q(cuDeviceGetCount, "cuDeviceGetCount") &&
               Q(cuDeviceGet, "cuDeviceGet") &&
               Q(cuCtxCreate, "cuCtxCreate") &&
               // ... etc
    }
    
    void unload() {
        if (dll) {
            FreeLibrary(dll);
            dll = nullptr;
        }
    }
};
```

### Initialization Sequence
```cpp
static bool initialize_cuda() {
    // Load driver
    if (!drv.load()) {
        return false; // Can't load nvcuda.dll
    }
    
    // Initialize CUDA
    if (drv.cuInit(0) != CUDA_SUCCESS) {
        return false;
    }
    
    // Get device
    int device_count = 0;
    if (drv.cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count == 0) {
        return false;
    }
    
    if (drv.cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
        return false;
    }
    
    // Create context - THIS IS WHERE IT FAILS
    CUresult res = drv.cuCtxCreate(&ctx, 0, dev);
    // res = 700 (CUDA_ERROR_ILLEGAL_ADDRESS)
}
```

## Theories

### Theory 1: Kernel Memory Corruption
The Windows kernel GPU driver has corrupted memory structures. The illegal address error occurs because the driver is trying to access kernel memory that's been corrupted or freed.

### Theory 2: TDR (Timeout Detection & Recovery) Side Effects
Windows TDR may have partially reset the GPU but left the driver in an inconsistent state.

### Theory 3: WSL2 Interaction
Running from WSL2 environment might be complicating the GPU passthrough, though the DLL runs on Windows side.

### Theory 4: Driver Version Bug
Driver 561.09 might have a bug with recovery from illegal address errors.

## What We've Tried

1. ✅ Fixed cuDevicePrimaryCtxRelease(0) → cuDevicePrimaryCtxRelease(dev)
2. ✅ Added mutex serialization
3. ✅ Switched to explicit context (cuCtxCreate)
4. ✅ Added proper cleanup with context type tracking
5. ✅ Implemented hard reset with driver unload/reload
6. ❌ All still fail with ILLEGAL_ADDRESS

## Questions for Distinguished Engineer

### 1. How is ILLEGAL_ADDRESS possible on cuCtxCreate?
- No memory operations have occurred yet
- Device handle is valid (cuDeviceGet succeeded)
- This should just be allocating kernel structures

### 2. Why doesn't driver unload/reload fix it?
```cpp
drv.unload();      // FreeLibrary("nvcuda.dll")
Sleep(2000);       // Wait for cleanup
drv.load();        // LoadLibrary("nvcuda.dll")
cuInit(0);         // Still fails at cuCtxCreate
```

### 3. Is there a deeper reset available?
- Is there a way to force Windows to fully reset GPU state?
- Any undocumented CUDA calls for recovery?
- Can we trigger a TDR reset programmatically?

### 4. Could this be a Windows kernel issue?
- The persistence across driver reloads suggests kernel-level corruption
- Only a system reboot fixes it
- Is there a Windows API to reset GPU kernel structures?

## Raw Logs Showing Progression

### After Your Fixes Applied
```
[GPU] Starting initialization...
[GPU] CUDA initialized
[GPU] Found 1 CUDA device(s)
[GPU] Got device 0
[GPU] cuCtxCreate failed: CUDA_ERROR_ILLEGAL_ADDRESS (700) - an illegal memory access was encountered
[GPU] CRITICAL: Driver corrupted (ILLEGAL_ADDRESS on context create)
[GPU] Try: brainstorm_gpu_hard_reset() or REBOOT SYSTEM
```

### After Hard Reset Attempt
```
[GPU] HARD RESET: Attempting full driver unload/reload
[GPU] Driver reloaded, attempting initialization...
[GPU] Starting initialization...
[GPU] CUDA initialized
[GPU] Found 1 CUDA device(s)
[GPU] Got device 0
[GPU] cuCtxCreate failed: CUDA_ERROR_ILLEGAL_ADDRESS (700) - an illegal memory access was encountered
[GPU] HARD RESET FAILED: Still getting errors after reload
[GPU] SYSTEM REBOOT REQUIRED
```

## Current Workaround

We've disabled GPU and return "RETRY" to prevent wrong seeds:
```cpp
if (g_driver_broken.load()) {
    return "";  // Triggers RETRY in DLL
}
```

But users want GPU acceleration working.

## Specific Help Needed

1. **Is there any way to recover without system reboot?**
   - Windows API to reset GPU kernel state?
   - NVIDIA-specific recovery mechanism?
   - Force TDR programmatically?

2. **Why does the corruption persist across driver unload/reload?**
   - Are there kernel objects not being cleaned up?
   - Is there shared memory that survives the unload?

3. **Could we use CUDA Runtime API instead?**
   - Would cudaDeviceReset() help?
   - Different initialization path that might work?

4. **Is this a known issue with driver 561.09?**
   - Any specific bugs with illegal address recovery?
   - Recommended driver version for stability?

## System Impact

- **GPU acceleration completely unavailable**
- **Only system reboot fixes it**
- **Users frustrated with having to reboot**
- **This happens after any ILLEGAL_ADDRESS error**

## Code References

All code in: `/home/krvh/personal/Brainstorm/ImmolateCPP/src/gpu/`
- `gpu_kernel_driver_prod.cpp` - Main driver (lines 174-200 for context creation)
- `cuda_driver_loader.h` - Dynamic loading wrapper
- `brainstorm_driver.cpp` - DLL entry point

## HELP!

This is beyond normal CUDA error recovery. The kernel-level corruption persists even after:
- Releasing all contexts
- Freeing all memory  
- Unloading the module
- Destroying contexts
- Unloading nvcuda.dll
- Waiting 2 seconds
- Reloading everything fresh

Only a Windows reboot clears it. We need expert guidance on kernel-level GPU recovery or this will keep happening every time there's an illegal address error.

Is there ANY way to recover without forcing users to reboot?