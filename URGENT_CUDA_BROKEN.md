# URGENT: CUDA Completely Broken - cuDevicePrimaryCtxRetain Failing

## Critical Failure
**CUDA is 100% broken. GPU acceleration completely unavailable. Falling back to buggy CPU implementation that returns WRONG SEEDS.**

## Error Pattern (Repeating Infinitely)
```
[GPU] gpu_search_with_driver called with seed: QFFEBVBR
[GPU] Filter params - voucher:1 pack1:2 pack2:2 tag1(s=17,b=17) tag2(s=6,b=6) souls:0 obs:0 perkeo:0
[GPU] Starting initialization...
[GPU] CUDA initialized
[GPU] Found 1 CUDA device(s)
[GPU] Got device 0
[GPU] ERROR: cuDevicePrimaryCtxRetain failed
[GPU] ERROR: CUDA initialization failed, using CPU fallback
```

## What's Happening

1. **cuInit(0)** - SUCCESS ✓
2. **cuDeviceGetCount()** - SUCCESS ✓ (finds 1 device)
3. **cuDeviceGet(&dev, 0)** - SUCCESS ✓
4. **cuDevicePrimaryCtxRetain(&ctx, dev)** - FAILS ✗ EVERY TIME

This is the CUDA initialization code that's failing:

```cpp
// gpu_kernel_driver_prod.cpp, line 166-172
if (drv.cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS) {
    if (log) {
        fprintf(log, "[GPU] ERROR: cuDevicePrimaryCtxRetain failed\n");
        fclose(log);
    }
    return false;
}
```

## Timeline of What Broke It

1. We had CUDA_ERROR_ILLEGAL_ADDRESS errors
2. We added cleanup/recovery code
3. Now cuDevicePrimaryCtxRetain fails ALWAYS
4. Possibly the context is in a permanently corrupted state?

## Previous Working State

Earlier today, CUDA was working but failing at cuMemcpyHtoD with ILLEGAL_ADDRESS:
```
[GPU] Device pointers: d_params=0000001417a00000 d_found=0000001417a00400 ...
[GPU] CUDA ERROR: CUDA_ERROR_ILLEGAL_ADDRESS (700): an illegal memory access was encountered at src/gpu/gpu_kernel_driver_prod.cpp:517
```

## Current State After Our "Fixes"

Now it can't even create a context:
```
[GPU] ERROR: cuDevicePrimaryCtxRetain failed
```

## What We Changed

1. Added CHECK_CUDA_LOG macro with error recovery
2. Added cleanup_cuda() that calls cleanup_gpu_driver()
3. Added context validation before operations
4. The cleanup might have left GPU in bad state?

## Cleanup Code That Might Be The Problem

```cpp
static void cleanup_cuda() {
    cleanup_gpu_driver();
}

extern "C" void cleanup_gpu_driver() {
    std::lock_guard<std::mutex> lock(init_mutex);
    
    if (!ready.load()) return;
    ready.store(false);
    
    // Set context and free resources
    if (ctx && drv.cuCtxSetCurrent) {
        drv.cuCtxSetCurrent(ctx);
        
        if (d_found) drv.cuMemFree(d_found);
        if (d_result) drv.cuMemFree(d_result);
        if (d_params) drv.cuMemFree(d_params);
        if (d_candidates) drv.cuMemFree(d_candidates);
        if (d_cand_count) drv.cuMemFree(d_cand_count);
        
        if (mod) drv.cuModuleUnload(mod);
        
        drv.cuDevicePrimaryCtxRelease(0);  // <-- SUSPICIOUS! Passing 0 instead of dev
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
}
```

## SUSPECT #1: Line 734
```cpp
drv.cuDevicePrimaryCtxRelease(0);  // <-- Should be dev, not 0!
```

We're releasing device 0 literally, but we should pass the CUdevice handle!

## Other Observations

1. The illegal address error triggered cleanup
2. Cleanup called cuDevicePrimaryCtxRelease(0) - WRONG!
3. Now cuDevicePrimaryCtxRetain fails forever
4. The primary context for the device might be in a corrupted/unreleased state

## System Info
- Windows 11
- NVIDIA RTX 4090
- CUDA Driver 561.09
- Device 0 is detected but context can't be retained

## Questions for Distinguished Engineer

1. **Is cuDevicePrimaryCtxRelease(0) the bug?** Should it be cuDevicePrimaryCtxRelease(dev)?

2. **How to recover from this state?** The context seems permanently broken. Do we need to:
   - Reset the device?
   - Use cuCtxCreate instead of primary context?
   - Restart the driver/system?

3. **Why does cuDevicePrimaryCtxRetain fail?** Possible reasons:
   - Context still retained elsewhere?
   - Device in error state?
   - Driver corrupted?
   - Reference counting issue?

4. **Should we use cuCtxCreate instead?** Primary context might be less robust?

5. **How to properly cleanup on ILLEGAL_ADDRESS?** Current approach seems to break everything worse.

## Immediate Fix Attempt

Change line 734:
```diff
-        drv.cuDevicePrimaryCtxRelease(0);
+        drv.cuDevicePrimaryCtxRelease(dev);
```

But `dev` is not in scope there! We need to store it globally or pass it in.

## Full CUDA Driver Loader Being Used

We're using cuda_driver_loader.h which dynamically loads nvcuda.dll. Could there be an issue with the function pointers?

## Error Not Being Reported

cuDevicePrimaryCtxRetain is returning non-SUCCESS but we're not logging WHICH error code. Need to add:

```cpp
CUresult res = drv.cuDevicePrimaryCtxRetain(&ctx, dev);
if (res != CUDA_SUCCESS) {
    const char* err_name = nullptr;
    drv.cuGetErrorName(res, &err_name);
    fprintf(log, "[GPU] ERROR: cuDevicePrimaryCtxRetain failed: %s (%d)\n", err_name, res);
    return false;
}
```

## HELP NEEDED URGENTLY

1. How to recover from this broken state?
2. Is the cuDevicePrimaryCtxRelease(0) the smoking gun?
3. Should we abandon primary context and use cuCtxCreate?
4. What's the proper cleanup sequence after ILLEGAL_ADDRESS?

The entire GPU acceleration is dead and users are getting wrong results from the CPU fallback!