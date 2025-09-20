# CUDA Fix Summary - Distinguished Engineer's Plan Implemented

## Deployed Changes (DLL checksum: 30be87a7d79f4c7ac8dad6e8aec9d290)

### Critical Fixes Applied

1. **Switched to Explicit Context (cuCtxCreate)**
   - Now tries `cuCtxCreate()` first for isolated context
   - Falls back to primary context if explicit fails
   - Releases any lingering primary context before retaining

2. **Added Serialization**
   - All searches now protected by mutex
   - Prevents concurrent CUDA operations

3. **Disabled Buggy CPU Fallback**
   - Returns empty string (triggers RETRY) instead of wrong seeds
   - No more incorrect seeds being applied

4. **Added Diagnostic Exports**
   ```cpp
   brainstorm_get_last_cuda_error()
   brainstorm_is_driver_ready()
   brainstorm_gpu_reset()
   brainstorm_gpu_disable_for_session()
   ```

5. **Proper Cleanup Based on Context Type**
   - Tracks whether using explicit or primary context
   - Uses `cuCtxDestroy()` for explicit contexts
   - Uses `cuDevicePrimaryCtxRelease(dev)` for primary contexts

6. **Safety Gates**
   - DLL already validates 8-char [0-9A-Z] seeds
   - Returns "RETRY" for any invalid/empty results
   - Driver broken flag prevents operations when GPU disabled

## What This Should Fix

1. **CUDA_ERROR_ILLEGAL_ADDRESS during context creation**
   - Explicit context isolates from corrupted primary context
   - Release before retain clears lingering references

2. **Wrong seeds being applied**
   - CPU fallback disabled (was buggy)
   - Only valid 8-char seeds accepted
   - Returns RETRY on any GPU failure

3. **Thread safety issues**
   - Mutex serializes all search operations
   - Context current checks before operations

## Next Steps After Restart

**You MUST completely restart Balatro** for these changes to take effect.

After restart, the GPU will:
1. Try to create an explicit context (more reliable)
2. If that fails, try primary context with release first
3. If both fail, GPU will be disabled for session
4. No wrong seeds will be applied

## If Still Broken

If CUDA still fails after restart:
1. Check gpu_driver.log for new error pattern
2. The explicit context should work around the corruption
3. If not, may need system reboot to clear driver state

## What We Didn't Implement Yet

- CUdeviceptr fixes (lower priority, current code works)
- Error recovery with retry (can add if needed)
- Hard reset function (can add if needed)
- Lua-side validation (DLL already does it)