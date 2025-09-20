# Critical Issue: GPU Search Returning Wrong Seeds

## Executive Summary
The Brainstorm mod is exhibiting critical failures in production:
1. GPU search immediately returns incorrect seeds that don't match ANY filters
2. CUDA memory operations failing with "Failed to upload filter params"
3. Auto-reroll continues running despite finding (wrong) seeds
4. No Lua-side debug logs appearing despite debug_enabled=true

## Current Symptoms

### User Experience
- User presses Ctrl+A to start auto-reroll with filters (e.g., Double Tag + Investment Tag + Spectral Pack)
- Game immediately reloads to a seed (indicating a "match" was found)
- The seed clearly doesn't match the filters when checked
- Auto-reroll continues searching in the background

### Log Analysis

#### GPU Driver Log Pattern
```
[GPU] gpu_search_with_driver called with seed: EGS3IIQH
[GPU] Filter params - voucher:1 pack1:2 pack2:2 tag1(s=17,b=17) tag2(s=6,b=6) souls:0 obs:0 perkeo:0
[GPU] CUDA initialized, setting context
[GPU] Seed converted to numeric: 1133625770585
[GPU] ERROR: Failed to upload filter params
```

This pattern repeats for EVERY search attempt.

#### DLL Debug Log Pattern
```
[DLL] === BRAINSTORM ENTRY ===
[DLL] Input parameters:
[DLL]   seed       = 'EGS3IIQH' (ptr=00000000283d2c00)
[DLL]   voucher    = 'Clearance Sale' (ptr=0000000026b6e200)
[DLL]   pack       = 'Spectral Pack' (ptr=0000000026b664b0)
[DLL]   tag1       = 'Double Tag' (ptr=0000000026b6d748)
[DLL]   tag2       = 'Investment Tag' (ptr=0000000026b6d798)
[DLL] Resolved indices: voucher=1, pack1=2, pack2=2, tag1(s=17,b=17), tag2(s=6,b=6)
```

The indices ARE being resolved correctly now (no more 0xFFFFFFFF), but the GPU upload fails.

## Root Cause Analysis

### Issue 1: CUDA Memory Management Failure

The `cuMemcpyHtoD` call is failing consistently. Here's the relevant code:

```cpp
// gpu_kernel_driver_prod.cpp, line 475-481
if (drv.cuMemcpyHtoD(d_params, &params, sizeof(FilterParams)) != CUDA_SUCCESS) {
    if (log) {
        fprintf(log, "[GPU] ERROR: Failed to upload filter params\n");
        fclose(log);
    }
    return "";  // Returns empty string, which may be interpreted as "no match"
}
```

### Issue 2: Initialization Check Was Incomplete

We attempted to fix this by checking all device pointers:

```cpp
// gpu_kernel_driver_prod.cpp, line 105-107 (AFTER FIX)
// Already initialized? Check all required resources
if (ready.load() && ctx && mod && fn && d_params && d_result && d_found) {
    return true;
}
```

However, the issue persists, suggesting either:
1. The pointers are non-null but invalid (dangling)
2. The CUDA context itself is corrupted
3. The fix wasn't properly applied (though we verified the DLL was updated)

### Issue 3: No Lua Debug Output

Despite `debug_enabled=true` in config.lua, no Lua debug logs appear. We fixed the logging function:

```lua
-- Brainstorm.lua, line 166-185
Brainstorm.debug_log = function(module, format, ...)
  if not Brainstorm.config or not Brainstorm.config.debug_enabled then
    return
  end
  
  local timestamp = os.date("%H:%M:%S")
  local message = string.format(format, ...)
  local full_msg = string.format("[%s] [%-12s] %s", timestamp, module, message)
  
  -- Direct file write (always append)
  local log_path = (Brainstorm.PATH or "") .. "/debug_full.log"
  local file = io.open(log_path, "a")
  if file then
    file:write(full_msg .. "\n")
    file:close()
  end
  
  -- Also print to console
  print(full_msg)
end
```

But still no output, suggesting the Lua code path isn't being executed as expected.

### Issue 4: Wrong Seed Returned

When GPU search fails (returns empty string), something is still triggering a seed reload. This could be:
1. Fallback logic incorrectly returning the current/starting seed
2. Empty string being misinterpreted as a valid seed
3. Race condition with auto-reroll state

## Code Flow Analysis

### Expected Flow
1. User presses Ctrl+A → triggers `Brainstorm.auto_reroll()`
2. Lua calls `brainstorm_update_pools()` to send pool JSON to DLL
3. Lua calls `brainstorm()` with 8 parameters
4. DLL resolves filter indices using pool manager
5. DLL calls `gpu_search_with_driver()` 
6. GPU uploads params and searches
7. Returns matching seed or "RETRY"

### Actual Flow
1. User presses Ctrl+A
2. NO Lua debug logs appear (suggesting pool update might not run)
3. DLL receives call with correct parameters
4. Filter indices resolve correctly
5. GPU fails to upload params → returns ""
6. Empty string somehow triggers seed reload
7. Auto-reroll continues (wrong state management)

## Critical Code Sections

### GPU Initialization (gpu_kernel_driver_prod.cpp)
```cpp
static bool initialize_cuda() {
    std::lock_guard<std::mutex> lock(init_mutex);
    
    // Already initialized? Check all required resources
    if (ready.load() && ctx && mod && fn && d_params && d_result && d_found) {
        return true;
    }
    
    // ... initialization code ...
    
    // Allocate GPU memory
    if (drv.cuMemAlloc(&d_params, sizeof(FilterParams)) != CUDA_SUCCESS ||
        drv.cuMemAlloc(&d_result, sizeof(uint64_t)) != CUDA_SUCCESS ||
        drv.cuMemAlloc(&d_found, sizeof(int)) != CUDA_SUCCESS ||
        drv.cuMemAlloc(&d_candidates, CANDIDATE_CAPACITY * sizeof(uint64_t)) != CUDA_SUCCESS ||
        drv.cuMemAlloc(&d_cand_count, sizeof(uint32_t)) != CUDA_SUCCESS) {
        
        // Cleanup on failure
        if (d_params) drv.cuMemFree(d_params);
        // ... more cleanup ...
        return false;
    }
}
```

### Pool Update (Brainstorm.lua)
```lua
-- Update pools before searching (critical!)
if immolate.brainstorm_update_pools then
  local pools_json = Brainstorm.get_pool_json()
  if pools_json then
    Brainstorm.debug_log("DLL", "Sending pools JSON to DLL: %s", pools_json:sub(1, 200))
    local ok, err = pcall(immolate.brainstorm_update_pools, pools_json)
    if not ok then
      Brainstorm.debug_log("DLL", "ERROR: Failed to update pools: %s", tostring(err))
    else
      Brainstorm.debug_log("DLL", "Pools updated successfully")
    end
  else
    Brainstorm.debug_log("DLL", "ERROR: Failed to generate pools JSON")
  end
else
  Brainstorm.debug_log("DLL", "ERROR: brainstorm_update_pools not found in DLL")
end
```

## Hypotheses

### Hypothesis 1: CUDA Context Corruption
The CUDA context might be getting corrupted between calls, possibly due to:
- Windows TDR (Timeout Detection and Recovery) resetting the GPU
- Another process interfering with CUDA
- Memory corruption from previous failed operations

### Hypothesis 2: Dangling Pointers
The device pointers (d_params, etc.) might be non-null but pointing to freed memory:
- Cleanup might be called from destructor or another path
- Context might be invalidated without clearing pointers

### Hypothesis 3: Race Condition
Multiple rapid calls to brainstorm() might be causing race conditions:
- The logs show multiple calls with the same seed in quick succession
- Mutex might not be protecting all critical sections

### Hypothesis 4: Lua Override
The game might be overriding or bypassing our Lua functions:
- No debug logs suggest our code isn't running
- Game might have cached old version of functions

## Attempted Fixes

1. **Fixed debug logging** - Made it write directly to file
2. **Fixed pool JSON format** - Changed from `{ctx:[...]}` to named contexts
3. **Added pool update call** - Before each DLL search
4. **Fixed CUDA init check** - Added device pointer validation
5. **Removed obsolete validation** - Alphanumeric seed warnings

## Questions for Distinguished Engineer

1. **CUDA Memory Management**: Should we be checking `cuCtxGetCurrent()` before each operation to ensure context validity? Should we recreate the context if it's lost?

2. **Error Recovery**: When `cuMemcpyHtoD` fails, should we attempt full reinitialization rather than returning empty string?

3. **Thread Safety**: Are we missing critical sections? The mutex protects `initialize_cuda()` but not the actual search operations.

4. **State Management**: How should we handle the auto-reroll state when GPU search fails? Currently it seems to continue despite "finding" a seed.

5. **Lua Integration**: Why might our Lua debug logs not appear? Is there a way to force our functions to be called?

6. **Fallback Logic**: Should empty string from GPU trigger CPU fallback? Current code seems to treat "" as a valid result.

## Urgent Request

This is blocking the v1.0 release. Users are experiencing false positives where the mod claims to find seeds that don't match their filters. We need guidance on:

1. Proper CUDA error recovery strategy
2. Ensuring Lua→DLL→GPU pipeline integrity  
3. Correct handling of search failures vs. no matches
4. Debugging why Lua logs aren't appearing

## Environment Details
- Windows 11
- NVIDIA RTX 4090
- CUDA Driver 561.09
- LuaJIT 2.1 (via Balatro)
- MinGW-w64 compiler
- WSL2 Ubuntu for development

## Reproduction Steps
1. Set filters (any combination of voucher/pack/tags)
2. Press Ctrl+A to start auto-reroll
3. Observe immediate reload to wrong seed
4. Check logs showing "Failed to upload filter params"

## Files to Review
- `/ImmolateCPP/src/gpu/gpu_kernel_driver_prod.cpp` - GPU driver with CUDA operations
- `/ImmolateCPP/src/brainstorm_driver.cpp` - Main DLL entry point
- `/Core/Brainstorm.lua` - Lua mod integration (lines 1770-1800 for pool update)
- `/ImmolateCPP/src/pool_manager.cpp` - Pool resolution logic

---

**Please advise on the best path forward. The community is eagerly awaiting a working release.**