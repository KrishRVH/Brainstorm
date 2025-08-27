# Brainstorm Test Documentation

## Overview
This document describes the testing approach for the Brainstorm mod, including unit tests, integration tests, and debugging procedures.

## Test Structure

### Basic Tests (`basic_test.lua`)
Simple Lua-based tests that work without LuaJIT:
- File existence checks
- DLL size validation
- Config syntax verification
- Logger module validation

**Run with:** `lua basic_test.lua`

### Full Test Suite (`run_tests.lua`)
Comprehensive tests requiring LuaJIT:
- Erratic deck validation
- Save state compression/decompression
- CUDA fallback behavior
- DLL interface testing

**Run with:** `lua run_tests.lua` (or `luajit run_tests.lua`)

### C++ Unit Tests (`ImmolateCPP/tests/`)
Native tests for the DLL implementation:
- `test_critical_functions.cpp` - Core algorithm validation
- `test_memory_safety.cpp` - Memory leak detection
- `test_gpu_validation.cpp` - GPU initialization and queries

## Testing GPU Implementation

### 1. Pre-Flight Checks
Before enabling GPU, verify the build:
```bash
# Check DLL size
ls -lh Immolate.dll
# Should be 2.6MB for GPU-enabled build

# Check for CUDA files
ls -lh seed_filter.ptx seed_filter.fatbin
# Both should be present
```

### 2. Safe GPU Testing Procedure

#### Step 1: Disable GPU in Config
```lua
-- config.lua
use_cuda = false,
```

#### Step 2: Test Basic Functionality
- Launch Balatro with mod
- Press Ctrl+T to open settings
- Press Ctrl+R for manual reroll
- Verify no crashes

#### Step 3: Enable GPU
```lua
-- config.lua
use_cuda = true,
debug_enabled = true,
```

#### Step 4: Monitor Initialization
Check `gpu_debug.log` in mod folder:
```
[GPU INIT] CUDA runtime loaded: cudart64_12.dll
[GPU INIT] Found 1 CUDA devices
[GPU INIT] Device name: NVIDIA GeForce RTX 4090, Compute: 8.9
[GPU INIT] GPU initialization successful
```

#### Step 5: Test Auto-Reroll
- Press Ctrl+A to start auto-reroll
- Monitor for crashes
- Check gpu_debug.log for errors

### 3. Debugging GPU Crashes

#### Crash Indicators
Look for these patterns in `gpu_debug.log`:

**Struct Mismatch (FIXED)**:
```
[GPU INIT] Device name: ???, Compute: 1024.64
```
This indicates ABI mismatch - update to latest build.

**Missing CUDA Runtime**:
```
[GPU] Failed to load any CUDA runtime DLL
```
Copy appropriate `cudart64_*.dll` to mod folder.

**Insufficient Compute Capability**:
```
[GPU INIT] Device does not meet requirements (Compute < 6.0)
```
GPU too old - will fall back to CPU.

### 4. Performance Testing

#### CPU Baseline
```lua
-- Disable GPU
use_cuda = false
-- Note seeds/second in debug output
```

#### GPU Performance
```lua
-- Enable GPU
use_cuda = true
-- Compare seeds/second
-- Expect 10-100x improvement for complex filters
```

## Integration Testing

### Test Scenarios

#### 1. Dual Tag Search
- Set Tag1: "Investment Tag"
- Set Tag2: "Investment Tag" (same)
- Expected: ~0.1% match rate

#### 2. Erratic Deck Limits
- Face cards: 0-23 (realistic)
- Suit ratio: up to 75%
- Test boundary conditions

#### 3. Save States
- Create state with Z+1
- Make changes
- Load with X+1
- Verify restoration

#### 4. DLL Fallback
- Rename Immolate.dll temporarily
- Verify mod still works (slowly)
- Restore DLL

## Automated Testing

### GitHub Actions (Future)
```yaml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build DLL
        run: |
          cd ImmolateCPP
          ./build_simple.sh
      - name: Run Tests
        run: |
          lua basic_test.lua
```

### Local Test Script
```bash
#!/bin/bash
# test_all.sh
echo "=== Running All Tests ==="

# Lua tests
lua basic_test.lua || exit 1

# Build tests
cd ImmolateCPP
./build_simple.sh || exit 1
./build_gpu.sh || exit 1

# Check outputs
ls -lh Immolate.dll | grep "2.[46]M" || exit 1

echo "=== All Tests Passed ==="
```

## Memory Testing

### Detecting Leaks
```cpp
// Use AddressSanitizer
cd ImmolateCPP
mkdir build && cd build
cmake -DUSE_ASAN=ON ..
make test_with_sanitizers
```

### Known Issues
- `strdup()` allocations in DLL interface
- Need to call `free_result()` from Lua

## Test Coverage Goals

### Current Coverage (~30%)
- Core filter logic
- Basic GPU initialization
- Save/load states

### Target Coverage (80%)
- All filter combinations
- Edge cases (empty strings, nulls)
- GPU error paths
- Memory management
- Thread safety

## Debugging Tools

### 1. File-Based Logging
Most reliable for crash debugging:
```cpp
FILE* f = fopen("debug.log", "a");
fprintf(f, "Message\n");
fflush(f);  // Critical!
fclose(f);
```

### 2. Console Output
For non-crash scenarios:
```lua
log:debug("Message", {data = value})
```

### 3. GPU Debug Mode
Compile with `-DDEBUG_GPU=1` for kernel traces.

## Best Practices

1. **Always test CPU-only first** - Establish baseline
2. **Use file logging for crashes** - Console output lost on crash
3. **Check DLL size** - Validates correct build
4. **Monitor gpu_debug.log** - First indicator of issues
5. **Test incrementally** - Enable features one by one
6. **Keep logs** - Archive interesting failures

## Common Test Failures

### "DLL not found"
- Check deployment succeeded
- Verify path in Brainstorm.lua

### "0 seeds/second"
- Parameter count mismatch
- Check enhanced (8) vs original (7) params

### "GPU init timeout"
- Driver issue or hung GPU
- Reboot and try again

### "Corrupt compute capability"
- Struct mismatch (use latest build)
- Check using real CUDA headers

## Validation Checklist

Before releasing:
- [ ] CPU-only build works
- [ ] GPU build works on RTX 2060+
- [ ] Fallback works without DLL
- [ ] Save states functional
- [ ] Dual tags working
- [ ] No memory leaks
- [ ] Documentation updated
- [ ] Tests passing