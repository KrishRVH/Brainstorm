# GPU Debug Instrumentation Guide

## Overview

The GPU implementation includes comprehensive debug instrumentation for troubleshooting crashes, struct mismatches, and performance issues.

## Critical: Debugging Crash Issues (Updated 2025-08-24)

### File-Based Debug Logging

The most reliable way to debug GPU crashes is through file-based logging that survives process termination:

```cpp
// In brainstorm_unified.cpp and gpu_searcher_dynamic.cpp
FILE* debug_file = fopen("C:\\Users\\[YourName]\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm\\gpu_debug.log", "a");
if (debug_file) {
    fprintf(debug_file, "[GPU INIT] Message here\n");
    fflush(debug_file);  // CRITICAL: Flush immediately to survive crashes
    fclose(debug_file);
}
```

### Common Crash Patterns

1. **Struct ABI Mismatch** (RESOLVED in latest version)
   - **Symptom**: Bogus compute capability like "1024.64" instead of "8.9"
   - **Cause**: Hand-rolled `cudaDeviceProp` struct layout mismatch
   - **Fix**: Use real CUDA headers and attribute-based queries

2. **Stack Corruption**
   - **Symptom**: Crash after "GPUSearcher constructor completed"
   - **Cause**: CUDA runtime writing past allocated struct memory
   - **Fix**: Use `cudaDeviceGetAttribute` instead of `cudaGetDeviceProperties`

### Compile-Time Debug

To enable kernel debug output, compile with the `DEBUG_GPU` flag:

```bash
# Enable GPU debugging
nvcc -DDEBUG_GPU=1 -o seed_filter seed_filter.cu

# Or in the build script, add:
NVCCFLAGS="-DDEBUG_GPU=1"
```

### Debug Features

When debugging is enabled:

1. **File-based logging** - Survives crashes, written to `gpu_debug.log`
2. **Detailed initialization trace** - Every CUDA API call logged
3. **Input validation** - All parameters are bounds-checked
4. **Algorithm tracing** - Step-by-step RNG calculations
5. **Performance statistics** - Automatic stats collection and reporting
6. **Thread identification** - Block/thread IDs in all messages

## Debug Output Format

### Thread Identification
```
[GPU] Thread <block>.<thread> (global <idx>): message
```
Example: `[GPU] Thread 42.15 (global 10767): starting seed test`

### Algorithm Stages
- `[GPU] pseudohash_device: input='...' len=N`
- `[GPU] pseudorandom_device: seed=0x... key_hash=0x...`  
- `[GPU] get_tag: seed=0x... ante=1 blind=0`
- `[GPU] get_voucher: seed=0x... random_val=0x... voucher_id=N`

### Match/Rejection Tracking
- `[GPU] Thread X.Y: TAG MATCH (single_tag) - continuing`
- `[GPU] Thread X.Y: VOUCHER REJECTION (got 5, wanted 12) - seed failed`
- `[GPU] Thread X.Y: *** COMPLETE MATCH *** seed=0x... passed all filters`

### Performance Statistics
```
[GPU STATS] Seeds tested: 1048576
[GPU STATS] Tag matches: 38401 (rejections: 1010175)
[GPU STATS] Voucher matches: 1200 (rejections: 37201)
[GPU STATS] Pack matches: 80 (rejections: 1120)
[GPU STATS] Total final matches: 1
```

## Debug Output Management

### Controlling Verbosity

The debug output can be very verbose. To filter:

```bash
# Show only matches and errors
./program 2>&1 | grep -E "(MATCH|ERROR|STATS)"

# Show only specific thread
./program 2>&1 | grep "Thread 0.0"

# Show only algorithm results
./program 2>&1 | grep -E "(tag_id=|voucher_id=|pack_id=)"
```

### Performance Impact

Debug mode significantly slows execution:
- **Production**: ~10M seeds/second
- **Debug mode**: ~100K seeds/second  

Only enable for troubleshooting, not production use.

## Validation Checks

The debug version validates:

### Input Ranges
- Tag IDs: 0-26 (27 possible)
- Voucher IDs: 0-31 (32 possible)
- Pack IDs: 0-14 (15 possible)
- Seed characters: A-Z only
- Ante values: 1-99
- Blind values: 0-1

### Algorithm Correctness
- No NaN/Inf in floating point calculations
- Hash values stay in [0,1] range
- String lengths within expected bounds
- Atomic operations succeed

### Thread Safety
- Race condition detection in result storage
- Memory access validation
- Atomic counter verification

## Troubleshooting Common Issues

### "Tag out of range" errors
- Check that `NUM_TAGS = 27` matches Balatro's tag count
- Verify pseudorandom calculation doesn't overflow

### "Invalid seed character" errors  
- Seeds must be 8 uppercase letters A-Z
- Check seed packing/unpacking logic

### "Kernel launch failed"
- GPU memory insufficient 
- Too many threads requested
- CUDA driver issues

### Performance degradation
- Disable debug mode for production
- Check thread block size (256-512 optimal)
- Monitor GPU temperature/throttling

## Analysis Scripts

Create helper scripts for log analysis:

### Parse Performance
```bash
#!/bin/bash
# extract_perf.sh
grep "GPU STATS" brainstorm.log | tail -5
```

### Find Hotspots  
```bash
#!/bin/bash
# find_hotspots.sh
grep -E "Thread.*:" brainstorm.log | cut -d: -f2 | sort | uniq -c | sort -nr | head -10
```

### Track Match Rates
```bash
#!/bin/bash
# match_rates.sh  
total=$(grep "seeds_tested:" brainstorm.log | tail -1 | awk '{print $3}')
matches=$(grep "total_matches:" brainstorm.log | tail -1 | awk '{print $3}')
echo "Match rate: $matches/$total = $(echo "scale=6; $matches/$total*100" | bc)%"
```

## Integration with Host Code

To use debug stats from C++:

```cpp
// Allocate debug statistics on GPU
DebugStats* d_debug_stats;
cudaMalloc(&d_debug_stats, sizeof(DebugStats));

// Launch with debug support
launch_seed_search(start_seed, count, d_params, d_result, d_found, d_debug_stats);

// Retrieve and print stats
DebugStats host_stats;
cudaMemcpy(&host_stats, d_debug_stats, sizeof(DebugStats), cudaMemcpyDeviceToHost);
printf("Seeds tested: %llu\n", host_stats.seeds_tested);

// Cleanup
cudaFree(d_debug_stats);
```

## Best Practices

1. **Development**: Always test with debug mode first
2. **Performance testing**: Disable debug for accurate benchmarks  
3. **Production**: Never ship with debug enabled
4. **Logging**: Redirect stdout to files for large runs
5. **Validation**: Run small batches first to verify correctness

## Example Debug Session

```bash
# Compile with debug
cd ImmolateCPP
nvcc -DDEBUG_GPU=1 -o test_kernel seed_filter.cu

# Run small test
./test_kernel 2>&1 | tee debug.log

# Analyze results  
grep "COMPLETE MATCH" debug.log
grep "GPU STATS" debug.log | tail -5
```

This instrumentation should help identify any algorithm discrepancies or performance bottlenecks in the GPU implementation.

## Solution Approach for CUDA Crashes (2025-08-24)

### The Root Cause
The primary cause of CUDA-related crashes was **ABI mismatch** between hand-rolled CUDA types and the actual CUDA runtime. When `cudaGetDeviceProperties` wrote to our custom struct, it expected a different memory layout, causing stack corruption.

### The Fix Applied
1. **Use Real Headers**: Include `<cuda_runtime_api.h>` for correct type definitions
2. **Attribute Queries**: Use `cudaDeviceGetAttribute` for individual properties
3. **Dynamic Loading**: Properly resolve both `_v2` and legacy function symbols
4. **Include Paths**: Add CUDA include directory to build commands

### Build Configuration
```bash
# Detect CUDA headers
CUDA_INC="/usr/local/cuda/include"

# Include in compilation
x86_64-w64-mingw32-g++ \
    -DGPU_ENABLED \
    -I "$CUDA_INC" \
    -I ../src/ \
    ...
```

### Verification
A properly fixed build will show:
- DLL size: 2.6MB (up from 2.4MB CPU-only)
- gpu_debug.log: "Device name: NVIDIA GeForce RTX 4090, Compute: 8.9"
- No crash on Ctrl+A (auto-reroll)

### Future-Proofing
- Always use official headers when available
- Prefer attribute-based queries over struct-based ones
- Test with file-based logging that survives crashes
- Verify struct sizes match between your code and the runtime