# CRITICAL: Complete GPU Failure Analysis - Need Distinguished Engineer Help

## Executive Summary
**The GPU kernel fails with `CUDA_ERROR_ILLEGAL_ADDRESS` on EVERY first real launch, despite implementing ALL recommended fixes.** Calibration works, but real searches fail immediately. We've fixed context management, pointer safety, null checks, but something fundamental is still wrong.

## Failure Pattern (100% Reproducible)

```
1. System reboot → Balatro start → GPU initializes perfectly
2. Calibration kernel runs successfully (10M seeds/sec)  
3. First real kernel launch → CUDA_ERROR_ILLEGAL_ADDRESS
4. GPU permanently corrupted until reboot
```

## Current Implementation State

### Successfully Implemented Fixes
✅ Explicit context (cuCtxCreate) instead of primary  
✅ Context current checks before ALL operations  
✅ Device pointer registry and validation  
✅ Mutex serialization of searches  
✅ Sticky error detection and recovery  
✅ Pools pointer set to nullptr (was host pointer bug)  
✅ Kernel checks for null pools before access  
✅ Soft/hard reset functions  
✅ Smoke test capability  

### The Kernel Launch That Fails

```cpp
// gpu_kernel_driver_prod.cpp lines 788-823
void* args[] = {
    &current_start,    // uint64_t start_seed_index = 729437542727
    &count,           // uint32_t total_seeds = 2500000  
    &chunk_size,      // uint32_t chunk_size = 1024
    &d_params,        // CUdeviceptr params = 0x1417a00000 (valid)
    &d_pools_ptr,     // CUdeviceptr pools = 0 (nullptr)
    &d_candidates,    // CUdeviceptr candidates = 0x1417a00600 (valid)
    &cap,             // uint32_t cap = 4096
    &d_cand_count,    // CUdeviceptr cand_count = 0x1417a08600 (valid)
    &d_found          // CUdeviceptr found = 0x1417a00400 (valid)
};

CUresult launch_result = drv.cuLaunchKernel(
    fn,                     // Function handle (valid, used in calibration)
    GRID_SIZE, 1, 1,       // 256x1x1 grid
    BLOCK_SIZE, 1, 1,      // 256x1x1 block = 65536 threads
    0,                     // No shared memory
    0,                     // Default stream
    args,
    nullptr
);
// Returns: CUDA_ERROR_ILLEGAL_ADDRESS (700)
```

### The Kernel Signature

```cuda
// seed_filter_kernel_balatro.cu
extern "C" __global__ void find_seeds_kernel_balatro(
    uint64_t start_seed_index,
    uint32_t total_seeds,
    uint32_t chunk_size,
    const FilterParams* params,  // 40-byte struct
    const DevicePools* pools,    // Can be null
    uint64_t* candidates,
    uint32_t cap,
    uint32_t* cand_count,
    volatile int* found
)
```

## Critical Observations

### 1. Calibration Works, Real Doesn't
- **Calibration**: start_index=0, count=2000000 → SUCCESS
- **Real**: start_index=729437542727, count=2500000 → ILLEGAL_ADDRESS
- Same kernel, same pointers, different seed values

### 2. The FilterParams Structure (40 bytes)
```cpp
struct FilterParams {
    uint32_t tag1_small;     // 0-4
    uint32_t tag1_big;       // 4-8
    uint32_t tag2_small;     // 8-12
    uint32_t tag2_big;       // 12-16
    uint32_t voucher;        // 16-20
    uint32_t pack1;          // 20-24
    uint32_t pack2;          // 24-28
    uint32_t require_souls;  // 28-32
    uint32_t require_observatory; // 32-36
    uint32_t require_perkeo; // 36-40
};
static_assert(sizeof(FilterParams) == 40);
```

### 3. Device Memory Layout
```
d_params:      0x1417a00000 - 0x1417a00028 (40 bytes)
d_result:      0x1417a00200 - 0x1417a00208 (8 bytes)
d_found:       0x1417a00400 - 0x1417a00404 (4 bytes)
d_candidates:  0x1417a00600 - 0x1417a08600 (32KB)
d_cand_count:  0x1417a08600 - 0x1417a08604 (4 bytes)
```

## Theories We've Tested (All Failed)

### ❌ Theory 1: Primary Context Corruption
- Switched to explicit context (cuCtxCreate)
- Still fails with ILLEGAL_ADDRESS

### ❌ Theory 2: Stale Device Pointers
- Added allocation registry
- Validates all pointers before use
- All pointers confirmed valid

### ❌ Theory 3: Pools Pointer Was Host Memory
- Fixed by passing nullptr
- Kernel modified to handle null pools
- Still fails

### ❌ Theory 4: Context Not Current
- Added ensure_context_current() checks
- Context verified before launch
- Still fails

## Remaining Theories

### Theory A: Large Seed Value Overflow
```cuda
// In kernel:
char seed[8];
idx_to_chars(start_seed_index + chunk_start, seed);
// start_seed_index = 729437542727 (0xA9D4C38247)
// Could overflow cause illegal access?
```

### Theory B: Kernel Stack Overflow
- Complex kernel with multiple function calls
- Local variables in nested functions
- Default stack size might be too small?

### Theory C: Argument Passing Mismatch
- Driver API uses void* args[]
- Are we passing addresses correctly?
- Alignment issues between 32/64-bit values?

### Theory D: PTX JIT Bug
```bash
# Compilation flags
nvcc -ptx -arch=sm_80 -O3
    -Xptxas -fmad=false  # Disable FMA
    -prec-div=true       # Precise division
    -prec-sqrt=true      # Precise sqrt
```
- Could these flags cause issues?
- JIT compilation for RTX 4090 (sm_89)?

## Debugging Attempts

### What We Can't Do (Windows Limitations)
- ❌ cuda-memcheck (not available)
- ❌ cuda-gdb (Linux only)
- ❌ Nsight Compute (requires special driver)
- ❌ printf in kernel (no console)

### What We've Tried
- ✅ Extensive logging (shows failure point)
- ✅ Pointer validation (all valid)
- ✅ Context checks (current)
- ✅ Simplified pools (nullptr)

## Critical Code Sections

### 1. Kernel Entry (Where It Might Fail)
```cuda
// First lines of kernel
const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
const uint32_t stride = blockDim.x * gridDim.x;
const uint32_t total_chunks = (total_seeds + chunk_size - 1) / chunk_size;

// Early exit check
#define CHECK_EVERY 16
uint32_t local_counter = tid;
if ((local_counter & (CHECK_EVERY - 1)) == 0 && *found) return;
// Could dereferencing 'found' fail?
```

### 2. Seed Conversion (Arithmetic)
```cuda
__device__ void idx_to_chars(uint64_t idx, char* out) {
    for (int i = 7; i >= 0; i--) {
        uint64_t val = idx % 36;
        out[i] = (val < 10) ? ('0' + val) : ('A' + val - 10);
        idx /= 36;
    }
}
// idx = 729437542727 - could this overflow?
```

### 3. Memory Access Pattern
```cuda
for (uint32_t chunk = tid; chunk < total_chunks; chunk += stride) {
    uint32_t chunk_start = chunk * chunk_size;
    // chunk_size = 1024
    // If tid=65535, chunk could be huge
    // chunk_start could overflow?
}
```

## Build Environment
```bash
# Host: WSL2 Ubuntu on Windows 11
# Target: Windows x64 DLL
# Compiler: x86_64-w64-mingw32-g++ (MinGW)
# CUDA: nvcc from /usr/local/cuda/bin/
# GPU: RTX 4090 (Compute 8.9)
# Driver: 561.09 (Windows)
```

## What We Need From You

### 1. Debug Strategy
How can we identify WHICH line causes ILLEGAL_ADDRESS without cuda-memcheck?

### 2. Simplification Test
Should we try a minimal kernel that just writes one value?
```cuda
extern "C" __global__ void test_kernel(int* found) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *found = 42;
    }
}
```

### 3. Argument Verification
Is our args array correct for Driver API?
```cpp
void* args[] = {
    &uint64_value,     // Address of uint64_t
    &uint32_value,     // Address of uint32_t  
    &deviceptr_value,  // Address of CUdeviceptr
};
```

### 4. Alternative Approaches
- Should we try CUDA Runtime API instead?
- Reduce grid/block size?
- Split kernel into smaller pieces?
- Use different PTX flags?

### 5. System-Level Issues
- Known issues with Driver 561.09?
- WSL2 cross-compilation problems?
- Windows WDDM timeout issues?

## File Locations for Investigation
```
/home/krvh/personal/Brainstorm/ImmolateCPP/
├── src/gpu/gpu_kernel_driver_prod.cpp    # Lines 788-823 (launch)
├── src/gpu/seed_filter_kernel_balatro.cu # Kernel implementation
├── src/gpu/gpu_types.h                   # FilterParams definition
└── build.sh                               # Compilation flags
```

## Summary
Despite implementing EVERY recommended fix, the kernel fails immediately with ILLEGAL_ADDRESS. The GPU hardware works (calibration succeeds) but something about our real kernel launch is fundamentally wrong. We need expert help to identify what's causing the illegal memory access.

**This is blocking the entire GPU feature. Users cannot use GPU acceleration at all.**