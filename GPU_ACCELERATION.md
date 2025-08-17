# GPU Acceleration for Brainstorm Seed Finding

## Overview

Utilizing an RTX 4090 could theoretically provide 100-1000x speedup over CPU-based seed testing by parallelizing the RNG calculations across thousands of CUDA cores.

## Architecture Design

### 1. CUDA Kernel Implementation

```cuda
// seed_filter.cu
#include <cuda_runtime.h>
#include <stdint.h>

struct FilterParams {
    uint32_t tag1;
    uint32_t tag2;
    uint32_t voucher;
    uint32_t pack;
    bool require_souls;
    bool require_observatory;
};

// Balatro's PRNG implementation (needs reverse engineering)
__device__ uint32_t pseudorandom(uint64_t seed, const char* key) {
    // Implementation of Balatro's RNG
    // This is the critical piece that needs reverse engineering
}

__device__ uint32_t get_tag(uint64_t seed, int ante, int blind) {
    char key[32];
    sprintf(key, "Tag_ante_%d_blind_%d", ante, blind);
    return pseudorandom(seed, key) % NUM_TAGS;
}

__global__ void find_seeds_kernel(
    uint64_t start_seed,
    uint32_t count,
    FilterParams params,
    uint64_t* result,
    volatile int* found
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count || *found) return;
    
    uint64_t seed = start_seed + idx;
    
    // Check tags (cheapest operation first)
    uint32_t small_tag = get_tag(seed, 1, 0);
    uint32_t big_tag = get_tag(seed, 1, 1);
    
    bool tags_match = false;
    if (params.tag1 == params.tag2) {
        tags_match = (small_tag == params.tag1 && big_tag == params.tag1);
    } else {
        bool has_tag1 = (small_tag == params.tag1 || big_tag == params.tag1);
        bool has_tag2 = (small_tag == params.tag2 || big_tag == params.tag2);
        tags_match = has_tag1 && has_tag2;
    }
    
    if (!tags_match) return;
    
    // Check voucher if needed
    if (params.voucher != 0xFFFFFFFF) {
        uint32_t first_voucher = pseudorandom(seed, "Voucher_1") % NUM_VOUCHERS;
        if (first_voucher != params.voucher) return;
    }
    
    // Check pack if needed
    if (params.pack != 0xFFFFFFFF) {
        uint32_t first_pack = pseudorandom(seed, "shop_pack_1") % NUM_PACKS;
        if (first_pack != params.pack) return;
    }
    
    // Found a match!
    if (atomicCAS((int*)found, 0, 1) == 0) {
        *result = seed;
    }
}
```

### 2. Host-Side Integration

```cpp
// gpu_searcher.cpp
#include <cuda_runtime.h>
#include <string>

class GPUSearcher {
private:
    FilterParams* d_params;
    uint64_t* d_result;
    int* d_found;
    
public:
    GPUSearcher() {
        cudaMalloc(&d_params, sizeof(FilterParams));
        cudaMalloc(&d_result, sizeof(uint64_t));
        cudaMalloc(&d_found, sizeof(int));
    }
    
    std::string search(uint64_t start_seed, FilterParams params) {
        // Reset found flag
        int zero = 0;
        cudaMemcpy(d_found, &zero, sizeof(int), cudaMemcpyHostToDevice);
        
        // Copy parameters to GPU
        cudaMemcpy(d_params, &params, sizeof(FilterParams), cudaMemcpyHostToDevice);
        
        // Launch kernel with optimal block/grid size for RTX 4090
        // RTX 4090 has 128 SMs, 128 cores per SM = 16,384 cores
        int threads_per_block = 256;
        int blocks = 16384 / threads_per_block;
        uint32_t seeds_per_batch = blocks * threads_per_block;
        
        uint64_t current_seed = start_seed;
        uint64_t max_seeds = 100000000;
        
        for (uint64_t tested = 0; tested < max_seeds; tested += seeds_per_batch) {
            find_seeds_kernel<<<blocks, threads_per_block>>>(
                current_seed, seeds_per_batch, 
                params, d_result, d_found
            );
            
            // Check if found
            int found_flag;
            cudaMemcpy(&found_flag, d_found, sizeof(int), cudaMemcpyDeviceToHost);
            if (found_flag) {
                uint64_t result;
                cudaMemcpy(&result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                return seed_to_string(result);
            }
            
            current_seed += seeds_per_batch;
        }
        
        return "";  // Not found
    }
    
    ~GPUSearcher() {
        cudaFree(d_params);
        cudaFree(d_result);
        cudaFree(d_found);
    }
};
```

### 3. Lua FFI Binding

```lua
-- gpu_ffi.lua
local ffi = require("ffi")

ffi.cdef[[
    typedef struct {
        uint32_t tag1;
        uint32_t tag2;
        uint32_t voucher;
        uint32_t pack;
        bool require_souls;
        bool require_observatory;
    } FilterParams;
    
    void* gpu_searcher_create();
    const char* gpu_searcher_search(void* searcher, uint64_t start_seed, FilterParams params);
    void gpu_searcher_destroy(void* searcher);
    void gpu_searcher_free_result(const char* result);
]]

local gpu_lib = ffi.load("BrainstormGPU.dll")

local GPUSearcher = {}
GPUSearcher.__index = GPUSearcher

function GPUSearcher:new()
    local self = setmetatable({}, GPUSearcher)
    self.handle = gpu_lib.gpu_searcher_create()
    return self
end

function GPUSearcher:search(seed, filters)
    local params = ffi.new("FilterParams")
    params.tag1 = filters.tag1 or 0xFFFFFFFF
    params.tag2 = filters.tag2 or 0xFFFFFFFF
    params.voucher = filters.voucher or 0xFFFFFFFF
    params.pack = filters.pack or 0xFFFFFFFF
    params.require_souls = filters.souls or false
    params.require_observatory = filters.observatory or false
    
    local result = gpu_lib.gpu_searcher_search(self.handle, seed, params)
    if result ~= nil then
        local seed_str = ffi.string(result)
        gpu_lib.gpu_searcher_free_result(result)
        return seed_str
    end
    return nil
end

function GPUSearcher:destroy()
    gpu_lib.gpu_searcher_destroy(self.handle)
end

return GPUSearcher
```

## Build Process

### Prerequisites
- CUDA Toolkit 12.x
- Visual Studio 2022 with CUDA support
- CMake 3.25+

### Build Script (build_gpu.bat)
```batch
@echo off
mkdir build_gpu
cd build_gpu

cmake -G "Visual Studio 17 2022" ^
    -DCMAKE_CUDA_ARCHITECTURES=89 ^
    -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3" ^
    ..

cmake --build . --config Release
copy Release\BrainstormGPU.dll ..\..\
```

### CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.25)
project(BrainstormGPU CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Create shared library
add_library(BrainstormGPU SHARED
    src/seed_filter.cu
    src/gpu_searcher.cpp
    src/ffi_interface.cpp
)

# Set CUDA architecture for RTX 4090 (Ada Lovelace)
set_target_properties(BrainstormGPU PROPERTIES
    CUDA_ARCHITECTURES "89"
    CUDA_SEPARABLE_COMPILATION ON
)

# Link CUDA runtime
target_link_libraries(BrainstormGPU
    CUDA::cudart
)
```

## Performance Expectations

### Theoretical Performance
- **RTX 4090**: 16,384 CUDA cores
- **Clock**: ~2.5 GHz
- **Memory Bandwidth**: 1008 GB/s

### Estimated Throughput
- **Simple filters (tags only)**: 10-50 million seeds/second
- **Complex filters (all checks)**: 1-10 million seeds/second
- **vs CPU (current)**: 100-10,000x speedup

### Bottlenecks
1. **RNG complexity** - If Balatro's PRNG is computationally expensive
2. **Memory access patterns** - Random memory access for lookup tables
3. **Kernel occupancy** - Need to optimize threads per block
4. **CPU-GPU transfer** - Minimal with batch processing

## Implementation Challenges

### 1. Reverse Engineering Balatro's RNG
The biggest challenge is accurately replicating Balatro's pseudorandom number generator:
- Need to extract exact algorithm from obfuscated Lua
- Must match floating point precision
- String hashing for seed keys must be identical

### 2. Erratic Deck Generation
Erratic decks require complex card shuffling that's hard to parallelize:
- Each card depends on previous RNG calls
- May need different kernel for Erratic vs normal

### 3. Complex Game State
Some filters require simulating game state:
- Shop generation with reroll costs
- Pack contents with specific probabilities
- Voucher unlocks and prerequisites

## Alternative: OpenCL for Cross-Platform

If you want AMD GPU support too:
```c
// seed_filter.cl
__kernel void find_seeds(
    __global ulong* start_seed,
    __global uint* params,
    __global ulong* result,
    __global volatile int* found
) {
    int idx = get_global_id(0);
    // Similar logic to CUDA kernel
}
```

## Quick Start Implementation

1. **Start simple**: Port just tag checking to GPU
2. **Benchmark**: Measure speedup vs CPU
3. **Iterate**: Add more filters if performance justifies complexity
4. **Optimize**: Profile and tune kernel parameters

## Integration with Current Codebase

```lua
-- In Brainstorm.lua
local gpu_available = pcall(require, "gpu_ffi")
local gpu_searcher = nil

if gpu_available then
    local GPUSearcher = require("gpu_ffi")
    gpu_searcher = GPUSearcher:new()
    Brainstorm.debug.using_gpu = true
end

-- In auto_reroll function
local found_seed = nil
if gpu_searcher and not Brainstorm.config.ar_prefs.erratic_required then
    -- Use GPU for non-Erratic searches
    found_seed = gpu_searcher:search(current_seed, {
        tag1 = tag_to_id(Brainstorm.config.ar_filters.tag_name),
        tag2 = tag_to_id(Brainstorm.config.ar_filters.tag2_name),
        voucher = voucher_to_id(Brainstorm.config.ar_filters.voucher_name),
        pack = pack_to_id(Brainstorm.config.ar_filters.pack_name)
    })
else
    -- Fall back to CPU DLL
    found_seed = use_current_dll()
end
```

## Next Steps

1. **Reverse engineer** Balatro's exact RNG algorithm
2. **Prototype** simple CUDA kernel for tag checking only
3. **Benchmark** to verify performance gains
4. **Expand** to more complex filters if successful
5. **Package** as drop-in replacement for current DLL

The RTX 4090's massive parallel compute power could turn 30-second searches into sub-second results!