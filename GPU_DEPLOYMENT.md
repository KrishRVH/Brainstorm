# GPU-Accelerated DLL Deployment Strategy

## Yes, It Can Still Be a Single DLL!

The GPU acceleration can be packaged as a drop-in replacement for `Immolate.dll` with automatic CPU fallback.

## Architecture

### 1. Single DLL with Runtime Detection

```cpp
// brainstorm_unified.cpp
#include <cuda_runtime.h>
#include <windows.h>

class BrainstormEngine {
private:
    bool cuda_available = false;
    void* gpu_context = nullptr;
    
public:
    BrainstormEngine() {
        // Check for CUDA at runtime
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        
        if (err == cudaSuccess && device_count > 0) {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, 0);
            
            // Check if it's a capable GPU (RTX 2000 series or newer)
            if (props.major >= 7) {  // Compute capability 7.0+
                cuda_available = true;
                initialize_gpu();
                printf("[Brainstorm] GPU acceleration enabled: %s\n", props.name);
            }
        }
        
        if (!cuda_available) {
            printf("[Brainstorm] GPU not available, using CPU\n");
        }
    }
    
    const char* search(const char* seed, FilterParams params) {
        if (cuda_available) {
            return search_gpu(seed, params);
        } else {
            return search_cpu(seed, params);
        }
    }
};

// Export the same C interface
extern "C" {
    __declspec(dllexport) const char* brainstorm(
        const char* seed,
        const char* voucher,
        const char* pack,
        const char* tag1,
        const char* tag2,
        double souls,
        bool observatory,
        bool perkeo
    ) {
        static BrainstormEngine engine;
        return engine.search(seed, {voucher, pack, tag1, tag2, souls, observatory, perkeo});
    }
}
```

### 2. Build Configuration

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.25)
project(Immolate CUDA CXX)

# Make CUDA optional
include(CheckLanguage)
check_language(CUDA)

option(USE_CUDA "Enable GPU acceleration" ON)

if(CMAKE_CUDA_COMPILER AND USE_CUDA)
    enable_language(CUDA)
    add_definitions(-DGPU_ENABLED)
    set(CUDA_SOURCES 
        src/gpu/seed_filter.cu
        src/gpu/gpu_searcher.cpp
    )
else()
    message(STATUS "Building without GPU support")
    set(CUDA_SOURCES "")
endif()

# Create unified DLL
add_library(Immolate SHARED
    src/brainstorm_unified.cpp
    src/cpu/brainstorm_enhanced.cpp
    src/cpu/items.cpp
    src/cpu/rng.cpp
    ${CUDA_SOURCES}
)

if(CMAKE_CUDA_COMPILER AND USE_CUDA)
    set_target_properties(Immolate PROPERTIES
        CUDA_ARCHITECTURES "70;75;80;86;89"  # Support RTX 2000-4000 series
        CUDA_RUNTIME_LIBRARY Static  # IMPORTANT: Static link CUDA runtime
    )
    
    # Static link CUDA runtime so users don't need CUDA installed
    target_link_libraries(Immolate
        CUDA::cudart_static
        CUDA::cuda_driver
    )
endif()

# Ensure it's named Immolate.dll
set_target_properties(Immolate PROPERTIES
    OUTPUT_NAME "Immolate"
    PREFIX ""
)
```

### 3. Hybrid Build Script

```bash
#!/bin/bash
# build_hybrid.sh - Build with optional GPU support

echo "Checking for CUDA..."
if command -v nvcc &> /dev/null; then
    echo "CUDA found, building with GPU support"
    GPU_FLAG="-DUSE_CUDA=ON"
else
    echo "CUDA not found, building CPU-only"
    GPU_FLAG="-DUSE_CUDA=OFF"
fi

# For Windows cross-compile from WSL2
mkdir -p build
cd build

# Use MinGW for CPU parts, NVCC for CUDA parts
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../toolchain-mingw.cmake \
    $GPU_FLAG \
    -DCMAKE_BUILD_TYPE=Release

make -j8
cp Immolate.dll ../../

echo "Built Immolate.dll ($(du -h ../../Immolate.dll | cut -f1))"
```

## Deployment Scenarios

### Scenario 1: User Has Compatible GPU
- DLL detects GPU at runtime
- Automatically uses GPU acceleration
- 100-1000x performance boost
- No user configuration needed

### Scenario 2: User Has No/Old GPU
- DLL detects no CUDA support
- Automatically falls back to CPU code
- Works exactly like current enhanced DLL
- No error messages or crashes

### Scenario 3: Mixed Hardware
```lua
-- The DLL can report its capabilities
ffi.cdef[[
    const char* get_hardware_info();
]]

local info = ffi.string(immolate.get_hardware_info())
-- Returns: "GPU: RTX 4090 (16384 cores)" or "CPU: No GPU acceleration"
```

## File Size Considerations

### Current Sizes
- **CPU-only DLL**: ~2.4 MB (current enhanced)
- **GPU-enabled DLL**: ~8-15 MB (with static CUDA runtime)
- **Dynamic CUDA**: ~3-4 MB (requires CUDA installed)

### Optimization Options

1. **Two DLL Strategy**
```lua
-- In Brainstorm.lua
local function load_best_dll()
    -- Try GPU version first
    local gpu_path = Brainstorm.PATH .. "/Immolate_GPU.dll"
    if file_exists(gpu_path) then
        local ok, dll = pcall(ffi.load, gpu_path)
        if ok then return dll, "GPU" end
    end
    
    -- Fall back to CPU version
    return ffi.load(Brainstorm.PATH .. "/Immolate.dll"), "CPU"
end

local immolate, accel_type = load_best_dll()
print(string.format("[Brainstorm] Using %s acceleration", accel_type))
```

2. **Download on Demand**
```lua
-- Check if user wants GPU acceleration
if Brainstorm.config.enable_gpu and not file_exists("Immolate_GPU.dll") then
    show_notification("GPU acceleration available! Download enhanced DLL?")
    -- Download from GitHub releases if user agrees
end
```

## Distribution Options

### Option A: Single Universal DLL
**Pros:**
- Drop-in replacement
- No user configuration
- Automatic optimization

**Cons:**
- Larger file size (8-15 MB)
- Includes code user might not use

### Option B: Separate DLLs
```
Brainstorm/
├── Immolate.dll          # CPU version (2.4 MB)
├── Immolate_GPU.dll      # GPU version (15 MB)
└── Core/Brainstorm.lua   # Auto-detects and loads best option
```

**Pros:**
- Smaller download for CPU users
- Can ship GPU version separately

**Cons:**
- More complex deployment
- User confusion potential

### Option C: Modular System
```
Brainstorm/
├── Immolate.dll           # Core DLL (1 MB)
├── backends/
│   ├── cpu_backend.dll   # CPU implementation (2 MB)
│   └── gpu_backend.dll   # GPU implementation (12 MB)
```

## Runtime Requirements

### For GPU Acceleration
- **NVIDIA GPU**: GTX 1050 or newer (Compute 6.0+)
- **Best Performance**: RTX 2000 series or newer
- **No CUDA Toolkit needed** if using static linking
- **Windows 10/11** (for WDDM 2.0+ driver)

### For CPU Fallback
- **Same as current**: Windows 7+ with MSVC runtime
- **No GPU required**
- **No additional dependencies**

## Integration Example

```lua
-- In Brainstorm.lua
function Brainstorm.init_dll()
    local ffi = require("ffi")
    
    -- Standard FFI definitions
    ffi.cdef[[
        const char* brainstorm(const char* seed, ...);
        void free_result(const char* result);
        const char* get_hardware_info();
        int get_acceleration_type(); // 0=CPU, 1=GPU
    ]]
    
    -- Load DLL
    local dll = ffi.load(Brainstorm.PATH .. "/Immolate.dll")
    
    -- Check acceleration type
    local accel = dll.get_acceleration_type()
    if accel == 1 then
        Brainstorm.debug.gpu_enabled = true
        local info = ffi.string(dll.get_hardware_info())
        print("[Brainstorm] " .. info)
        
        -- GPU can handle more seeds per frame
        Brainstorm.SEEDS_PER_FRAME_LIMIT = 1000  -- vs 10 for CPU
    else
        Brainstorm.debug.gpu_enabled = false
        print("[Brainstorm] Using CPU acceleration")
    end
    
    return dll
end
```

## Performance Metrics Display

```lua
-- Show GPU performance in debug mode
if Brainstorm.debug.gpu_enabled then
    debug_text = string.format(
        "GPU: %.0f seeds/sec (%.1fx faster)",
        seeds_per_second,
        seeds_per_second / cpu_baseline_speed
    )
end
```

## Advantages of DLL Approach

1. **Seamless Integration** - No changes to existing Lua code
2. **Automatic Optimization** - Uses best available hardware
3. **Backward Compatible** - Works on all systems
4. **Single File Distribution** - Easy to install/update
5. **No User Configuration** - It just works

## Development Workflow

1. **Build both versions**:
   ```bash
   ./build_cpu.sh     # Creates Immolate_CPU.dll
   ./build_gpu.sh     # Creates Immolate_GPU.dll
   ./merge_dlls.sh    # Creates unified Immolate.dll
   ```

2. **Test fallback**:
   ```lua
   -- Force CPU mode for testing
   Brainstorm.config.force_cpu = true
   ```

3. **Profile performance**:
   ```lua
   -- DLL can export performance counters
   local stats = dll.get_performance_stats()
   -- "GPU: 15,234,567 seeds/sec, 98% occupancy"
   ```

## Conclusion

Yes, you can absolutely ship GPU acceleration as a DLL! The best approach is:

1. **Single unified DLL** with runtime detection
2. **Static link CUDA runtime** (no user dependencies)
3. **Automatic CPU fallback** for compatibility
4. **Same API** as current DLL (no Lua changes needed)

File size increases from 2.4 MB to ~10-15 MB, but users get:
- 100-1000x speedup on compatible hardware
- Zero configuration required
- Full backward compatibility

The GPU acceleration becomes completely transparent to the end user - it just makes their searches incredibly fast when they have the hardware for it!