# CLAUDE.md - Development Guide for Claude Code

This guide provides essential context for Claude Code when working with the Brainstorm mod codebase.

## Project Overview

Brainstorm is a GPU-accelerated seed finder mod for Balatro. It uses:
- **Lua** for game integration (LuaJIT FFI)
- **C++ DLL** for high-performance seed filtering  
- **CUDA** for GPU acceleration via Driver API (PTX JIT compilation)

Current version: **v1.0.0-GA** (General Availability)  
Build size: **3.2M**  
SHA-256: `664b8e3c35c13bfe3cc6f1b171762daca3fe27a69521d3eb5de7429c994065b9`

## Architecture

```
Core/Brainstorm.lua  ←→  Immolate.dll  ←→  CUDA Kernel (PTX)
     (FFI calls)        (Driver API)      (GPU execution)
        ↓                    ↓                  ↓
   Pool Updates       Filter Params      Seed Generation
```

### Key Components

1. **Core/Brainstorm.lua** - Main mod logic
   - Hooks into Balatro's game loop
   - Manages save states and auto-reroll
   - Calls DLL via FFI with correct parameter order
   - Updates pools from game state

2. **Core/BrainstormPoolUpdate.lua** - Pool management
   - Captures exact pool state from game
   - Uses actual ctx_keys from game (not hardcoded)
   - Handles first-shop Buffoon forcing

3. **ImmolateCPP/** - Native DLL source
   - `src/brainstorm_driver.cpp` - Main DLL entry point
   - `src/gpu/gpu_kernel_driver_prod.cpp` - Production CUDA Driver API
   - `src/gpu/seed_filter_kernel_balatro.cu` - GPU kernel with exact Balatro RNG
   - `src/pool_manager.cpp` - Dynamic pool management with v2 resolver
   - `src/cpu_fallback_balatro.cpp` - CPU fallback implementation

4. **UI/ui.lua** - Settings interface
   - Integrates with Balatro's options menu
   - Manages filter configuration
   - Target MS adjustment

## Critical Implementation Details

### Parameter Passing (FFI → DLL)

```lua
-- Lua FFI signature (EXACT ORDER MATTERS)
immolate.brainstorm(seed, voucher, pack, tag1, tag2, souls, observatory, perkeo)
```

```cpp
// C++ DLL signature (MUST match order exactly)
const char* brainstorm(
    const char* seed,        // 8 chars A-Z or empty for resume
    const char* voucher,     // Exact key or empty string ""
    const char* pack,        // Exact key or empty string ""
    const char* tag1,        // Exact key or empty string ""
    const char* tag2,        // Exact key or empty string ""
    double souls,            // 0.0 or positive
    bool observatory,        // true/false
    bool perkeo             // true/false
);
```

### FilterParams Structure (40 bytes)

```cpp
struct FilterParams {
    uint32_t tag1_small;     // Tag1 index in tag_small pool
    uint32_t tag1_big;       // Tag1 index in tag_big pool
    uint32_t tag2_small;     // Tag2 index in tag_small pool
    uint32_t tag2_big;       // Tag2 index in tag_big pool
    uint32_t voucher;        // Voucher index in voucher pool
    uint32_t pack1;          // Pack index in pack1 pool
    uint32_t pack2;          // Pack index in pack2 pool  
    uint32_t require_souls;
    uint32_t require_observatory;
    uint32_t require_perkeo;
};
static_assert(sizeof(FilterParams) == 40, "FilterParams size mismatch");
```

### Dynamic Pool Resolution (v2)

```cpp
// Runtime index resolution - no hardcoded offsets!
extern "C" bool brainstorm_resolve_filter_indices_v2(
    const char* voucher_key,
    const char* pack_key,
    const char* tag1_key,
    const char* tag2_key,
    uint32_t* out_voucher_idx,
    uint32_t* out_pack1_idx,
    uint32_t* out_pack2_idx,
    uint32_t* out_tag1_small_idx,
    uint32_t* out_tag1_big_idx,
    uint32_t* out_tag2_small_idx,
    uint32_t* out_tag2_big_idx
);
```

### Filter Semantics

**Pack Filtering**: OR semantics across slots
- Searching for "Spectral Pack" finds seeds where it appears in EITHER slot
- Different packs per slot: match if either slot satisfies

**Tag Filtering**: Per-context indices  
- Tags can have different indices in tag_small vs tag_big pools
- Either placement qualifies for a match

### Balatro RNG Constants

```cpp
namespace Balatro {
    constexpr double PI    = 3.14159265358979323846;
    constexpr double HM    = 1.1239285023;        // HASH_MULT
    constexpr double LCG_A = 2.134453429141;
    constexpr double LCG_B = 1.72431234;
}
```

**Critical Discovery**: pseudoseed returns `state/2.0`, not `(state + hashed)/2.0`

## Building

### Prerequisites
- WSL2/Linux with MinGW-w64 (`x86_64-w64-mingw32-g++`)
- CUDA Toolkit (`/usr/local/cuda/bin/nvcc`)
- GCC 13 for CUDA host compiler

### Build Command

```bash
cd ImmolateCPP
./build.sh    # Production build with precise FP
```

### Build Flags (CRITICAL)
```bash
# CUDA flags (precise math)
-Xptxas -fmad=false
-prec-div=true
-prec-sqrt=true
# NO -use_fast_math!

# Host compiler flags
-fno-fast-math
-ffp-contract=off
```

## Testing

### Validation Tests
```bash
# Pack OR semantics test
g++ -o test_pack_semantics test_pack_semantics.cpp -I. -std=c++17
./test_pack_semantics

# Tag per-context test  
g++ -o test_tag_contexts test_tag_contexts.cpp -I. -std=c++17
./test_tag_contexts

# Determinism self-test
g++ -o test_determinism test_determinism.cpp -I. -std=c++17
./test_determinism

# CPU-GPU differential (needs pools.json and seeds.txt)
g++ -o differential_runner differential_runner.cpp -I. -std=c++17
./differential_runner test_data/test_pools.json test_data/test_seeds.txt
```

### TDR Soak Test (Windows)
```bash
x86_64-w64-mingw32-g++ -o test_tdr_soak.exe test_tdr_soak.cpp -I. -std=c++17
# Run on Windows: test_tdr_soak.exe
```

## Deployment

### Verification
```bash
# Calculate checksum
sha256sum Immolate.dll
# Expected: 664b8e3c35c13bfe3cc6f1b171762daca3fe27a69521d3eb5de7429c994065b9

# Deploy to game
cp Immolate.dll /mnt/c/Users/Krish/AppData/Roaming/Balatro/Mods/Brainstorm/
```

### Manual Validation Gates

#### 1. In-Game Parity Test
1. Launch Balatro with the mod
2. For 20-50 seeds:
   - Call `Brainstorm.update_pools_from_game()`
   - Run GPU search with filters
   - Open shop and verify exact matches
3. Use diagnostics: `brainstorm_diag("TESTTEST")`

#### 2. TDR Soak Test  
1. Run `test_tdr_soak.exe` on Windows
2. Let it run for 5 minutes
3. Verify 0 timeouts, stable throughput

## Debugging

### Logs
- `gpu_driver.log` - DLL operations and parameters
- `pool_update.log` - Pool management operations
- Enable debug mode for 17-digit precision logging

### Diagnostic Functions
```cpp
// Get detailed seed analysis
const char* brainstorm_diag(const char* seed);
// Returns JSON with r-values, indices, pool_id

// Generate repro bundle for bug reports  
bool brainstorm_save_repro(const char* seed);
// Creates repro_bundle/ with pools.json, seeds.txt, logs
```

### Common Issues

**Parameters Not Working**
1. Check `gpu_driver.log` shows all 8 parameters
2. Verify DLL checksum matches expected
3. **MUST restart game after DLL changes** (Windows caches DLLs)

**Mismatches**
1. Run `differential_runner` with exact pool snapshot
2. Use `brainstorm_diag()` to compare r-values
3. Check pool versions match

## Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Throughput | >5M/s | 10.23M/s |
| Kernel Time | <250ms | 245-248ms |
| Memory | <100MB | 32MB |
| TDR Safety | >80% margin | 87.5% |

## Infrastructure Features

### Correctness Safeguards
- **Determinism self-test**: Validates FP math on init
- **Shadow verification**: Every 50th call validates with CPU
- **Pool IDs**: SHA-256 hash for reproducibility
- **CPU fallback**: Automatic on any CUDA error

### Production Features  
- **Resume support**: Empty seed continues from last position
- **Target MS control**: Adjustable kernel time [50-1000ms]
- **Rolling logs**: 5MB × 2 with rotation
- **Repro bundles**: One-click bug report packages

## File Structure

```
Brainstorm/
├── Core/                          # Lua mod core
│   ├── Brainstorm.lua            # Main mod logic
│   └── BrainstormPoolUpdate.lua  # Pool capture from game
├── UI/                           
│   └── ui.lua                    # Settings interface
├── ImmolateCPP/                  # C++ source
│   ├── src/                      
│   │   ├── brainstorm_driver.cpp # DLL entry point
│   │   ├── pool_manager.cpp     # Dynamic pools with v2 resolver
│   │   ├── cpu_fallback_balatro.cpp # CPU implementation
│   │   ├── balatro_rng.hpp      # Exact RNG implementation
│   │   ├── determinism_test.hpp # FP drift detection
│   │   ├── diagnostics.hpp      # In-game parity helpers
│   │   └── pool_hash.hpp        # SHA-256 for pool IDs
│   ├── src/gpu/                 
│   │   ├── gpu_kernel_driver_prod.cpp    # CUDA Driver API
│   │   ├── seed_filter_kernel_balatro.cu # GPU kernel
│   │   ├── gpu_types.h          # Shared structures (40-byte FilterParams)
│   │   └── pool_types.h         # Dynamic pool structures
│   ├── test_pack_semantics.cpp  # Pack OR validation
│   ├── test_tag_contexts.cpp    # Tag per-context validation
│   ├── test_determinism.cpp     # FP consistency check
│   ├── differential_runner.cpp  # CPU-GPU comparison tool
│   └── build.sh                 # Build script with precise FP
├── test_data/                    # Test fixtures
│   ├── test_pools.json          
│   └── test_seeds.txt           
├── Immolate.dll                 # Compiled DLL (3.2M)
├── config.lua                   # User settings
├── checksums.txt                # SHA-256 verification
└── *.md                         # Documentation

Key Reports:
- GA_FINAL_SIGNOFF.md       # Complete validation checklist
- FINAL_GA_REPORT_FOR_DE.md  # Technical summary
- CHANGELOG.md               # Version history
```

## Critical Reminders

1. **ALWAYS verify DLL checksum after building**
2. **ALWAYS restart game completely after DLL changes**
3. **NEVER use -use_fast_math flag**
4. **ALWAYS test with differential_runner before deploying**
5. **Check FilterParams is 40 bytes (not 32)**
6. **Use v2 resolver, not hardcoded enum offsets**
7. **Pack filtering uses OR semantics**
8. **Tags have per-context indices**

## Known Limitations

- Long double warnings on GPU (hardware limitation, cosmetic)
- Windows-only (DLL is PE format)
- Requires NVIDIA GPU with Compute 6.0+
- Single GPU support (multi-GPU in v2.0)
- JSON pools must use flat arrays

---

**Last Updated**: August 30, 2025 - v1.0.0-GA Release
**All correctness issues resolved, comprehensive validation infrastructure in place**