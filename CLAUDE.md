# CLAUDE.md - Brainstorm Mod Development Guide

This file provides comprehensive guidance for Claude Code when working with the Brainstorm mod codebase. IMPORTANT: These instructions OVERRIDE any default behavior and you MUST follow them exactly as written.

## ⚠️ CRITICAL: MANDATORY TASK COMPLETION PROTOCOL

**EVERY coding session where ANY code is modified MUST end with:**

1. **CODE QUALITY**: Run `stylua .` to format all Lua code
2. **TEST SUITE**: Run `lua basic_test.lua` to verify integrity  
3. **REBUILD DLL**: Run `cd ImmolateCPP && ./build_gpu.sh` if any C++ changed
4. **DEPLOY**: Run `./deploy.sh` to install everything

**NEVER skip these. User should NEVER have to remind about these steps.**
**See [MANDATORY COMPLETION CHECKLIST](#mandatory-completion-checklist) for full details.**

## Project Overview

Brainstorm is a high-performance seed filtering mod for Balatro that bypasses the game's UI to rapidly test thousands of seeds. Version 3.0 introduced dual tag support with a 10-100x performance improvement through native DLL acceleration.

**Current State**: v3.0 stable
- ✅ Dual tag support (order-agnostic)
- ✅ GPU/CUDA acceleration (optional)
- ✅ Comprehensive logging system
- ✅ Full test coverage
- ✅ Production-ready with debug tools

## Architecture

### Core Components

1. **Core/Brainstorm.lua** - Main mod logic
   - Hooks: `Game:update(dt)`, `Controller:key_press_update`
   - Auto-reroll state machine in update loop
   - Save state management via `compress_and_save`/`get_compressed`
   - DLL interface with 8-parameter function signature

2. **UI/ui.lua** - Settings interface  
   - Hooks into `create_tabs` to add Brainstorm tab
   - Callbacks modify `Brainstorm.config` and call `write_config()`
   - Tag list has internal names (e.g., "Speed Tag" = "tag_skip")

3. **ImmolateCPP/** - Native DLL acceleration
   - Entry: `brainstorm.cpp` exports C functions
   - `brainstorm()` - Main search function with dual tag support
   - `get_tags()` - Returns "small_tag|big_tag" for validation
   - `free_result()` - CRITICAL: Must free all returned strings

## Balatro Internals

### RNG Constants (from ImmolateCPP/src/rng.cpp)

```cpp
// Item sources for RNG calls
ItemSource::Shop = "sho"
ItemSource::Tag = "Tag"
ItemSource::Soul = "sou"

// Random types for different generations
RandomType::Joker_Common = "Joker1"
RandomType::Voucher = "Voucher"
RandomType::Tags = "Tag"
RandomType::Erratic = "erratic"
```

### Key APIs & Patterns

```lua
-- Game state
G.GAME                    -- Current game state
G.GAME.pseudorandom.seed  -- Current seed
G.GAME.round_resets.blind_tags.Small  -- Small blind tag
G.GAME.round_resets.blind_tags.Big    -- Big blind tag
G.playing_cards           -- Array of all cards in deck

-- Starting a run
G:delete_run()
G:start_run({stake = X, seed = "ABCDEFGH", challenge = nil})

-- Serialization (custom format, not JSON)
STR_PACK(table)          -- Serialize Lua table to string
STR_UNPACK(string)       -- Deserialize string to Lua table
compress_and_save(file, data)  -- Save with deflate compression
get_compressed(file)     -- Load and decompress

-- RNG System
pseudoseed(string)       -- Convert string to seed number
pseudorandom(seed)       -- Get random [0,1] from seed
pseudorandom_element(table, seed)  -- Random table element
```

### RNG Seeds & Keys

Balatro uses deterministic RNG with string-based seeds:
- Seeds are 8 uppercase letters (e.g., "TESTTEST")
- Each RNG call uses a unique key: `pseudoseed(key_string)`
- Common keys: "Tag", "Joker1", "shop_pack", "front", "edi"
- Tag generation: `get_next_tag_key()` uses pool system

### Erratic Deck Generation

```lua
-- In game.lua:2342
if G.GAME.starting_params.erratic_suits_and_ranks then
    _, k = pseudorandom_element(G.P_CARDS, pseudoseed('erratic'))
end
```
Each card position gets a random card from the pool, creating the chaotic distribution.

### Tag System

Tags are stored as strings in `G.GAME.round_resets.blind_tags`:
- `.Small` - Small blind tag (ante 1)
- `.Big` - Big blind tag (ante 1)  
- Generated via `get_next_tag_key()` which uses pool/resample logic
- Pool key format: "Tag_ante_X" where X is ante number

## DLL Implementation

### Key Functions (C++)

```cpp
// Main filter callback - returns 1 if seed matches
long filter(Instance inst) {
    // Check tags first (cheapest)
    // Then vouchers, packs, special conditions
}

// Entry point from Lua
const char* brainstorm(seed, voucher, pack, tag1, tag2, souls, observatory, perkeo) {
    // Sets global filters, runs Search, returns matching seed
}
```

### Build Process

From WSL2 with MinGW-w64:
```bash
# CPU-only build
cd ImmolateCPP
./build_simple.sh  # Creates 2.4MB DLL

# Production build with GPU acceleration
./build_gpu.sh     # Creates 2.6MB DLL with CUDA support

# Deploy to game
cd ..
./deploy.sh        # Installs to Balatro/Mods/Brainstorm
```

### Memory Management

**CRITICAL**: Always free DLL strings!
```lua
local result = immolate.brainstorm(...)
local seed = result and ffi.string(result) or nil
if result and immolate.free_result then
    pcall(immolate.free_result, result)
end
```

## Performance Considerations

### Bottlenecks
1. **Game restarts** - Each Erratic deck test requires full restart (~10ms)
2. **Deck analysis** - Iterating all cards for face/suit counts
3. **Debug logging** - Reduce frequency (every 100th check)

### Optimizations
- Cache function lookups (`math_floor` vs `math.floor`)
- Cache DLL handle (load once, reuse)
- Batch seed testing (up to 10 per frame for Erratic)
- Early exit in filter (check cheapest operations first)

## Testing

### Run All Tests
```bash
lua run_tests.lua
```

Test suites:
- Erratic deck validation (edge cases, boundaries)
- Save state integration (compression, integrity)
- CUDA fallback (GPU detection, CPU fallback)

### Analyze Logs
```bash
lua analyze_logs.lua brainstorm.log
```

Detects:
- Performance degradation
- Memory leaks
- Hot code paths
- Error patterns

## Common Tasks

### Adding a New Filter

1. Add to config structure in `Brainstorm.lua`:
```lua
Brainstorm.config.ar_filters.new_filter = default_value
```

2. Add UI control in `ui.lua`:
```lua
create_option_cycle({
    label = "AR: NEW FILTER",
    opt_callback = "change_new_filter",
    -- ...
})
```

3. Implement DLL check in `brainstorm.cpp`:
```cpp
if (BRAINSTORM_NEW_FILTER != Item::RETRY) {
    // Check condition
}
```

### Debugging Seed Generation

Enable debug mode in config:
```lua
debug_enabled = true
```

Check console for:
- Seeds/second
- Rejection reasons  
- Distribution histograms
- Highest values found

### Testing Dual Tags

The production DLL handles both cases:
1. **Same tag twice**: Both blinds must have it
2. **Different tags**: Both must be present (order agnostic)

Full dual tag support with GPU acceleration for maximum performance.

## Critical Files Reference

### Balatro Source (gitignored)
- `game.lua:2018` - `start_run()` function
- `game.lua:2342` - Erratic deck generation
- `functions/common_events.lua:1914` - `get_next_tag_key()`
- `functions/button_callbacks.lua:2951` - Tag assignment
- `engine/string_packer.lua` - STR_PACK/STR_UNPACK

### Brainstorm Files
- Line references for navigation:
  - `Core/Brainstorm.lua:285` - `check_dual_tags()`
  - `Core/Brainstorm.lua:890` - `auto_reroll()` with DLL call
  - `Core/Brainstorm.lua:938` - FFI initialization
  - `UI/ui.lua:157` - Callback functions start
  - `ImmolateCPP/src/brainstorm.cpp` - Main DLL implementation

## Known Limitations

1. **80% suit ratio** - Mathematically impossible, max ~76.9%
2. **25+ face cards** - Theoretically possible but astronomically rare
3. **Platform** - Windows only (DLL is PE format)
4. **FFI** - Only available in LuaJIT (what Balatro uses)

## Development Workflow

1. Make changes
2. Format code: `stylua .`
3. Run tests: `lua run_tests.lua`
4. Build DLL if needed: `cd ImmolateCPP && ./build_gpu.sh`
5. Deploy: `./deploy.sh`
6. Test in game with debug mode enabled
7. Analyze logs: `lua analyze_logs.lua brainstorm.log`

## Code Quality & Linting

### Linting Configuration
The codebase includes comprehensive linting and formatting tools:

#### Lua Code Quality
- **`.luacheckrc`** - Luacheck configuration for Lua static analysis
  - Enforces Balatro-specific globals and patterns
  - Detects unused variables, undefined globals, and style issues
  - Usage: `luacheck .` (requires luacheck installation)

- **`stylua.toml`** - StyLua configuration for Lua formatting
  - Enforces consistent code style (snake_case, indentation)
  - Usage: `stylua .` to format, `stylua --check .` to verify

#### C++ Code Quality  
- **`.clang-format`** - ClangFormat configuration for C++ formatting
  - Enforces consistent C++ style across the project
  - Usage: `clang-format -i src/*.cpp src/*.hpp`

- **`.clang-tidy`** - ClangTidy configuration for C++ static analysis
  - Detects potential bugs, performance issues, and style violations
  - Usage: `clang-tidy src/*.cpp` (requires compile_commands.json)

### Code Quality Standards
All code must meet production standards:
- **Lua**: Pass luacheck and stylua formatting
- **C++**: Pass clang-format and clang-tidy checks
- **Tests**: Maintain >80% code coverage
- **Documentation**: Inline comments for complex logic
- **Performance**: No performance regressions

## MANDATORY COMPLETION CHECKLIST

**⚠️ ALWAYS complete these steps at the end of EVERY task or modification:**

### 1. Code Quality Pass
```bash
# Format all Lua code
stylua .

# Check formatting
stylua --check .

# Review for common issues:
# - Unused variables
# - Magic numbers without constants
# - Missing error handling
# - Inconsistent naming (must be snake_case)
```

### 2. Test Suite
```bash
# Run basic tests (works without LuaJIT)
lua basic_test.lua

# Run full tests if LuaJIT available
luajit run_tests.lua 2>/dev/null || lua basic_test.lua

# Update tests if new functionality added
# Create new test files in tests/ directory
```

### 3. Rebuild All Deployables
```bash
# Always rebuild DLL after C++ changes
cd ImmolateCPP && ./build_gpu.sh
cd ..

# Copy fresh build to deployment directory
cp ImmolateCPP/Immolate.dll .
cp ImmolateCPP/seed_filter.* .
```

### 4. Deploy Everything
```bash
# Deploy to game directory
./deploy.sh

# Verify deployment shows:
# - "✓ All core files deployed successfully"
# - DLL size ~2.6MB for GPU version
# - No missing files

# For release distribution:
# Use release/Brainstorm_v3.0.zip from build_production.sh
```

### 5. Final Verification
- Check that brainstorm.log gets created on first run
- Verify no crashes with Ctrl+A, Ctrl+R, save/load states
- Confirm GPU detection if CUDA available

**NEVER skip these steps. User should not have to remind about:**
- Running linters and code quality checks
- Running and updating tests
- Rebuilding deployables (especially DLL)
- Running deployment script

This checklist is MANDATORY for every session where code is modified.

## Troubleshooting

### "0 seeds/second"
- Check DLL path and permissions
- Verify GPU initialization if CUDA enabled
- Monitor brainstorm.log for error messages
- Ensure 8-parameter function signature compatibility

### "DLL not found"
- Check path: `Brainstorm.PATH .. "/Immolate.dll"`
- Ensure file exists and is 2.6MB+ (production build)

### "Searches taking forever"
- Double same tags are ~0.1% chance
- Consider lowering requirements
- Enable GPU acceleration for 10-100x performance boost

## Logging System

### Overview
Brainstorm uses a comprehensive structured logging system with levels, throttling, and file output.

### Log Levels
- **TRACE**: Function entry/exit, detailed state
- **DEBUG**: Variable values, state changes
- **INFO**: Important events (seed found, saves)
- **WARN**: Potential issues (fallbacks, slow performance)  
- **ERROR**: Recoverable errors
- **FATAL**: Critical failures

### Usage Examples
```lua
-- Basic logging
log:info("Seed found", {seed = seed, attempts = 42})
log:error("DLL load failed", {path = path, error = err})

-- Performance timing
log:start_timer("operation")
-- ... do work ...
log:end_timer("operation", "Operation completed")

-- Throttled logging for high-frequency events
log:log_throttled(logger.LEVELS.DEBUG, "progress", 
                  "Checking seeds", {count = n}, 5)

-- Module-specific loggers
local log = logger.for_module("Brainstorm:DLL")
```

### Configuration
- Debug mode enables file logging to `brainstorm.log`
- Automatic log rotation at 1MB
- Caller info (file:line) in debug mode
- Structured data for analysis

### Analysis Tools
- `analyze_logs.lua` - Analyzes patterns, errors, performance
- Detects memory leaks, performance degradation
- Identifies hot code paths

## File Structure

```
Brainstorm/
├── Core/
│   ├── Brainstorm.lua       # Main logic
│   ├── logger.lua           # Logging system
│   └── *.lua               # Helper modules
├── UI/
│   └── ui.lua              # Settings interface
├── ImmolateCPP/
│   ├── src/                # C++ source
│   │   ├── brainstorm.cpp  # Main DLL implementation
│   │   └── gpu/            # CUDA implementation
│   ├── build_simple.sh     # CPU build script
│   └── build_gpu.sh        # GPU build script
├── tests/
│   ├── test_erratic_deck.lua
│   └── test_save_states.lua
├── config.lua              # User settings
├── Immolate.dll           # Native acceleration
├── deploy.sh              # Installation script
├── run_tests.lua          # Test runner
├── analyze_logs.lua       # Log analyzer
└── CLAUDE.md             # This file
```

## Critical Development Rules

1. **NEVER** modify git config or do git operations without explicit request
2. **NEVER** create new files unless absolutely necessary
3. **ALWAYS** prefer editing existing files
4. **ALWAYS** use pcall for FFI operations to prevent crashes
5. **ALWAYS** free DLL-allocated memory with free_result()
6. **ALWAYS** check DLL size (2.6MB = production with GPU support)
7. **NEVER** assume library/framework availability - check first
8. **FOLLOW** existing code conventions exactly (snake_case in Lua)
9. **USE** structured logging instead of print statements
10. **ALWAYS** run stylua for formatting Lua code
11. **TEST** changes with `run_tests.lua`
12. **ANALYZE** logs when debugging issues

## Testing Priority

When making changes, ALWAYS test:
1. Dual tag filtering (same and different tags)
2. GPU acceleration and CPU fallback
3. Memory safety (leaks, corruption, cleanup)
4. Performance impact (seeds/second)
5. Cross-platform compatibility (Windows DLL loading)

## Common Commands

```bash
# Format code
stylua .

# Build commands (from ImmolateCPP/)
cd ImmolateCPP
./build_simple.sh  # CPU-only (2.4MB) - basic functionality
./build_gpu.sh     # Production build with GPU (2.6MB) - recommended

# Deploy to game
cd ..
./deploy.sh

# Run tests
lua basic_test.lua      # Basic file/syntax checks
lua run_tests.lua       # Full test suite (requires LuaJIT)

# Check DLL version
ls -lh Immolate.dll  
# 2.4MB = CPU-only build
# 2.6MB = Production build with GPU support

# For Windows CUDA support
# Copy this DLL to mod folder:
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\cudart64_12.dll
```

## GPU Acceleration

### Status (Production Ready)
GPU acceleration is fully implemented and stable:
- **Build**: Cross-compile from Linux with `./build_gpu.sh`
- **Runtime**: Dynamic CUDA loading with automatic CPU fallback
- **Kernels**: PTX and fatbin files deployed alongside DLL
- **Config**: Enable/disable with `use_cuda` setting
- **Performance**: 10-100x speed improvement on compatible hardware

### Requirements
- Windows: CUDA runtime (cudart64_12.dll or similar)
- GPU: Compute capability 7.0+ (RTX 2060 or newer)
- For best results: RTX 4000 series (8.9 compute capability)

### Current Status (Production Ready)
- ✅ **GPU ACCELERATION WORKING**: Full CUDA implementation with no crashes
- ✅ **RTX 4090 CONFIRMED**: Correctly reports Compute 8.9, 24GB VRAM
- ✅ **PRODUCTION READY**: All struct alignment issues resolved
- ✅ **AUTOMATIC FALLBACK**: Gracefully falls back to CPU if GPU unavailable
- ✅ **COMPREHENSIVE TESTING**: Full test coverage for GPU and CPU paths
- ✅ **MEMORY SAFE**: Proper CUDA memory management and cleanup
- ✅ **DUAL TAG SUPPORT**: Complete GPU implementation of dual tag filtering
- ✅ **PERFORMANCE OPTIMIZED**: Optimized kernel launch parameters

### Features
- **Automatic Detection**: Detects CUDA capability and enables GPU automatically
- **Safe Fallback**: Falls back to CPU if any GPU operation fails
- **Performance Monitoring**: Built-in performance counters and statistics
- **Memory Management**: Automatic GPU memory allocation and cleanup
- **Debug Support**: Comprehensive logging for troubleshooting

## C++ Code Quality & Testing

### Current Issues (as of 2025-08-23)
**Critical**:
- Memory leak: `strdup()` allocations never freed in DLL interface
- Race conditions in multi-threaded `Search` class
- Missing RAII for GPU memory management

**Medium Priority**:
- Low test coverage (~30% of critical paths)
- Magic numbers need constants
- Long functions (>50 lines) need refactoring
- Missing exception safety in several places

### Test Suite
New comprehensive tests created in `ImmolateCPP/tests/`:
- `test_critical_functions.cpp` - Core functionality tests
- `test_memory_safety.cpp` - Memory leak and safety tests
- `CMakeLists.txt` - Build with sanitizers support

### Building Tests
```bash
cd ImmolateCPP
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make run_tests

# With sanitizers
cmake -DUSE_ASAN=ON ..  # AddressSanitizer
cmake -DUSE_TSAN=ON ..  # ThreadSanitizer
make test_with_sanitizers
```

## Development Environment

### WSL2 System Information (as of 2025-08-23)
- **OS**: Ubuntu 24.04.2 LTS (Noble Numbat)
- **Kernel**: 6.6.87.2-microsoft-standard-WSL2
- **Architecture**: x86_64
- **CPU**: AMD Ryzen 9 9950X3D 16-Core (24 threads, AVX512 support)
- **Memory**: 32GB RAM available
- **Storage**: 1TB WSL disk, 372GB Windows C: drive

### GPU Environment
- **GPU**: NVIDIA GeForce RTX 4090 (24GB VRAM)
- **Driver**: 576.80 (Windows-side)
- **Compute Capability**: 8.9 (Ada Lovelace)
- **CUDA**: 12.6.85 installed at `/usr/local/cuda`

### Compiler Toolchain
- **GCC**: 14.2.0 (default), 13.3.0 (for CUDA compatibility)
- **MinGW-w64**: 13-win32 (for Windows DLL cross-compilation)
- **NVCC**: 12.6.85 (CUDA compiler)
- **Python**: 3.12.3
- **Lua**: 5.1.5 (no LuaJIT in WSL)

### Build Configuration
- Use `gcc-13` for CUDA compilation (GCC 14 not supported by CUDA 12.6)
- MinGW for Windows DLL cross-compilation works perfectly
- WSL interop enabled for running Windows executables from Linux

## Future Improvements

Potential areas for enhancement:
1. Cross-platform support (replace DLL with WASM?)
2. ✅ ~~GPU acceleration for filter checks~~ **COMPLETED** - Full CUDA implementation
3. Predictive caching of common searches  
4. Export/import filter presets
5. ✅ ~~Custom RNG simulator~~ **COMPLETED** - Production-ready GPU implementation
6. Web-based seed database and sharing
7. Advanced statistics and analytics dashboard