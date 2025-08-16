# CLAUDE.md

This file provides guidance to Claude Code when working with the Brainstorm mod codebase.

## Project Overview

Brainstorm is a Lua-based mod for Balatro that provides advanced seed rerolling, filtering capabilities, and save state management. It uses the Lovely mod loader and includes a native C/C++ DLL component for performance-critical operations.

## Current State (v2.3.0 - Enhanced DLL Edition)

### Completed Work
- ✅ Full snake_case naming convention throughout
- ✅ Comprehensive debug system with statistics tracking
- ✅ Save state system (5 slots) fully functional
- ✅ Erratic deck validation with realistic limits
- ✅ Performance optimization for different seed rates
- ✅ Deep merge config loading for backward compatibility
- ✅ Error handling improved (no more crashes, graceful degradation)
- ✅ UI limits set to realistic values (75% suit ratio, 23 face cards max)
- ✅ **NEW: Dual tag search UI** (TAG 1 and TAG 2 dropdowns)
- ✅ **NEW: Enhanced DLL with direct dual tag validation** (10-100x faster)
- ✅ **NEW: Order-agnostic tag matching** (finds tags in either blind position)
- ✅ **NEW: Built from source using MinGW in WSL2**

### Key Findings from Testing
Based on analysis of 5,790+ seeds:
- **Maximum achievable suit ratio**: ~75% (76.9% was highest found)
- **80% suit ratio**: Appears to be mathematically impossible
- **Maximum face cards found**: 23 (25 is theoretically possible but extremely rare)
- **Performance with original DLL**: ~110 seeds/second with Erratic deck
- **Performance with enhanced DLL**: ~1000+ seeds/second for dual tags (no restarts needed!)

## Architecture

### Core Components

1. **Core/Brainstorm.lua** (Main Module)
   - Hooks into `Game:update()` and `Controller:key_press_update`
   - Manages auto-reroll state machine
   - Implements save/load state functionality
   - Validates Erratic deck requirements
   - Debug statistics and reporting
   - Dual tag validation for Erratic decks

2. **UI/ui.lua** (Settings Interface)
   - Creates Brainstorm tab in game settings
   - Manages all filter options and preferences
   - **NEW: TAG 2 SEARCH dropdown for dual tag filtering**
   - Callbacks update global config

3. **Immolate.dll** (Native Component - Two Versions Available)
   
   **Original (106KB):**
   - High-performance seed filtering
   - Tests vouchers, tags, and packs
   - Can only check if ONE tag exists
   - Requires game restart to validate tag positions
   
   **Enhanced (2.4MB) - NEW:**
   - Direct dual tag validation (checks both Small and Big blinds)
   - Exports: `brainstorm()`, `get_tags()`, `free_result()`
   - Order-agnostic matching
   - 10-100x faster for dual tag searches
   - Built with MinGW-w64 from WSL2

4. **config.lua** (Persistent Settings)
   - Auto-saved using STR_PACK/STR_UNPACK
   - Deep merged on load for compatibility
   - Stores `tag2_name` and `tag2_id` for dual tags

### Key Systems

#### Auto-Reroll Logic (Enhanced)
```lua
-- For dual tags with enhanced DLL:
-- 1. DLL directly validates BOTH tags
-- 2. Returns only seeds that match
-- 3. No game restart needed!

-- For Erratic decks (still requires restarts):
-- 1. Generate/get seed
-- 2. Restart game with seed
-- 3. Analyze deck
-- 4. Check requirements including dual tags
-- 5. Continue or stop
```

#### Enhanced DLL Functions
```cpp
// New brainstorm function with dual tag support
const char* brainstorm(
    const char* seed,
    const char* voucher,
    const char* pack,
    const char* tag1,      // First tag to search for
    const char* tag2,      // Second tag (can be empty or same as tag1)
    double souls,
    bool observatory,
    bool perkeo
);

// Get both blind tags for any seed
const char* get_tags(const char* seed);
// Returns: "small_tag|big_tag"

// Memory management
void free_result(const char* result);
```

#### Debug System
- Tracks seeds tested, rejection reasons, distributions
- Periodic updates every 5 seconds
- **NEW: Tracks dual tag checks and successes**
- Final report with recommendations
- Enabled via `debug_enabled` in config

## Building the Enhanced DLL

### From WSL2
```bash
# Install MinGW if not present
sudo apt-get install mingw-w64

# Navigate to source
cd ImmolateCPP

# Build
./build_simple.sh

# Result: ../Immolate_new.dll
```

### From Windows (Visual Studio)
```cmd
build_brainstorm_dll.bat
```

## Deployment

```bash
# Auto-deploys enhanced DLL if present
./deploy.sh
```

The deploy script now:
- Checks for `Immolate_new.dll` and uses it if available
- Backs up original DLL to `Immolate_original.dll`
- Copies all mod files to Balatro mods folder

## Known Limitations

1. **Erratic Deck**: Enhanced DLL doesn't simulate Erratic deck generation (still requires restarts)
2. **Platform**: Windows-only DLL (but buildable from WSL2 with MinGW)
3. **Impossible Combinations**: 80%+ suit ratio doesn't exist mathematically

## Performance Comparison

| Search Type | Original DLL | Enhanced DLL | Improvement |
|------------|--------------|--------------|-------------|
| Single Tag | ~0.01s | ~0.01s | Same |
| Dual Different Tags | 2-10s | 0.1-1s | **10-100x** |
| Dual Same Tag (e.g., 2x Investment) | 5-30s | 0.5-3s | **10-100x** |
| Tag + Voucher + Pack | 3-15s | 0.2-2s | **10-50x** |
| Erratic + Dual Tags | 10-60s | 10-60s | Same (bottleneck is deck generation) |

## Next Steps

### Short Term
1. ~~Monitor user feedback on realistic limits~~ ✅ Done
2. ~~Add dual tag search capability~~ ✅ Done
3. Add export/import for save states
4. Consider adding "quick presets" for common searches

### Medium Term
1. Add Erratic deck simulation to DLL (eliminate ALL restarts)
2. Multi-threaded seed searching
3. GPU acceleration via OpenCL (like original Immolate)

### Research Needed
1. Reverse engineer Balatro's exact Erratic deck generation
2. Map complete RNG sequence for all game elements
3. Optimize search algorithms for rare combinations

## Code Style Guide

- **Naming**: snake_case for all functions and variables
- **Error Handling**: Use pcall for FFI and critical operations
- **Performance**: Cache function references, minimize allocations
- **Config**: Always use deep merge when loading
- **Debug**: Use Brainstorm.debug structure for statistics
- **Comments**: Minimal, code should be self-documenting

## File Structure
```
Brainstorm/
├── Core/
│   ├── Brainstorm.lua           # Main logic with dual tag support
│   └── Brainstorm_enhanced.lua  # Enhanced DLL integration examples
├── UI/
│   └── ui.lua                   # Settings interface with TAG 2 dropdown
├── ImmolateCPP/                 # C++ source for DLL
│   ├── src/
│   │   ├── brainstorm_enhanced.cpp  # Our enhanced implementation
│   │   ├── functions.hpp/cpp
│   │   ├── items.hpp/cpp
│   │   ├── rng.hpp/cpp
│   │   ├── search.hpp
│   │   ├── seed.hpp/cpp
│   │   └── util.hpp/cpp
│   ├── build_simple.sh          # WSL2 build script
│   └── CMakeLists_Brainstorm.txt # CMake config for DLL
├── config.lua                   # User settings (includes tag2)
├── Immolate.dll                # Original DLL (106KB)
├── Immolate_new.dll            # Enhanced DLL (2.4MB) when built
├── deploy.sh                   # Smart deployment script
├── lovely.toml                 # Mod loader config
├── nativefs.lua                # File I/O module
├── steamodded_compat.lua       # Compatibility header
├── stylua.toml                 # Code formatter config
├── README.md                   # User documentation
├── CLAUDE.md                   # This file
├── BUILD_INSTRUCTIONS.md       # How to build the enhanced DLL
└── INSTALLATION.md             # How to install and use
```

## Testing Checklist
- [x] Manual reroll (Ctrl+R)
- [x] Auto-reroll toggle (Ctrl+A)
- [x] Save states (Z+1-5)
- [x] Load states (X+1-5)
- [x] Face card filtering
- [x] Suit ratio filtering
- [x] Single tag filtering
- [x] **Dual tag filtering (same tag)**
- [x] **Dual tag filtering (different tags)**
- [x] Voucher/pack filtering
- [x] Observatory combo (Telescope + Celestial)
- [x] Perkeo combo (Investment + Soul)
- [x] Debug reporting
- [x] Config persistence

## Support

For issues or improvements:
1. Check debug logs with `debug_enabled = true`
2. For dual tag searches taking too long, ensure using enhanced DLL (2.4MB)
3. Remember that Erratic deck requirements still require game restarts
4. Report build issues with MinGW version and error messages