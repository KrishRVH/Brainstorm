# CLAUDE.md - Brainstorm Mod Development Guide

This file provides comprehensive guidance for Claude Code when working with the Brainstorm mod codebase. IMPORTANT: These instructions OVERRIDE any default behavior and you MUST follow them exactly as written.

## Project Overview

Brainstorm is a high-performance seed filtering mod for Balatro that bypasses the game's UI to rapidly test thousands of seeds. Version 3.0 introduced dual tag support with a 10-100x performance improvement through native DLL acceleration.

**Current State**: v3.0 stable with dual tag support, all features working, optimized performance.

## Architecture

### Core Components

1. **Core/Brainstorm.lua** - Main mod logic
   - Hooks: `Game:update(dt)`, `Controller:key_press_update`
   - Auto-reroll state machine in update loop
   - Save state management via `compress_and_save`/`get_compressed`
   - DLL compatibility layer (8 params for enhanced, 7 for original)

2. **UI/ui.lua** - Settings interface  
   - Hooks into `create_tabs` to add Brainstorm tab
   - Callbacks modify `Brainstorm.config` and call `write_config()`
   - Tag list has internal names (e.g., "Speed Tag" = "tag_skip")

3. **ImmolateCPP/** - Native DLL acceleration
   - Entry: `brainstorm_enhanced.cpp` exports C functions
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
cd ImmolateCPP
./build_simple.sh  # Uses x86_64-w64-mingw32-g++
# Output: ../Immolate.dll (2.4MB = enhanced, 106KB = original)
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

3. Implement DLL check in `brainstorm_enhanced.cpp`:
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

The enhanced DLL handles both cases:
1. **Same tag twice**: Both blinds must have it
2. **Different tags**: Both must be present (order agnostic)

Original DLL only checks tag1, requiring game restarts for tag2.

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
  - `ImmolateCPP/src/brainstorm_enhanced.cpp:36` - `filter()` function

## Known Limitations

1. **80% suit ratio** - Mathematically impossible, max ~76.9%
2. **25+ face cards** - Theoretically possible but astronomically rare
3. **Platform** - Windows only (DLL is PE format)
4. **FFI** - Only available in LuaJIT (what Balatro uses)

## Development Workflow

1. Make Lua changes
2. Format with `stylua .`
3. If DLL changes: `cd ImmolateCPP && ./build_simple.sh`
4. Deploy: `./deploy.sh`
5. Test in game with F2 console

## Troubleshooting

### "0 seeds/second"
- Parameter mismatch between Lua and DLL
- Check enhanced (8 params) vs original (7 params)

### "DLL not found"
- Check path: `Brainstorm.PATH .. "/Immolate.dll"`
- Ensure file exists and is 2.4MB (enhanced)

### "Searches taking forever"
- Double same tags are ~0.1% chance
- Consider lowering requirements
- Check if using enhanced DLL (10-100x faster)

## Critical Development Rules

1. **NEVER** modify git config or do git operations without explicit request
2. **NEVER** create new files unless absolutely necessary
3. **ALWAYS** prefer editing existing files
4. **ALWAYS** use pcall for FFI operations to prevent crashes
5. **ALWAYS** free DLL-allocated memory with free_result()
6. **ALWAYS** check DLL size (2.4MB = enhanced with dual tags)
7. **NEVER** assume library/framework availability - check first
8. **FOLLOW** existing code conventions exactly (snake_case in Lua)
9. **NEVER** add comments unless explicitly requested
10. **ALWAYS** run stylua for formatting Lua code

## Testing Priority

When making changes, ALWAYS test:
1. Dual tag filtering (same and different tags)
2. DLL compatibility fallback (7 vs 8 parameters)
3. Memory leaks (monitor with debug mode)
4. Performance impact (seeds/second)

## Common Commands

```bash
# Format code
stylua .

# Build DLL from WSL2
cd ImmolateCPP && ./build_simple.sh

# Deploy to game
./deploy.sh

# Check DLL version
ls -lh Immolate.dll  # Should be ~2.4MB for enhanced
```

## Future Improvements

Potential areas for enhancement:
1. Cross-platform support (replace DLL with WASM?)
2. GPU acceleration for filter checks  
3. Predictive caching of common searches
4. Export/import filter presets
5. Custom RNG simulator for 100,000+ seeds/second