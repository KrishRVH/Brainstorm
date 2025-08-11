# CLAUDE.md

This file provides guidance to Claude Code when working with the Brainstorm mod codebase.

## Project Overview

Brainstorm is a Lua-based mod for Balatro that provides advanced seed rerolling, filtering capabilities, and save state management. It uses the Lovely mod loader and includes a native C/C++ DLL component for performance-critical operations.

## Current State (v2.2.0)

### Completed Work
- ✅ Full snake_case naming convention throughout
- ✅ Comprehensive debug system with statistics tracking
- ✅ Save state system (5 slots) fully functional
- ✅ Erratic deck validation with realistic limits
- ✅ Performance optimization for different seed rates
- ✅ Deep merge config loading for backward compatibility
- ✅ Error handling improved (no more crashes, graceful degradation)
- ✅ UI limits set to realistic values (75% suit ratio, 23 face cards max)

### Key Findings from Testing
Based on analysis of 5,790+ seeds:
- **Maximum achievable suit ratio**: ~75% (76.9% was highest found)
- **80% suit ratio**: Appears to be mathematically impossible
- **Maximum face cards found**: 23 (25 is theoretically possible but extremely rare)
- **Performance**: ~110 seeds/second with Erratic deck requirements

## Architecture

### Core Components

1. **Core/Brainstorm.lua** (Main Module)
   - Hooks into `Game:update()` and `Controller:key_press_update`
   - Manages auto-reroll state machine
   - Implements save/load state functionality
   - Validates Erratic deck requirements
   - Debug statistics and reporting

2. **UI/ui.lua** (Settings Interface)
   - Creates Brainstorm tab in game settings
   - Manages all filter options and preferences
   - Callbacks update global config

3. **Immolate.dll** (Native Component)
   - High-performance seed filtering
   - Tests vouchers, tags, and packs
   - Does NOT understand Erratic deck generation
   - Windows-only, MSVC-compiled

4. **config.lua** (Persistent Settings)
   - Auto-saved using STR_PACK/STR_UNPACK
   - Deep merged on load for compatibility

### Key Systems

#### Auto-Reroll Logic
```lua
-- For Erratic decks with requirements:
-- 1. Generate/get seed
-- 2. Restart game with seed
-- 3. Analyze deck
-- 4. Check requirements
-- 5. Continue or stop

-- Performance limited to prevent lag:
-- - 250-1000 seeds/sec: Full speed
-- - 1000-5000 seeds/sec: Capped at 10 seeds/frame
```

#### Debug System
- Tracks seeds tested, rejection reasons, distributions
- Periodic updates every 5 seconds
- Final report with recommendations
- Enabled via `debug_enabled` in config

#### Save State System
- Uses `compress_and_save` and `get_compressed` 
- Stores in profile directory as `save_state_[1-5].jkr`
- Full game state including deck, money, jokers, etc.

## Commands

### Development
```bash
# Format code
stylua .

# Check formatting
stylua --check .
```

### Testing
Test with different configurations:
1. Erratic deck + face cards only
2. Erratic deck + suit ratio only  
3. Normal deck + voucher/tag filters
4. Combinations (expect long search times)

## Known Limitations

1. **Performance Bottleneck**: Each seed test requires full game restart
2. **DLL Limitations**: Doesn't understand Erratic deck generation
3. **Platform**: Windows-only due to native DLL
4. **Impossible Combinations**: 80%+ suit ratio doesn't exist

## Next Steps

### Short Term
1. Monitor user feedback on realistic limits
2. Consider adding "quick presets" for common searches
3. Add export/import for save states

### Long Term (Custom DLL)
Create a new native library (Rust/Zig) that:

1. **Simulates full deck generation** including Erratic
2. **Tests 100,000+ seeds/second** without game restarts
3. **Batch processing** for efficiency
4. **Cross-platform** support

Implementation approach:
```rust
// Pseudo-code for new DLL
fn test_seeds(start: u64, count: u32, filters: Filters) -> Vec<SeedResult> {
    (0..count).parallel_map(|i| {
        let seed = start + i;
        let deck = generate_deck(seed, filters.deck_type);
        let shop = generate_shop(seed);
        
        SeedResult {
            seed,
            matches: validate_all(deck, shop, filters),
            face_cards: deck.face_count(),
            suit_ratio: deck.suit_ratio(),
        }
    }).filter(|r| r.matches).collect()
}
```

### Research Needed
1. Reverse engineer Balatro's exact PRNG algorithm
2. Map Erratic deck generation logic
3. Understand shop/voucher generation

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
│   └── Brainstorm.lua      # Main logic
├── UI/
│   └── ui.lua              # Settings interface
├── config.lua              # User settings
├── Immolate.dll           # Native component
├── lovely.toml            # Mod loader config
├── nativefs.lua           # File I/O module
├── steamodded_compat.lua  # Compatibility header
├── stylua.toml            # Code formatter config
├── README.md              # User documentation
├── CLAUDE.md              # This file
└── IMMOLATE_SPEC.md       # DLL replacement spec
```

## Testing Checklist
- [ ] Manual reroll (Ctrl+R)
- [ ] Auto-reroll toggle (Ctrl+A)
- [ ] Save states (Z+1-5)
- [ ] Load states (X+1-5)
- [ ] Face card filtering
- [ ] Suit ratio filtering
- [ ] Voucher/tag/pack filtering
- [ ] Debug reporting
- [ ] Config persistence

## Support

For issues or improvements, consider:
1. Debug logs with `debug_enabled = true`
2. Realistic filter combinations
3. Performance vs. accuracy tradeoffs
4. User experience over technical perfection