# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Brainstorm is a Lua-based mod for Balatro that provides advanced seed rerolling, filtering capabilities, and save state management. It uses the Lovely mod loader and includes a native C/C++ DLL component for performance-critical operations.

## Commands

### Code Formatting
```bash
# Check formatting
stylua --check .

# Auto-format code
stylua .
```

### Git Workflow
- Pre-commit hooks automatically format code using StyLua
- Feature branches merge into `master` via pull requests

## High-Level Architecture

### Core Components

1. **Core/Brainstorm.lua** - Main mod logic that:
   - Hooks into Balatro's `Game:update()` loop
   - Manages reroll state machine
   - Handles keybinds (Ctrl+R for manual, Ctrl+A for auto, z/x for save states)
   - Persists configuration to `config.lua`
   - Validates Erratic deck requirements (face cards, suit distribution)
   - Manages save states (5 slots)

2. **UI/ui.lua** - Configuration UI that:
   - Defines the Brainstorm tab in game settings
   - Creates checkboxes and sliders for all filter options
   - Updates the global config table
   - Provides Erratic Deck settings (face cards, suit ratio)

3. **Immolate.dll** - Native component accessed via FFI:
   - High-performance seed generation
   - Called through `ffi.cdef` declarations in Core/Brainstorm.lua
   - Tests multiple seeds internally per call
   - Windows-only (x86-64 PE32+ DLL)

4. **nativefs.lua** - Filesystem operations module:
   - Provides file I/O capabilities
   - Required for config persistence
   - Patched in before main.lua by Lovely

### Mod Loading Flow

1. Lovely loader reads `lovely.toml`
2. Patches `nativefs.lua` module
3. Appends `Core/Brainstorm.lua` to main.lua
4. Injects initialization into `game.lua`
5. Steamodded compatibility headers load if needed

### Key Features

#### Save State System
- **5 Save Slots**: Store complete game states
- **Save Keybind**: Hold `z` + 1-5 to save current run
- **Load Keybind**: Hold `x` + 1-5 to load saved state
- **Visual Feedback**: Alert messages confirm save/load actions
- **File Format**: Compressed `.jkr` files in profile directory

#### Auto-Reroll System
- **Performance Settings**: 250-5000 seeds per second
- **Multi-seed Testing**: Tests multiple seeds per frame based on performance setting
- **Attention Text**: Shows "Rerolling..." after 60 frames

#### Erratic Deck Validation
- **Face Card Count**: Minimum number of face cards required (0-25)
- **Suit Ratio**: Top 2 suits must comprise X% of deck (or disabled)
- **Deck Analysis**: Counts all cards, suits, faces, aces

#### Filter System
- **Tags**: Various tags like Charm, Double, Uncommon, etc.
- **Vouchers**: Overstock, Crystal Ball, Telescope, etc.
- **Packs**: Arcana, Celestial, Standard, Buffoon, Spectral
- **Soul Skip**: Number of soul cards to skip
- **Instant Bonuses**: Observatory, Perkeo

### Important Patterns

- **Function Naming**: Uses snake_case throughout
- **Config Persistence**: Auto-saves to `config.lua` using STR_PACK/STR_UNPACK
- **FFI Integration**: Wrapped in pcall for error handling
- **Cached References**: Performance optimization through cached function refs
- **Deep Merge**: Config loading uses deep merge to preserve existing settings
- **Event System**: Uses G.E_MANAGER for delayed actions and alerts

### Known Limitations

1. **DLL Dependency**: Windows-only due to Immolate.dll
2. **Filter Combinations**: Using both DLL filters AND Erratic deck requirements can make finding seeds nearly impossible
3. **Performance Bottleneck**: Game instance creation for each seed test is expensive

## Development Notes

- This is a runtime mod - no build step required
- Test changes by running Balatro with the mod installed
- Place mod in `%AppData%/Balatro/Mods/Brainstorm/`
- Config changes persist automatically
- Visual feedback appears as "attention text" in-game
- Save states are stored as `save_state_[1-5].jkr` in profile folder

## Current Configuration Defaults

```lua
{
  keybinds = {
    modifier = "lctrl",
    f_reroll = "r",      -- Manual reroll
    a_reroll = "a",      -- Toggle auto-reroll
    save_state = "z",    -- Save state (with 1-5)
    load_state = "x",    -- Load state (with 1-5)
  },
  ar_prefs = {
    spf_int = 1000,      -- Seeds per second
    face_count = 0,      -- Min face cards for Erratic
    suit_ratio_percent = "Disabled",  -- Suit distribution requirement
    suit_ratio_decimal = 0,
  },
  ar_filters = {
    tag_name = "",       -- Empty = no tag filter
    voucher_name = "",   -- Empty = no voucher filter
    pack = {},           -- Empty = no pack filter
    soul_skip = 0,
    inst_observatory = false,
    inst_perkeo = false,
  }
}
```

## Keybind Reference

- **Ctrl+R**: Manual reroll (fast reroll)
- **Ctrl+A**: Toggle auto-reroll
- **z+1 to z+5**: Save game state to slot 1-5
- **x+1 to x+5**: Load game state from slot 1-5