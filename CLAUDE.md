# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Brainstorm is a Lua-based mod for Balatro that provides advanced seed rerolling and filtering capabilities. It uses the Lovely mod loader and includes a native C/C++ DLL component for performance-critical operations.

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
   - Handles keybinds (Ctrl+R for manual, Ctrl+A for auto)
   - Persists configuration to `config.lua`

2. **UI/ui.lua** - Configuration UI that:
   - Defines the Brainstorm tab in game settings
   - Creates checkboxes and sliders for all filter options
   - Updates the global config table

3. **Immolate.dll** - Native component accessed via FFI:
   - High-performance seed generation
   - Called through `ffi.cdef` declarations in Core/Brainstorm.lua
   - Critical for achieving 500-1000 seeds/frame performance

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

### Key Patterns

- **Function Hooking**: Override Balatro functions to inject mod behavior
- **Config Persistence**: Auto-saves to `config.lua` using nativefs
- **FFI Integration**: Uses LuaJIT FFI for native DLL calls
- **Cached References**: Performance optimization through cached function refs (e.g., `math_abs`, `string_format`)

### Erratic Deck Analysis

The mod performs sophisticated deck analysis for Erratic edition:
- Counts face cards and suits
- Calculates distribution ratios
- Validates against user-configured thresholds
- Only stops auto-reroll when all conditions are met

## Development Notes

- This is a runtime mod - no build step required
- Test changes by running Balatro with the mod installed
- Place mod in `%AppData%/Balatro/Mods/Brainstorm/`
- Config changes persist automatically
- Visual feedback appears as "attention text" in-game