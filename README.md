# Brainstorm v2.2.0

Advanced seed rerolling and save state mod for Balatro, featuring high-performance native seed filtering and Erratic deck validation.

## Features

### 🎲 Advanced Seed Rerolling
- **Manual Reroll**: `Ctrl+R` - Instantly reroll to a new seed
- **Auto-Reroll**: `Ctrl+A` - Automatically search for seeds matching your criteria
- **Performance Options**: 250-5000 seeds per second
- **Smart Filtering**: Find specific vouchers, tags, and booster packs
- **Erratic Deck Support**: Validate face card counts and suit distributions

### 💾 Save State System
- **5 Save Slots**: Store complete game states
- **Quick Save**: Hold `Z` + `1-5` to save current run
- **Quick Load**: Hold `X` + `1-5` to load saved state
- **Visual Feedback**: On-screen alerts confirm save/load actions
- **Persistent Storage**: States saved as compressed `.jkr` files

### 🃏 Erratic Deck Filters
- **Face Card Requirements**: Set minimum face cards (0-23)
- **Suit Distribution**: Require top 2 suits to comprise X% of deck
- **Realistic Limits**: Based on analysis of 5,790+ seeds
  - Max achievable face cards: ~23
  - Max achievable suit ratio: ~75%

### 🔍 Filter Options
- **Tags**: Charm, Double, Uncommon, Rare, Holographic, Foil, etc.
- **Vouchers**: Overstock, Crystal Ball, Telescope, Grabber, etc.
- **Booster Packs**: Arcana, Celestial, Standard, Buffoon, Spectral
- **Instant Bonuses**: Observatory, Perkeo
- **Soul Skip**: Number of soul cards to skip

## Installation

1. Install [Lovely](https://github.com/ethangreen-dev/lovely-injector) mod loader
2. Download the latest release
3. Extract to: `%AppData%/Balatro/Mods/Brainstorm/`
4. Launch Balatro

## Usage

### Basic Controls
- `Ctrl+R`: Manual reroll
- `Ctrl+A`: Toggle auto-reroll
- `Z + 1-5`: Save state to slot
- `X + 1-5`: Load state from slot

### Configuration
Access the Brainstorm tab in game settings to configure:
- Filter criteria (tags, vouchers, packs)
- Erratic deck requirements
- Performance settings
- Debug mode

### Performance Tips
- **For Erratic Decks**: Limited to ~1000 seeds/sec to prevent lag
- **Without Erratic**: Can achieve full speed (up to 5000/sec)
- **Realistic Targets**: 
  - 15-18 face cards + 60% suit ratio = Rare but findable
  - 20+ face cards OR 70%+ suit ratio = Very rare
  - Both together = Extremely rare

## Technical Details

### Architecture
- **Core/Brainstorm.lua**: Main mod logic, hooks, and state management
- **UI/ui.lua**: Settings interface and configuration
- **Immolate.dll**: Native C++ component for high-performance seed filtering
- **config.lua**: Persistent settings storage

### Limitations
- **Windows Only**: Due to native DLL dependency
- **Erratic Deck Performance**: Each seed test requires game restart
- **Suit Ratio Maximum**: 75% is the realistic maximum (80% appears impossible)

### Debug Mode
Enable `debug_enabled = true` in config for:
- Seeds per second tracking
- Distribution analysis
- Rejection reason statistics
- Performance metrics

## Known Issues

- Searching for 80%+ suit ratio will run indefinitely (mathematically impossible)
- Combining strict Erratic requirements with voucher/tag filters may take very long
- Frame drops when testing many seeds per frame

## Development

### Requirements
- Lua 5.1+ (included with Balatro)
- Lovely mod loader
- Windows OS (for DLL)

### Code Style
- Snake_case naming throughout
- Cached function references for performance
- Comprehensive error handling with pcall
- Deep merge for config loading

### Future Improvements
- Custom Rust/Zig DLL for 100,000+ seeds/sec
- GPU acceleration for parallel seed testing
- Pre-computed seed database
- Cross-platform support

## Credits

- **Author**: OceanRamen
- **Contributors**: Enhanced debugging and Erratic deck support by community
- **Framework**: Uses Lovely mod loader and Steamodded compatibility

## License

This mod is provided as-is for the Balatro community. Feel free to modify and share.