# Brainstorm v3.0 - Lightning-Fast Seed Filtering for Balatro

A high-performance mod that finds perfect seeds in seconds instead of hours. Features dual tag searching, Erratic deck filtering, and save states.

## Quick Start

```bash
# Deploy to Balatro
./deploy.sh

# Build DLL from source (optional)
cd ImmolateCPP && ./build_simple.sh
```

## In-Game Usage

| Key | Action |
|-----|--------|
| `Ctrl+T` | Open settings |
| `Ctrl+R` | Manual reroll |
| `Ctrl+A` | Toggle auto-reroll |
| `Z+1-5` | Save state |
| `X+1-5` | Load state |

## Features

- **Dual Tag Search** - Find seeds with ANY two tag combination (e.g., double Investment)
- **Erratic Deck Filters** - Face cards (0-23) and suit ratio (up to 75%)
- **Smart Performance** - 100-1000 seeds/second with automatic throttling
- **Save States** - 5 slots for experimentation

## Development

### Project Structure
```
Brainstorm/
├── Core/Brainstorm.lua    # Main logic & auto-reroll
├── UI/ui.lua              # Settings interface
├── ImmolateCPP/           # C++ DLL source
│   ├── src/               # Enhanced dual tag implementation
│   └── build_simple.sh    # MinGW build script
├── config.lua             # User settings (auto-saved)
├── deploy.sh              # Installation script
└── Immolate.dll           # Native acceleration (2.4MB = enhanced)
```

### Building the DLL

Requires MinGW-w64 (works from WSL2):
```bash
cd ImmolateCPP
./build_simple.sh  # Creates Immolate.dll
```

### Key Concepts

**Dual Tag Validation** - The enhanced DLL checks both blind positions internally for 10-100x speedup over the original approach of restarting the game for each seed.

**Erratic Deck Limits** - Testing shows 75% suit ratio is maximum achievable, 23 face cards is realistic max (25 theoretically possible but extremely rare).

**Performance Throttling** - Automatically limits seeds per frame based on requirements to maintain 60 FPS.

## Configuration

Settings saved to `config.lua` using Balatro's STR_PACK format:

- `ar_filters` - Tag, voucher, pack requirements
- `ar_prefs` - Speed, face cards, suit ratio
- `debug_enabled` - Performance statistics

## Troubleshooting

**"Mod not loading"** - Check Lovely injector is installed  
**"Searches are slow"** - Ensure you have the 2.4MB enhanced DLL  
**"Can't find good seeds"** - Double same tags are extremely rare (~0.1% chance)

## License

MIT