# Brainstorm - Advanced Seed Searcher for Balatro

## Overview

Brainstorm is a comprehensive mod for Balatro that provides:
- **In-game auto-reroll** with customizable filters (vouchers, tags, packs, Erratic deck requirements)
- **Save state system** with 5 slots for experimentation
- **GPU-accelerated seed searching** via enhanced Immolate (100,000+ seeds/second)
- **Full Erratic deck support** including face card and suit ratio filtering

## Quick Start

### 1. Install the Mod

Copy to `%AppData%\Balatro\Mods\Brainstorm\`:
- `Core\` folder
- `UI\` folder  
- `config.lua`
- `lovely.toml`
- `nativefs.lua`
- `steamodded_compat.lua`
- `Immolate.dll`

### 2. Use In-Game

| Key | Action |
|-----|--------|
| `Ctrl+T` | Open settings |
| `Ctrl+R` | Manual reroll |
| `Ctrl+A` | Toggle auto-reroll |
| `Z+1-5` | Save state to slot |
| `X+1-5` | Load state from slot |

### 3. (Optional) Build GPU Tool for Erratic Pre-Search

```cmd
# Build on Windows (not WSL)
build_immolate_enhanced.bat

# Search for Erratic seeds
cd ImmolateSourceCode
Immolate.exe -f erratic_brainstorm -n 1000000 -c 2000
```

## Realistic Filter Settings

### Face Cards (Erratic Deck)
- **Easy**: 10-15 face cards  
- **Medium**: 15-20 face cards
- **Hard**: 20-23 face cards
- **Nearly Impossible**: 25+ face cards (extremely rare)

### Suit Ratio (Erratic Deck)  
- **Easy**: 40-50% single suit
- **Medium**: 50-65% single suit
- **Hard**: 65-75% single suit
- **Mathematically Impossible**: 80%+ single suit

To adjust, edit `ImmolateSourceCode/filters/erratic_brainstorm.cl`

## Performance

| Hardware | Expected Speed |
|----------|---------------|
| CPU only | 1-5K seeds/sec |
| Integrated GPU | 10-50K seeds/sec |
| Gaming GPU | 100K+ seeds/sec |

## How It Works

1. **In-Game (Brainstorm Mod)**:
   - Hooks into game update loop
   - Tests seeds by restarting game
   - ~110 seeds/second maximum
   - Uses original `Immolate.dll` for voucher/tag filtering

2. **GPU Tool (Enhanced Immolate)**:
   - Replicates Balatro's exact RNG
   - Tests 100,000+ seeds/second
   - Full Erratic deck generation
   - Accurately handles glitched seeds (e.g., `7LB2WVPK`)

## Development

See [CLAUDE.md](CLAUDE.md) for:
- Architecture details
- Code style guide
- Testing procedures
- Future roadmap

## Troubleshooting

### "No OpenCL devices"
Install GPU drivers:
- NVIDIA: [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- AMD: Latest drivers with OpenCL
- Intel: Intel Graphics Driver

### "Mod not loading"
1. Check Lovely injector is installed
2. Verify files in correct folder
3. Press F2 in game for console/errors

### "Can't find good seeds"
Your criteria may be too strict:
- Lower requirements
- Run longer searches
- Check realistic settings above

## Credits

- **Original Immolate**: SpectralPack team
- **RNG Analysis**: jimbo_extreme1 (Reddit)
- **Brainstorm Mod**: Original authors
- **Enhancements**: This implementation