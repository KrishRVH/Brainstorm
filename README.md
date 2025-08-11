# Immolate Enhanced - Erratic Deck Support for Brainstorm

## Overview

This repository contains enhancements to the original Immolate OpenCL seed searcher to add full Erratic deck support for the Brainstorm mod.

## What's Included

### Core Components

1. **Original Immolate** (`ImmolateSourceCode/`)
   - GPU-accelerated OpenCL seed searcher
   - Handles vouchers, tags, and pack filtering
   - 100,000+ seeds/second performance

2. **Erratic Deck Enhancements** 
   - `filters/erratic_brainstorm.cl` - New filter for Erratic deck requirements
   - `lib/erratic_support.cl` - Helper functions for deck analysis
   - Full support for face card counting and suit ratio filtering

3. **Analysis Tools**
   - `balatro_rng_analyzer.c` - Demonstrates RNG flaws (glitched seeds)
   - `README_GLITCHED_SEEDS.md` - Documentation of RNG issues

## Quick Start

### Using Enhanced Immolate

1. **Build the enhanced version**:
   ```cmd
   build_immolate_enhanced.bat
   ```

2. **Search for Erratic seeds with specific requirements**:
   ```cmd
   cd ImmolateSourceCode
   
   # Find seeds with 20+ face cards and 50%+ of one suit
   Immolate.exe -f erratic_brainstorm -n 100000 -c 2000
   
   # Test the infamous glitched seed
   Immolate.exe -f erratic_brainstorm -s 7LB2WVPK -n 1
   ```

3. **Use with Brainstorm**:
   - Keep original `Immolate.dll` for voucher/tag filtering
   - Use enhanced Immolate for Erratic deck pre-filtering
   - Brainstorm validates final results in-game

## Filter Parameters

Edit `filters/erratic_brainstorm.cl` to adjust:

```c
const int MIN_FACE_CARDS = 20;      // Minimum face cards
const int MAX_FACE_CARDS = 52;      // Maximum (52 = no limit)
const float MIN_SUIT_RATIO = 0.5;   // Minimum suit ratio (50%)
const int TARGET_SUIT = -1;         // -1=any, 0=C, 1=D, 2=H, 3=S
```

## Performance

| Method | Seeds/Second | Use Case |
|--------|--------------|----------|
| Game Restarts | ~110 | Final validation |
| CPU Implementation | ~10,000 | Learning/testing |
| **GPU Immolate** | **100,000+** | **Production searching** |

## How It Works

The enhancement replicates Balatro's Erratic deck generation:

1. Each card position randomly selects from all 52 cards
2. Uses the same RNG constants: `1.72431234` and `2.134453429141`
3. Maintains state through the node caching system
4. Accurately reproduces glitched seeds

## Known Issues

### The RNG Flaw

Balatro's RNG has a self-feeding issue identified by the community:
- Lacks entropy (no external randomness)
- Can produce statistically impossible seeds
- Example: Seed `7LB2WVPK` generates 52 copies of 10 of Spades

Our implementation **accurately replicates this flaw** for compatibility.

## Testing

Run the test suite:
```cmd
test_immolate_erratic.bat
```

This will:
1. Test known glitched seeds
2. Search for high face card counts
3. Verify suit ratio filtering

## Integration with Brainstorm

The enhanced Immolate works alongside Brainstorm:

1. **Pre-filtering**: Immolate quickly eliminates bad seeds
2. **Final validation**: Brainstorm verifies in-game
3. **Best of both**: GPU speed + game accuracy

## Credits

- **Original Immolate**: SpectralPack team
- **RNG Analysis**: jimbo_extreme1 (Reddit)
- **Brainstorm Mod**: Original authors
- **Enhancements**: This implementation

## License

Enhancements are provided as-is for the Balatro modding community.