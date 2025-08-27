# Brainstorm v3.0 - GPU-Accelerated Seed Finder for Balatro

A high-performance mod that bypasses Balatro's UI to rapidly test thousands of seeds, finding those that match your exact criteria. Features CUDA GPU acceleration for 10-100x faster searching.

## Features

- **GPU Acceleration**: Utilizes NVIDIA GPUs for blazing-fast seed searching
- **Dual Tag Support**: Search for seeds with specific tag combinations
- **Save States**: Quick save/load functionality (Z+1-5 to save, X+1-5 to load)
- **Auto-Reroll**: Continuously searches for matching seeds (Ctrl+A to toggle)
- **Advanced Filters**: Vouchers, packs, souls, observatory, and more
- **Seamless Integration**: Works directly within Balatro's game loop

## Requirements

- **Balatro** (Steam version)
- **Windows 64-bit**
- **NVIDIA GPU** (optional, for GPU acceleration)
  - RTX 2060 or newer recommended
  - CUDA Compute Capability 6.0+
  - Latest NVIDIA drivers

## Installation

1. Download the latest release from GitHub Releases
2. Extract to `%AppData%/Roaming/Balatro/Mods/Brainstorm/`
3. Ensure the mod structure looks like:
   ```
   Balatro/Mods/Brainstorm/
   ├── Core/
   │   ├── Brainstorm.lua
   │   └── logger.lua
   ├── UI/
   │   └── ui.lua
   ├── Immolate.dll
   ├── config.lua
   └── lovely.toml
   ```
4. Launch Balatro - the mod loads automatically

## Usage

### In-Game Controls

| Key | Action |
|-----|--------|
| **Ctrl+T** | Open Brainstorm settings tab |
| **Ctrl+R** | Manual reroll (single seed test) |
| **Ctrl+A** | Toggle auto-reroll (continuous search) |
| **Z + 1-5** | Save current state to slot |
| **X + 1-5** | Load state from slot |

### Settings

Access settings via **Ctrl+T** or the Brainstorm tab in the options menu:

#### Filter Options
- **Tags**: Select up to 2 tags for ante 1 (order-agnostic)
- **Starting Voucher**: Filter by specific voucher
- **Shop Pack**: Filter by pack availability
- **Souls Required**: Number of soul cards needed
- **Observatory**: Telescope + Celestial pack combo
- **Perkeo**: Investment tag + Soul card

#### Performance Options
- **Debug Mode**: Enable detailed logging
- **GPU Acceleration**: Toggle CUDA acceleration (auto-detected)

## Performance

### Speed Benchmarks

| Hardware | Seeds/Second | Notes |
|----------|--------------|-------|
| **RTX 4090** | 1M+ | < 1ms per million seeds |
| **RTX 3080** | 500K+ | Excellent performance |
| **RTX 2060** | 200K+ | Good performance |
| **CPU Only** | 10-50K | Varies by processor |

### GPU Acceleration

The mod automatically detects and uses NVIDIA GPUs when available. GPU acceleration provides:
- 10-100x faster seed searching
- Near-instant results for single tag searches
- Sub-second results for dual tag combinations
- Efficient batch processing of millions of seeds

## Advanced Features

### Save States

Save states capture the complete game state including:
- Current seed and ante
- All cards and jokers
- Money and hands
- Shop contents
- All game flags

### Dual Tag Searching

Search for seeds with two specific tags in ante 1:
- **Same Tag Twice**: Both blinds must have the tag (extremely rare ~0.1%)
- **Different Tags**: Both must appear (order doesn't matter ~1-5%)

### Filter Combinations

Combine multiple filters for precise seed hunting:
- Tags + Voucher + Pack
- Observatory setup (Telescope + Celestial pack)
- Perkeo setup (Investment tag + Soul in arcana pack)

## Technical Details

### Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Lua Mod   │────▶│  Native DLL  │────▶│ CUDA Kernel  │
│ (Brainstorm)│ FFI │ (Immolate)   │     │ (PTX/Driver) │
└─────────────┘     └──────────────┘     └──────────────┘
```

### Implementation
- **Lua Layer**: Game integration, UI, state management
- **C++ DLL**: High-performance seed filtering, RNG simulation
- **CUDA Kernel**: Massively parallel GPU computation
- **Driver API**: Runtime PTX compilation for compatibility

### Build Requirements

For development/building from source:
- MinGW-w64 (cross-compilation from Linux/WSL2)
- CUDA Toolkit 12.0+ (for GPU support)
- GCC 13 (for CUDA compatibility)

## Troubleshooting

### GPU Not Detected
- Ensure latest NVIDIA drivers are installed
- Check that GPU meets minimum requirements (Compute 6.0+)
- Verify `nvcuda.dll` is present in system

### Crashes on Launch
- Verify all files are in correct locations
- Check `%AppData%/Roaming/Balatro/Mods/lovely/log` for errors
- Ensure using 64-bit Windows

### Slow Performance
- Enable GPU acceleration in settings
- Close other GPU-intensive applications
- Reduce filter complexity for faster results

### Save State Issues
- Ensure write permissions in Mods folder
- Don't load states from different Balatro versions
- States are compressed - don't edit manually

## Configuration

Settings are stored in `config.lua` and persist between sessions:

```lua
{
  debug_enabled = false,        -- Enable debug logging
  use_cuda = true,              -- Use GPU acceleration
  ar_filters = {
    enabled = false,            -- Auto-reroll state
    tag_name = "Standard Tag",  -- Primary tag filter
    tag2_name = "Standard Tag", -- Secondary tag filter
    -- ... other filters
  }
}
```

## Debug Mode

Enable debug mode for troubleshooting:
1. Set `debug_enabled = true` in settings
2. Check `brainstorm.log` for detailed information
3. Includes timing, rejection reasons, and GPU metrics

## Credits

- **Development**: Balatro modding community
- **GPU Acceleration**: CUDA Driver API implementation
- **RNG Simulation**: Reverse-engineered from Balatro v1.0.1

## License

MIT License - See LICENSE file for details

## Support

- Report issues on GitHub
- Mod compatibility: Works with most other Balatro mods
- Updates: Check releases for latest versions

---

**Note**: This mod is designed for single-player experimentation and does not modify save files permanently. Use responsibly and enjoy finding your perfect seeds!
