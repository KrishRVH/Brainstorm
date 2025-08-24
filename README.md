# Brainstorm v3.0 - Lightning-Fast Seed Filtering for Balatro

A high-performance mod that finds perfect seeds in seconds instead of hours. Features dual tag searching, Erratic deck filtering, save states, and optional GPU acceleration.

## Status (2025-08-24)
- ✅ **GPU CRASH FIXED** - RTX 4090 confirmed working, no crashes on Ctrl+A
- ✅ **Struct mismatch resolved** - Correctly detects Compute 8.9, 24GB VRAM
- ✅ Dual tag support fully working
- ✅ Save states functional
- ✅ GPU initialization stable (RTX 2060+ supported)
- ⚠️ PTX kernel launch pending (currently falls back to CPU)

## Quick Start

```bash
# Deploy to Balatro
./deploy.sh

# Build DLL from source (optional)
cd ImmolateCPP
./build_simple.sh # CPU-only version (2.4MB)
./build_gpu.sh    # GPU-accelerated (2.6MB, requires CUDA)
```

### GPU Acceleration (Optional)
For RTX GPU support, you may need to copy CUDA runtime to mod folder:
```powershell
copy "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudart64_12.dll" "C:\Users\%USERNAME%\AppData\Roaming\Balatro\Mods\Brainstorm\"
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
└── Immolate.dll           # Native acceleration (2.4MB = CPU, 2.6MB = GPU)
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
**"Searches are slow"** - Ensure you have the enhanced DLL (2.4MB CPU or 2.6MB GPU)  
**"Can't find good seeds"** - Double same tags are extremely rare (~0.1% chance)  
**"Game crashes on Ctrl+A"** - Check `gpu_debug.log` in mod folder for diagnostics  
**"GPU not detected"** - Copy correct `cudart64_*.dll` to mod folder, check compute capability ≥7.0

## License

MIT