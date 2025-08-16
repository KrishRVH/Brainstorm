# Brainstorm Workflow Guide

## How This Repository Works

This is the **Brainstorm mod** for Balatro - an advanced seed searching and save state management tool. It has two main components:

1. **The Lua mod** (Core/UI folders) - Runs inside Balatro for auto-reroll and save states
2. **GPU tool** (ImmolateSourceCode) - Optional external tool for ultra-fast Erratic deck searching

## What You Can Do

### 1. Play with the Mod (Most Common)

**Install the mod:**
- Copy these files to `%AppData%\Balatro\Mods\Brainstorm\`:
  - `Core/` folder
  - `UI/` folder
  - `config.lua`
  - `lovely.toml`
  - `nativefs.lua`
  - `steamodded_compat.lua`
  - `Immolate.dll`

**Use in-game:**
- `Ctrl+T` - Open settings (configure filters)
- `Ctrl+R` - Manual reroll
- `Ctrl+A` - Auto-reroll (searches for seeds matching your filters)
- `Z+1-5` - Save current game state
- `X+1-5` - Load saved state

### 2. Pre-Search Erratic Seeds (Advanced)

If you want specific Erratic deck configurations (face cards, suit ratios):

**Build the GPU tool (Windows only):**
```cmd
build_immolate_enhanced.bat
```

**Search for seeds:**
```cmd
cd ImmolateSourceCode
Immolate.exe -f erratic_brainstorm -n 1000000 -c 2000
```

**Use found seeds:**
- Copy the seed string
- In Balatro: New Run → Set Seed → Paste

### 3. Development

**Modify the mod:**
- Edit `Core/Brainstorm.lua` for logic changes
- Edit `UI/ui.lua` for new settings options
- Format code: `stylua .`

**Modify Erratic filter:**
- Edit `ImmolateSourceCode/filters/erratic_brainstorm.cl`
- Rebuild: `build_immolate_enhanced.bat`
- Test: `test_immolate_erratic.bat`

## Common Tasks

### "I want to find a seed with specific vouchers/tags"
1. Launch Balatro with the mod installed
2. Press `Ctrl+T` to open settings
3. Configure your desired vouchers/tags
4. Press `Ctrl+A` to start auto-search
5. Wait for a match (uses original Immolate.dll)

### "I want an Erratic deck with 20+ face cards"
1. Build the GPU tool: `build_immolate_enhanced.bat`
2. Run: `cd ImmolateSourceCode && Immolate.exe -f erratic_brainstorm -n 1000000 -c 2000`
3. Copy a good seed from the output
4. Use it in Balatro's seed input

### "I want to experiment with a specific setup"
1. Get to your desired game state
2. Press `Z+1` to save to slot 1
3. Experiment freely
4. Press `X+1` to restore your save

### "I want to modify filter requirements"
1. Edit `ImmolateSourceCode/filters/erratic_brainstorm.cl`
2. Change the constants at the top:
   ```c
   const int MIN_FACE_CARDS = 22;     // was 20
   const float MIN_SUIT_RATIO = 0.6;  // was 0.5
   ```
3. Rebuild: `build_immolate_enhanced.bat`
4. Test your changes

## Performance Expectations

- **In-game auto-reroll**: ~110 seeds/second
- **GPU tool on gaming GPU**: 100,000+ seeds/second
- **GPU tool on integrated GPU**: 10,000-50,000 seeds/second

## File Structure

```
Brainstorm/
├── Core/Brainstorm.lua        # Main mod logic
├── UI/ui.lua                  # Settings interface
├── config.lua                 # User preferences (auto-saved)
├── Immolate.dll               # Original voucher/tag filter
├── ImmolateSourceCode/        # GPU tool source
│   ├── filters/               # OpenCL filters
│   │   └── erratic_brainstorm.cl  # Erratic deck filter
│   └── immolate.c            # Main executable
├── build_immolate_enhanced.bat  # Build GPU tool
└── test_immolate_erratic.bat   # Test the filter
```

## Realistic Settings

Based on analysis of 5,790+ seeds:

**Face Cards (Erratic):**
- 10-15: Common (easy to find)
- 15-20: Uncommon (takes a bit)
- 20-23: Rare (long searches)
- 25+: Extremely rare (may not exist)

**Suit Ratio (Erratic):**
- 40-50%: Common
- 50-65%: Uncommon
- 65-75%: Rare
- 80%+: Mathematically impossible

## Troubleshooting

**"No OpenCL devices"**
- Install GPU drivers (NVIDIA CUDA, AMD OpenCL, Intel Graphics)

**"Mod not loading"**
- Check Lovely injector is installed
- Press F2 in Balatro for console errors

**"Can't find seeds"**
- Lower your requirements
- Run longer searches (millions of seeds)
- Check realistic settings above

## Important Notes

- The mod uses `Immolate.dll` for in-game voucher/tag filtering
- The GPU tool (`Immolate.exe`) is separate for pre-searching Erratic seeds
- Config auto-saves when you change settings
- Save states are stored in your Balatro profile directory
- The mod accurately replicates Balatro's RNG, including glitched seeds