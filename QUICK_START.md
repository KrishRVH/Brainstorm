# Brainstorm Quick Start Guide

## 🚀 Fast Setup (Windows)

### 1. Download & Extract
```
Download Brainstorm → Extract to any folder
```

### 2. Build GPU Tool
```cmd
scripts\build\build_windows.bat
```

### 3. Install Mod
Copy these to `%AppData%\Balatro\Mods\Brainstorm\`:
- `Core\` folder
- `UI\` folder  
- `config.lua`
- `lovely.toml`
- `nativefs.lua`
- `steamodded_compat.lua`
- `Immolate.dll`

### 4. Play!
- Launch Balatro
- Press `Ctrl+T` for settings
- Press `Ctrl+A` to auto-reroll

## 🔍 Find Erratic Seeds (Optional)

### Quick Search
```cmd
cd tools
Immolate.exe -f filters\erratic_brainstorm -n 100000
```

### Test Specific Seed
```cmd
Immolate.exe -f filters\erratic_brainstorm -s 7LB2WVPK -n 1
```

## ⚡ Hotkeys

| Key | Action |
|-----|--------|
| `Ctrl+T` | Open settings |
| `Ctrl+R` | Manual reroll |
| `Ctrl+A` | Toggle auto-reroll |
| `Z+1-5` | Save state to slot |
| `X+1-5` | Load state from slot |

## 🎯 Realistic Settings

### Face Cards (Erratic Deck)
- **Easy**: 10-15 face cards
- **Medium**: 15-20 face cards  
- **Hard**: 20-23 face cards
- **Impossible**: 25+ face cards

### Suit Ratio (Erratic Deck)
- **Easy**: 40-50% single suit
- **Medium**: 50-65% single suit
- **Hard**: 65-75% single suit
- **Impossible**: 80%+ single suit

## ❓ Troubleshooting

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
Your criteria may be too strict. Try:
- Lower face card requirement
- Reduce suit ratio requirement
- Run longer searches (1M+ seeds)

## 📊 Performance

| Hardware | Expected Speed |
|----------|---------------|
| CPU only | 1-5K seeds/sec |
| Integrated GPU | 10-50K seeds/sec |
| Gaming GPU | 100K+ seeds/sec |

## 🎮 Example Workflow

1. **Want 20+ face Erratic deck?**
   ```cmd
   tools\Immolate.exe -f filters\erratic_brainstorm -n 1000000
   ```

2. **Found seed `ABC12DEF`**
   - Copy the seed
   - In Balatro: New Run → Set Seed → Paste

3. **Want specific vouchers?**
   - Use in-game Brainstorm (Ctrl+A)
   - Original Immolate.dll handles this

## 💡 Pro Tips

- Pre-search Erratic seeds with GPU tool
- Save interesting seeds in a text file
- Use save states (Z+1-5) for experimentation
- Combine filters for unique runs
- Join Discord for seed sharing

## 🔧 Advanced

See [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) for:
- Building from source
- Modifying filters
- Creating custom searches
- Performance tuning