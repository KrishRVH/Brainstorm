# Installing Enhanced Brainstorm in Balatro

## Step 1: Build on Windows (Not WSL)

Exit WSL and open Windows Command Prompt or PowerShell:

```cmd
cd C:\path\to\Brainstorm
build_immolate_enhanced.bat
```

This creates the enhanced Immolate.exe in `ImmolateSourceCode\build\Release\`

## Step 2: Locate Your Balatro Mods Folder

The Brainstorm mod folder is typically at:
- `%AppData%\Balatro\Mods\Brainstorm\`
- Or: `C:\Users\[YourName]\AppData\Roaming\Balatro\Mods\Brainstorm\`

## Step 3: Copy Required Files

### Core Mod Files (REQUIRED):
Copy these to your `Mods\Brainstorm\` folder:

```
✅ Core\Brainstorm.lua
✅ UI\ui.lua  
✅ config.lua
✅ lovely.toml
✅ nativefs.lua
✅ steamodded_compat.lua
✅ Immolate.dll (original - keep this!)
```

### Enhanced Files (OPTIONAL but recommended):
These don't go in the mods folder - they're external tools:

```
📁 Create a folder anywhere (e.g., C:\BalatroTools\)
✅ Copy ImmolateSourceCode\build\Release\Immolate.exe
✅ Copy the entire ImmolateSourceCode\filters\ folder
✅ Copy test_immolate_erratic.bat
```

## Step 4: Your Final Setup

### In Balatro Mods Folder:
```
Balatro\Mods\Brainstorm\
├── Core\
│   └── Brainstorm.lua
├── UI\
│   └── ui.lua
├── config.lua
├── lovely.toml
├── nativefs.lua
├── steamodded_compat.lua
└── Immolate.dll          ← Original stays here!
```

### In Your Tools Folder (separate):
```
C:\BalatroTools\
├── Immolate.exe          ← Enhanced GPU version
├── filters\
│   └── erratic_brainstorm.cl
└── test_immolate_erratic.bat
```

## Step 5: How to Use

### In-Game (Brainstorm Mod):
1. Launch Balatro
2. Press `Ctrl+T` to open Brainstorm settings
3. Configure your filters (face cards, suit ratio, etc.)
4. Press `Ctrl+A` to start auto-reroll
5. The mod uses the original `Immolate.dll` for voucher/tag filtering

### For Erratic Deck Pre-Search (Outside Game):
1. Open Command Prompt
2. Navigate to your tools folder:
   ```cmd
   cd C:\BalatroTools
   ```
3. Search for Erratic seeds:
   ```cmd
   Immolate.exe -f filters\erratic_brainstorm -n 1000000 -c 2500
   ```
4. Note down good seeds
5. Enter them manually in Balatro's seed input

## Quick Test

To verify everything works:

### Test the Mod:
1. Launch Balatro
2. Press `Ctrl+T` - should open Brainstorm settings
3. Press `Ctrl+R` - should reroll seed

### Test Enhanced Immolate:
```cmd
cd C:\BalatroTools
Immolate.exe -f filters\erratic_brainstorm -s 7LB2WVPK -n 1
```
Should show the glitched seed with 52 copies of 10 of Spades

## Important Notes

⚠️ **DO NOT** replace the original `Immolate.dll` in the mods folder!
- The original handles in-game voucher/tag filtering
- The enhanced `Immolate.exe` is a separate tool for pre-searching

⚠️ **The enhanced Immolate.exe is NOT integrated with the mod**
- It's a standalone tool for finding seeds
- You manually enter found seeds in the game

## Workflow

1. **Want specific Erratic deck?** → Run enhanced Immolate.exe externally
2. **Found good seed?** → Copy seed string
3. **In Balatro** → Go to new run, paste seed
4. **Want voucher/tag filtering?** → Use Brainstorm mod normally (Ctrl+A)

## Troubleshooting

### "Immolate.exe not found"
- Make sure you ran `build_immolate_enhanced.bat` on Windows (not WSL)
- Check `ImmolateSourceCode\build\Release\` for the exe

### "Filter not found"
- Make sure you copied the `filters` folder
- Use relative path: `-f filters\erratic_brainstorm`

### "No OpenCL devices"
- Install your GPU drivers
- For NVIDIA: Install CUDA toolkit
- For AMD: Install AMD drivers with OpenCL

### Mod not loading
- Check lovely.toml is in the right place
- Verify you have Lovely injector installed
- Check Balatro console for errors (F2)