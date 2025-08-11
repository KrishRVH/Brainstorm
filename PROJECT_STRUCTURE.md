# Project Structure

## Core Brainstorm Mod
```
Core/
├── Brainstorm.lua          # Main mod logic
UI/
├── ui.lua                  # Settings interface
config.lua                  # User settings
Immolate.dll               # Original GPU-accelerated searcher
```

## Enhanced Immolate (NEW)
```
ImmolateSourceCode/
├── filters/
│   ├── erratic_brainstorm.cl    # NEW: Erratic deck filter
│   └── [other filters]
├── lib/
│   ├── erratic_support.cl       # NEW: Erratic helpers
│   └── [core libraries]
├── immolate.c                   # Main executable
└── CMakeLists.txt              # Build configuration
```

## Build & Test Scripts
```
build_immolate_enhanced.bat     # Builds GPU version with Erratic support
test_immolate_erratic.bat       # Tests the enhancement
```

## Analysis Tools
```
balatro_rng_analyzer.c          # Demonstrates RNG flaws
```

## Documentation
```
README.md                       # Main documentation
SETUP.md                        # Quick setup guide
CLAUDE.md                       # AI assistant instructions
PROJECT_STRUCTURE.md           # This file
```

## Balatro Source (Reference)
```
BalatroSource/
├── Balatro.exe                # Game executable (extractable)
├── extracted/                 # Extracted Lua source
└── [DLL files]               # Game dependencies
```

## Key Files to Edit

### To Change Erratic Filter Settings
`ImmolateSourceCode/filters/erratic_brainstorm.cl`

### To Modify Brainstorm Behavior  
`Core/Brainstorm.lua`

### To Update UI Options
`UI/ui.lua`

## Workflow

1. **GPU Filtering**: Immolate searches millions of seeds
2. **Mod Integration**: Brainstorm receives promising seeds
3. **Game Validation**: Final check in actual game

This structure separates concerns while maintaining compatibility.