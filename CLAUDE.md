# CLAUDE.md - Development Guide for Claude Code

This guide provides essential context for Claude Code when working with the Brainstorm mod codebase.

## Project Overview

Brainstorm is a GPU-accelerated seed finder mod for Balatro. It uses:
- **Lua** for game integration (LuaJIT FFI)
- **C++ DLL** for high-performance seed filtering
- **CUDA** for GPU acceleration via Driver API

Current version: **3.0** (stable, GPU acceleration working)

## Architecture

```
Core/Brainstorm.lua  ←→  Immolate.dll  ←→  CUDA Kernel (PTX)
     (FFI calls)        (Driver API)      (GPU execution)
```

### Key Components

1. **Core/Brainstorm.lua** - Main mod logic
   - Hooks into Balatro's game loop
   - Manages save states and auto-reroll
   - Calls DLL via FFI

2. **ImmolateCPP/** - Native DLL source
   - `src/brainstorm.cpp` - Main entry point
   - `src/gpu/gpu_kernel_driver.cpp` - CUDA Driver API integration
   - `src/gpu/seed_filter_kernel.cu` - GPU kernel

3. **UI/ui.lua** - Settings interface
   - Integrates with Balatro's options menu
   - Manages filter configuration

## Development Workflow

### Building

From WSL2/Linux with MinGW-w64 and CUDA Toolkit:

```bash
cd ImmolateCPP
./build_driver.sh    # Builds DLL with GPU support
cd ..
./deploy.sh         # Deploys to Balatro/Mods/Brainstorm
```

### Testing

```bash
stylua .            # Format Lua code
lua basic_test.lua  # Run basic tests
./validate.sh       # Full validation before deployment
```

### Debugging

- Enable debug mode in game settings (Ctrl+T)
- Check `brainstorm.log` for Lua-side logs
- GPU logs disabled in production for performance

## Critical Implementation Details

### RNG System
- Balatro uses deterministic string-based seeds (8 uppercase letters)
- Each RNG call uses unique keys: "Tag", "Joker1", "shop_pack", etc.
- Tags generated via pool system with resampling

### DLL Interface
```lua
-- FFI function signature (8 parameters)
immolate.brainstorm(seed, voucher, pack, tag1, tag2, souls, observatory, perkeo)
-- Returns matching seed or empty string
-- MUST call free_result() on non-empty returns
```

### GPU Context Management
- Uses CUDA Driver API (not Runtime API)
- Primary context with automatic management via ScopedCtx RAII
- PTX embedded in DLL, JIT-compiled at runtime

### Memory Safety
- Always free DLL-allocated strings with `free_result()`
- Use pcall for all FFI operations
- Proper CUDA context cleanup in destructor

## Performance Targets

- **GPU**: 1M+ seeds/second on RTX 4090
- **CPU**: 10-50K seeds/second fallback
- Debug logging disabled in production for max performance

## Code Standards

### Lua
- Use snake_case naming
- Gate all debug output behind `Brainstorm.debug.enabled`
- Format with stylua before committing

### C++
- Minimal console output (silent operation)
- RAII for resource management
- Comments for complex logic only

## Common Tasks

### Adding a New Filter
1. Add to config structure in `Brainstorm.lua`
2. Add UI control in `ui.lua`
3. Implement check in `brainstorm.cpp::filter_cpu()`
4. Add to GPU kernel if performance-critical

### Debugging GPU Issues
1. Set `FILE* debug_file = fopen(...)` in gpu_kernel_driver.cpp
2. Check initialization and context management
3. Verify PTX compilation and kernel launch
4. Monitor `gpu_driver.log` for errors

## File Structure

```
Brainstorm/
├── Core/           # Lua mod core
├── UI/             # Settings interface  
├── ImmolateCPP/    # C++ source
│   ├── src/        # Main source files
│   └── src/gpu/    # CUDA implementation
├── Immolate.dll    # Compiled DLL
├── config.lua      # User settings
└── deploy.sh       # Deployment script
```

## Deployment Checklist

Before deploying any changes:
1. Run `stylua .` to format code
2. Run `lua basic_test.lua` to verify integrity
3. Rebuild DLL if C++ changed: `cd ImmolateCPP && ./build_driver.sh`
4. Run `./deploy.sh` to install

## Known Limitations

- Windows-only (DLL is PE format)
- Requires LuaJIT (what Balatro uses)
- GPU requires NVIDIA with Compute 6.0+

## Support

Report issues at: https://github.com/anthropics/claude-code/issues

---

**Remember**: The codebase is currently stable and working. Make conservative changes and always test thoroughly before deployment.