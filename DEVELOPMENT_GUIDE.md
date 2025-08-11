# Brainstorm Development Guide

## Development Environment Setup

### Prerequisites

#### Windows
- Visual Studio 2019+ with C++ support
- GPU drivers (NVIDIA CUDA, AMD OpenCL, or Intel OpenCL)
- Git for Windows
- Python 3.8+ (for testing scripts)

#### WSL2
- Ubuntu 20.04+ on WSL2
- Build essentials: `sudo apt install build-essential cmake`
- OpenCL headers: `sudo apt install ocl-icd-opencl-dev opencl-headers`
- MinGW for cross-compilation: `sudo apt install mingw-w64`

## Project Structure

```
Brainstorm/
├── Core/                       # Mod core logic
│   └── Brainstorm.lua         # Main mod file
├── UI/                        # User interface
│   └── ui.lua                # Settings panel
├── ImmolateSourceCode/        # GPU-accelerated seed finder
│   ├── filters/              # OpenCL filters
│   │   ├── erratic_brainstorm.cl  # Erratic deck support
│   │   └── ...
│   ├── lib/                  # OpenCL libraries
│   └── immolate.c           # Main executable
├── scripts/                   # Development scripts
│   ├── dev/                  # Development tools
│   ├── test/                 # Testing scripts
│   └── build/                # Build scripts
└── docs/                      # Documentation
```

## Building

### Option 1: Build on Windows (Recommended)

```cmd
# Open Command Prompt or PowerShell
cd C:\path\to\Brainstorm
scripts\build\build_windows.bat
```

### Option 2: Build on WSL2

```bash
# Limited GPU support in WSL2
cd ~/personal/Brainstorm
./scripts/build/build_wsl.sh
```

### Option 3: Cross-compile from WSL2 for Windows

```bash
# Creates Windows binaries from Linux
cd ~/personal/Brainstorm
./scripts/build/cross_compile.sh
```

## Testing

### Quick Test Suite

```bash
# Run all tests
./scripts/test/run_all_tests.sh

# Test specific component
./scripts/test/test_rng.sh
./scripts/test/test_erratic.sh
./scripts/test/test_gpu.sh
```

### Manual Testing

#### Test RNG Implementation
```bash
# Verify glitched seed behavior
./balatro_rng_analyzer
```

#### Test GPU Acceleration
```cmd
# Windows
scripts\test\test_gpu.bat

# WSL2 (if GPU available)
./scripts/test/test_gpu.sh
```

#### Test Erratic Deck Filter
```cmd
# Find seeds with 20+ face cards
Immolate.exe -f filters\erratic_brainstorm -n 100000
```

## Development Workflow

### 1. Making Changes to Erratic Filter

Edit `ImmolateSourceCode/filters/erratic_brainstorm.cl`:

```c
// Adjust requirements
const int MIN_FACE_CARDS = 22;    // was 20
const float MIN_SUIT_RATIO = 0.6; // was 0.5
```

Rebuild and test:
```bash
./scripts/build/rebuild_filters.sh
./scripts/test/test_erratic.sh --min-face 22 --suit-ratio 0.6
```

### 2. Modifying Brainstorm Core

Edit `Core/Brainstorm.lua`:

```lua
-- Add new feature
Brainstorm.my_new_feature = function()
    -- implementation
end
```

Test in-game:
1. Copy to `%AppData%\Balatro\Mods\Brainstorm\`
2. Launch Balatro with `--console` flag
3. Press F2 for console
4. Test your feature

### 3. Adding UI Options

Edit `UI/ui.lua` to add new settings:

```lua
-- Add to create_brainstorm_tab()
{
    label = "My New Option",
    ref = "my_option",
    val = false,
    callback = function(val)
        Brainstorm.config.my_option = val
    end
}
```

## Debugging

### Enable Debug Mode

In `config.lua`:
```lua
debug_enabled = true
debug_verbose = true
```

### Console Commands

In Balatro console (F2):
```lua
-- Check Brainstorm state
print(Brainstorm.debug.seeds_tested)

-- Force reroll
Brainstorm.reroll_seed()

-- Test specific seed
G.GAME.pseudorandom.seed = "7LB2WVPK"
```

### Performance Profiling

```bash
# Profile GPU performance
./scripts/dev/profile_gpu.sh

# Analyze bottlenecks
./scripts/dev/analyze_performance.sh
```

## Common Issues

### "No OpenCL devices found"
- **Windows**: Install GPU drivers and CUDA toolkit
- **WSL2**: GPU support is limited, use Windows build

### "Immolate.dll not found"
- Ensure you're building with correct architecture (x64)
- Check DLL dependencies with `dumpbin /dependents Immolate.dll`

### "Erratic filter not working"
- Verify filter file exists in `filters/` directory
- Check OpenCL compilation errors in console

### "Mod not loading"
- Check `lovely.toml` configuration
- Verify Lovely injector is installed
- Look for errors in Balatro console (F2)

## Code Style

### Lua (Brainstorm)
- Use snake_case for functions and variables
- Minimal comments, self-documenting code
- Always use pcall for external calls

### C/OpenCL (Immolate)
- K&R style braces
- Descriptive variable names
- Comment complex algorithms

### Testing
- Test edge cases (glitched seeds)
- Verify performance targets
- Check memory usage

## Performance Targets

- **RNG Analysis**: 1000+ seeds/second (CPU)
- **GPU Filtering**: 100,000+ seeds/second
- **Erratic Validation**: 50,000+ seeds/second
- **Memory Usage**: < 100MB for mod, < 500MB for GPU tool

## Git Workflow

```bash
# Feature branch
git checkout -b feature/better-erratic-filter

# Make changes
git add -A
git commit -m "feat: improve erratic filter performance"

# Push
git push origin feature/better-erratic-filter
```

## Release Checklist

- [ ] All tests passing
- [ ] Performance targets met
- [ ] Documentation updated
- [ ] Version bumped in lovely.toml
- [ ] CHANGELOG.md updated
- [ ] Build artifacts created for Windows
- [ ] Installation guide verified

## Resources

- [Balatro Wiki](https://balatrogame.wiki)
- [Lovely Docs](https://github.com/ethangreen-dev/lovely-injector)
- [OpenCL Reference](https://www.khronos.org/opencl/)
- [Lua 5.3 Manual](https://www.lua.org/manual/5.3/)

## Support

- GitHub Issues: Report bugs and feature requests
- Discord: Join Balatro modding community
- Reddit: r/balatro for discussions