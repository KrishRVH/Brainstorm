# Brainstorm v3.0 – Balatro Seed Finder (CPU-first, GPU optional)

Brainstorm is a Balatro mod that rapidly searches for seeds matching voucher/pack/tag filters and can optionally use an experimental CUDA path. The default build is CPU-only and integrates directly into the game loop via Lua + a native DLL.

## Features
- Auto-reroll with dual-tag support (order-agnostic or same-tag-twice).
- First-shop filters: voucher, two pack slots (e.g., Mega Spectral), observatory (Telescope + Mega Celestial), Perkeo (Investment tag + soul).
- Save/load state (Z/X + 1-5), reroll hotkeys (Ctrl+R, Ctrl+A), settings UI (Ctrl+T).
- Debug logging (`brainstorm.log` when enabled) plus optional DLL metrics (`BRAINSTORM_CPU_DEBUG`).
- Experimental GPU build available separately (`ImmolateCUDA.dll`).

## Requirements
- Balatro (Steam, Windows 64-bit).
- WSL2 for building/deploying from this repo (MinGW-w64; optional CUDA toolkit for GPU build).
- Write access to `%AppData%\Roaming\Balatro\Mods`.

## Build & Deploy (from source)
**One-step deploy (CPU-only):**
```bash
./deploy.sh              # rebuilds ImmolateCPU.dll, cleans target mod folder, copies fresh files
```
If `/mnt/c` write permissions fail, run with `sudo` or adjust mount permissions.

**Manual build:**
```bash
cd ImmolateCPP && ./build_cpu.sh   # outputs ImmolateCPU.dll
```
Optional GPU build: `./build_gpu.sh` (outputs ImmolateCUDA.dll + PTX/fatbin; not deployed by default).

**Release packaging:** `./build_production.sh` (formats, lints, runs Lua smoke tests, builds CPU DLL, zips to `release/Brainstorm_v3.0.zip`; set `INCLUDE_GPU=1` to add GPU artifacts).

## Installation (prebuilt)
Place files under `%AppData%\Roaming\Balatro\Mods\Brainstorm\`:
```
Brainstorm/
├── Core/Brainstorm.lua
├── Core/logger.lua
├── UI/ui.lua
├── ImmolateCPU.dll        # CPU default DLL
├── config.lua
├── lovely.toml
├── nativefs.lua
└── steamodded_compat.lua
```
If using GPU experimental, also add `ImmolateCUDA.dll` and `seed_filter.ptx`/`seed_filter.fatbin`.

## Usage
- Open settings: Ctrl+T. Toggle auto-reroll: Ctrl+A. Manual reroll: Ctrl+R.
- Save/load state: Z/X + 1-5.
- Configure filters: dual tags, voucher, pack (two shop slots), souls, observatory, Perkeo.
- Debug: set `debug_enabled=true` in `config.lua`; logs to `brainstorm.log`. For DLL-side metrics, set env `BRAINSTORM_CPU_DEBUG=1`.

## Testing & Lint
- Lua smoke tests: `lua run_tests.lua` (basic file checks, config/UI syntax, CPU DLL presence).
- Lint/format: `./lint.sh` (stylua, luacheck if present, clang-format dry-run; optional clang-tidy).

## Notes on GPU
- CPU path is the stable default. GPU is experimental: build with `./build_gpu.sh` and deploy with `DEPLOY_GPU=1` if desired. Lua loads `ImmolateCUDA.dll` only when `use_gpu_experimental=true` in `config.lua`.

## Troubleshooting
- Missing DLL or wrong build: rerun `./deploy.sh` (CPU-only) to rebuild and clean-deploy.
- No `brainstorm.log`: ensure `debug_enabled=true`; fallback logger now writes even if structured logger fails.
- Packs/voucher mismatch: enable debug, check `brainstorm.log` and `brainstorm_cpu.log` (set `BRAINSTORM_CPU_DEBUG=1`) for the exact names sent to the DLL.
