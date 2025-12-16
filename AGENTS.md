# Repository Guidelines

Quick reference for contributing to Brainstorm (Balatro mod with CPU-first DLL and optional experimental GPU DLL).

## Project Structure & Module Organization
- Lua entry/UI: `Core/Brainstorm.lua`, `Core/logger.lua`, `UI/ui.lua`; config/compat in `config.lua`, `lovely.toml`, `nativefs.lua`, `steamodded_compat.lua`.
- Native sources: CPU entry `ImmolateCPP/src/brainstorm_cpu.cpp`; experimental GPU under `ImmolateCPP/src/gpu_experimental/` (`brainstorm_cuda.cpp` + `gpu/` loaders/kernels).
- Artifacts: CPU DLL is `ImmolateCPU.dll` (default); GPU DLL is `ImmolateCUDA.dll` + PTX/fatbin (optional). Scripts in repo root (`deploy.sh`, `build_production.sh`, `lint.sh`, `validate.sh`, `run_tests.lua`); fixtures in `test_data/`.

## Build, Test, and Development Commands
- CPU build: `cd ImmolateCPP && ./build_cpu.sh` (outputs `ImmolateCPU.dll`). GPU (experimental): `./build_gpu.sh [--cpu-only|--with-tests]` -> `ImmolateCUDA.dll`.
- One-shot deploy (CPU-only, cleans target): `./deploy.sh` (rebuilds CPU DLL, wipes target Brainstorm folder, copies fresh files). Set `DEPLOY_GPU=1` to also copy GPU artifacts.
- Release packaging: `./build_production.sh` (formats, lints, Lua smoke tests, builds CPU DLL, zips to `release/Brainstorm_v3.0.zip`; `INCLUDE_GPU=1` adds GPU bits).
- Tests: `lua run_tests.lua` (basic + Lua/UI/CPU smoke); lint/format with `./lint.sh` (stylua, luacheck if present, clang-format dry-run, optional clang-tidy). `validate.sh` checks DLL size/syntax; `VALIDATE_GPU=1` adds GPU DLL check.

## Architecture & FFI Safety
- CPU DLL entry: `immolate.brainstorm(seed, voucher, pack, tag1, tag2, souls, observatory, perkeo)`; always `free_result()` on non-empty returns and wrap FFI in `pcall`.
- Lua loads `ImmolateCPU.dll` by default; only loads `ImmolateCUDA.dll` when `use_gpu_experimental=true`. Debug logs go to `brainstorm.log` when `debug_enabled=true`; DLL metrics log to `brainstorm_cpu.log` if `BRAINSTORM_CPU_DEBUG` is set.
- Pack filter simulates both shop pack slots; voucher check is ante-1 voucher; observatory/perkeo paths reuse early RNG state.

## Coding Style & Naming Conventions
- Lua: Stylua (`stylua.toml`) — 2-space indent, ~80 cols. Avoid globals, guard debug, return tables explicitly.
- C++: C++17 with RAII; GPU code behind `#ifdef GPU_ENABLED`; keep stdout minimal. clang-format via scripts; clang-tidy optional.
- Naming: Lua locals/functions lower_snake; constants upper snake (`Brainstorm.VERSION`); C++ types PascalCase, globals with `g_` (GPU experimental) or plain statics in CPU path.

## Commit & Pull Request Guidelines
- Use short, imperative subjects (scope prefix optional: `core:`, `gpu:`, `ui:`). Do not commit `release/` artifacts.
- In PRs, state intent, tests run (CPU/GPU), and note binary artifacts touched (`ImmolateCPU.dll`, `ImmolateCUDA.dll`, `seed_filter.ptx/fatbin`, `gpu_worker.exe`). Attach UI screenshots for visual changes.

## Testing Guidelines
- Minimum: `lua basic_test.lua` and `./lint.sh`; mention results.
- Smoke: `lua run_tests.lua` (file checks + Lua/UI syntax + CPU DLL presence).
- GPU work: build with `./build_gpu.sh --with-tests` and/or run `ImmolateCPP/tests/cuda_drv_probe.cpp` (requires proper setup).
- Use `test_data/test_seeds.txt` for fixtures instead of hard-coding.
