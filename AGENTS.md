# Repository Guidelines

Quick reference for contributing to Brainstorm (Balatro mod with a native DLL).

## Credits
- Brainstorm created by OceanRamen. Rewrite by KRVH.
- Immolate created by MathIsFun0.

## Project Structure & Module Organization
- Lua entry/UI: `Brainstorm.lua`, `UI.lua`; config/compat in `config.lua`, `lovely.toml`, `nativefs.lua`, `steamodded_compat.lua`.
- Native sources: `Immolate/*.cpp` and `Immolate/*.hpp` (CPU-only; entry is `Immolate/brainstorm.cpp`).
- Artifacts: DLL is `Immolate.dll` (default). Build/lint/format/deploy all use the repo `Makefile`.
- `BalatroSource/` is the literal game source; never commit it to git and always use it as the source of truth for understanding game behavior.
- `BalatroSource_Guide.md` summarizes seed/search-relevant mechanics verified from `BalatroSource/`.
- Logging is currently disabled (commented out) in both Lua and C++; keep it off unless explicitly re-enabled.

## Build and Development Commands
- Build: `make build` outputs `Immolate.dll`.
- Deploy: `make deploy TARGET=/mnt/c/Users/Krish/AppData/Roaming/Balatro/Mods/Brainstorm`.
- Release: `make release` (builds the DLL and zips `release/Brainstorm_v3.1.zip`).
- Formatting: `make format` (runs stylua/clang-format when available).
- Lint: `make lint` (stylua/clang-format checks when available).
- Clean: `make clean`.
- No standalone scripts or test runners; use the Makefile targets and validate in-game.

## Architecture & FFI Safety
- DLL entry: `immolate.brainstorm_search(seed_start, voucher_key, pack_key, tag1_key, tag2_key, joker_name, joker_location, souls, observatory, perkeo, deck_key, erratic, no_faces, min_face_cards, suit_ratio, num_seeds, threads)`; pass Balatro keys (e.g. `v_telescope`, `tag_charm`, `p_spectral_mega_1`), always `free_result()` on non-empty returns, and wrap FFI in `pcall`.
- Lua loads `Immolate.dll`.
- Pack filter simulates both shop pack slots; voucher check is ante-1 voucher; observatory reuses the same pack/voucher rolls, and Perkeo requires The Soul to roll Perkeo (legendary pool).
- Joker search checks the first shop: location `shop` scans shop slots, `pack` scans Buffoon packs, `any` checks both (pack search respects the selected pack filter).
- Soul checks only apply to Arcana/Spectral packs in the current shop slots.
- Auto-reroll UI shows live scanned seed counts; SPF options go up to 100,000 seeds per pass.

## Coding Style & Naming Conventions
- Lua: Stylua (`stylua.toml`) — 2-space indent, ~80 cols. Avoid globals, return tables explicitly.
- C++: C++17 with RAII; keep stdout minimal. clang-format when available.
- Naming: Lua locals/functions lower_snake; constants upper snake (`Brainstorm.VERSION`); C++ types PascalCase, file-scope statics as needed.

## Commit & Pull Request Guidelines
- Use short, imperative subjects (scope prefix optional: `core:`, `ui:`, `dll:`). Do not commit `release/` artifacts.
- In PRs, state intent and note binary artifacts touched (`Immolate.dll`). Attach UI screenshots for visual changes.
