# Brainstorm with Joker search, Erratic deck search, Save states etc.

Brainstorm is a Balatro mod that rapidly searches for seeds matching voucher/pack/tag filters and integrates directly into the game loop via Lua + a native DLL.

## Setup (Required First)
1. Install `smods-1.0.0-beta` (Steamodded) for Balatro.
2. Install Lovely.
3. Build the DLL (from source):
```bash
make build
```
4. Deploy the mod (from source):
```bash
make deploy TARGET=/mnt/c/Users/Krish/AppData/Roaming/Balatro/Mods/Brainstorm
```
If `/mnt/c` write permissions fail, run with `sudo` or adjust mount permissions.
If you do not want to build from source, skip to "Installation (no build)" below.

## Credits
This project is licensed under CC BY-NC-SA 4.0.

- Brainstorm created by OceanRamen. The basis (Lua) are derived from https://github.com/OceanRamen/Brainstorm which is licensed under the Mozilla Public License Version 2.0
- Immolate native DLL source created by MathIsFun0. Project is CC BY-NC-SA 4.0 accordingly to be compliant with his source: https://github.com/SpectralPack/Immolate/tree/26f41efcc313f045bc8bdbf49e5851c56ac40b31
- I mostly finished the Immolate cpp-rewrite and ported functionality that was left unfinished and added the Joker search and stuff. The LLM I used was pretty shit at doing it itself so there was a lot of wrangling involved lol

## Features
- Auto-reroll with dual-tag support (order-agnostic or same-tag-twice).
- First-shop filters: voucher, two pack slots (e.g., Mega Spectral), specific Joker in shop slots or Buffoon packs, observatory (Telescope + Mega Celestial), Perkeo (The Soul rolls Perkeo).
- Joker list is alphabetized with a name filter for quick searching; Reset All clears filters and preferences back to defaults.
- Save/load state (Z/X + 1-5), reroll hotkeys (Ctrl+R, Ctrl+A), settings UI (Ctrl+T).

## Requirements
- Balatro (Steam, Windows 64-bit).
- Lovely injector (required): https://github.com/ethangreen-dev/lovely-injector
- WSL2 for building/deploying from this repo (MinGW-w64).
- Write access to `%AppData%\Roaming\Balatro\Mods`.

## Build & Deploy (from source)
**Release packaging:** `make release` (builds the DLL, creates `release/Brainstorm_v3.0.zip`).

## Installation (no build)
Download the latest release zip from https://github.com/KrishRVH/Brainstorm/releases/tag/3.0.0 and extract it into `%AppData%\Roaming\Balatro\Mods\Brainstorm\` (same payload as `make deploy`).
The folder name must be `Brainstorm`.
Reload the game to activate the mod.

Copy the mod files into `%AppData%\Roaming\Balatro\Mods\Brainstorm\` (same payload as `make deploy`) if you are assembling the payload manually:
```
Brainstorm/
├── Brainstorm.lua
├── UI.lua
├── Immolate.dll           # Native DLL
├── config.lua
├── lovely.toml
├── nativefs.lua
└── steamodded_compat.lua
```
You can copy these from a release zip (e.g. `release/Brainstorm_v3.0.zip`) or from the repo after someone provides `Immolate.dll`.
## Usage
- Open settings: Ctrl+T. Toggle auto-reroll: Ctrl+A. Manual reroll: Ctrl+R.
- Save/load state: Z/X + 1-5.
- Configure filters: dual tags, voucher, pack (two shop slots), Joker (searchable list + location), souls, observatory, Perkeo.
- Use "Reset All" in the Brainstorm tab to restore filter and Erratic deck settings to defaults.

## Troubleshooting
- Missing DLL or wrong build: rerun `make build` and `make deploy`.
