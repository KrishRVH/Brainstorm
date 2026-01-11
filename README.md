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
- Brainstorm created by OceanRamen.
- Immolate native DLL created by MathIsFun0.
- I mostly finished the Immolate cpp-rewrite and ported functionality that was left unfinished and added the Joker search and stuff. The LLM was pretty shit at doing it itself so there was a lot of wrangling involved lol

## Features
- Auto-reroll with dual-tag support (order-agnostic or same-tag-twice).
- First-shop filters: voucher, two pack slots (e.g., Mega Spectral), specific Joker in shop slots or Buffoon packs, observatory (Telescope + Mega Celestial), Perkeo (The Soul rolls Perkeo).
- Save/load state (Z/X + 1-5), reroll hotkeys (Ctrl+R, Ctrl+A), settings UI (Ctrl+T).

## Requirements
- Balatro (Steam, Windows 64-bit).
- WSL2 for building/deploying from this repo (MinGW-w64).
- Write access to `%AppData%\Roaming\Balatro\Mods`.

## Build & Deploy (from source)
**Release packaging:** `make release` (builds the DLL, creates `release/Brainstorm_v3.0.zip`).

## Installation (no build)
Copy the mod files into `%AppData%\Roaming\Balatro\Mods\Brainstorm\` (same payload as `make deploy`):
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
- Configure filters: dual tags, voucher, pack (two shop slots), Joker, souls, observatory, Perkeo.

## Troubleshooting
- Missing DLL or wrong build: rerun `make build` and `make deploy`.
