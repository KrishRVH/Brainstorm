# Brainstorm for Balatro
<img width="1536" height="864" alt="Brainstorm 3" src="https://github.com/user-attachments/assets/68a97977-3c80-4259-9378-58540a7b749e" />

**Just want to install it?** Open this repository's **Releases** page, click
the `latest` release, download the Brainstorm zip, and follow the installation
guide written in that release.

Brainstorm is a Balatro mod that rapidly searches for seeds matching
voucher/pack/tag/Joker/Erratic Deck filters and integrates directly into the
game loop through Lua plus a native Rust DLL.

This fork is a substantial expansion of OceanRamen's original Brainstorm mod:
it adds the Rust native search engine, first-shop Joker search, dual-tag
filters, Erratic Deck filters, save/load state slots, searchable Joker UI,
resettable preferences, live auto-reroll scan counts, benchmark automation,
release packaging, and compatibility fixes for the current Balatro mod stack.

## Setup (Required First)
1. Install `smods-1.0.0-beta` (Steamodded) for Balatro.
2. Install Lovely.
3. Build the DLL from source:
```bash
mise trust
mise run build
```
4. Deploy the mod from source:
```bash
TARGET=/mnt/c/Users/Krish/AppData/Roaming/Balatro/Mods/Brainstorm mise run deploy
```
If `/mnt/c` write permissions fail, run with `sudo` or adjust mount permissions.
If you do not want to build from source, skip to "Installation (no build)" below.

## Credits
This project is licensed under CC BY-NC-SA 4.0.

- Brainstorm was created by OceanRamen. This KRVH fork derives from
  https://github.com/OceanRamen/Brainstorm, which is licensed under the Mozilla
  Public License Version 2.0.
- The Immolate native DLL source was created in C++ by MathIsFun0. KRVH rewrote
  that native code in Rust, ported unfinished functionality, and added the Joker
  search workflow. This project uses CC BY-NC-SA 4.0 to remain compatible with
  the original Immolate source:
  https://github.com/SpectralPack/Immolate/tree/26f41efcc313f045bc8bdbf49e5851c56ac40b31.

## Features
- Auto-reroll with dual-tag support (order-agnostic or same-tag-twice).
- First-shop filters: voucher, two pack slots (e.g., Mega Spectral), specific
  Joker in shop slots or Buffoon packs, observatory (Telescope + Mega
  Celestial), Perkeo (The Soul rolls Perkeo).
- Erratic Deck filters for face-card count, no-face searches, and suit-ratio searches.
- Joker list is alphabetized with a name filter for quick searching; Reset All
  clears filters and preferences back to defaults.
- Save/load state (Z/X + 1-5), reroll hotkeys (Ctrl+R, Ctrl+A), settings UI (Ctrl+T).
- Rust benchmark harness compares current speed against the Original Brainstorm
  DLL where the older ABI supports the same fixture.
- Production release automation publishes the `latest` release with a versioned
  title and versioned zip artifact.

## Requirements
- Balatro (Steam, Windows 64-bit).
- Lovely injector (required): https://github.com/ethangreen-dev/lovely-injector
- WSL2 for building/deploying from this repo.
- mise for development tasks: https://mise.jdx.dev/
- Rust 1.96+ with the Windows GNU target:
```bash
rustup target add x86_64-pc-windows-gnu
```
- MinGW-w64 and Wine are required for Windows DLL builds, DLL validation, and
  benchmarks.
- Write access to `%AppData%\Roaming\Balatro\Mods`.

## Build & Deploy (from source)
`mise.toml` is the development interface. Run `mise trust` once per checkout,
then use `mise run <task>`.

`mise run build` builds the Rust native DLL and writes `Immolate.dll`.

`mise run lint` runs Lua formatting, LuaJIT bytecode syntax checks, luacheck,
rustfmt, and clippy. `mise run check-rust` runs Rust formatting, clippy, unit
tests, DLL export/import validation, and a benchmark smoke. `mise run check`
runs both.

Strict full-suite benchmark report:

```bash
mise run bench-full
```

Actual Lua UI UX benchmark report:

```bash
mise run bench-ux
```

See `Immolate/BENCH.md` for benchmark workflows.

## Versioning & Release
The source of truth for the mod version is `[manifest].version` in
`lovely.toml`. `steamodded_compat.lua` carries the same version for Steamodded
metadata and is checked by `mise run check-version`.

Use this when bumping versions:

```bash
VERSION=3.2 mise run bump-version
```

`mise run release` runs validation, builds `Immolate.dll`, and creates
`release/Brainstorm_v<VERSION>.zip`.

`.github/workflows/release.yml` runs on pushes to `master` and can also be
triggered manually. It rebuilds the release zip and updates the production
release titled `Brainstorm Supercharged v<VERSION>` at tag `latest`.

## Documentation
- `AGENTS.md`: contributor and agent-facing project rules.
- `BalatroSource_Guide.md`: verified Balatro source mechanics relevant to
  search parity and future mod work.
- `Immolate/BENCH.md`: benchmark harness, gates, and fixture groups.

## Installation (no build)
Download the latest release zip from
https://github.com/KrishRVH/Brainstorm/releases/tag/latest and extract it into
`%AppData%\Roaming\Balatro\Mods\Brainstorm\` (same payload as
`mise run deploy`).
The folder name must be exactly `Brainstorm`.
Reload the game to activate the mod.

Copy the mod files into `%AppData%\Roaming\Balatro\Mods\Brainstorm\` if you
are assembling the payload manually:
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

## Usage
- Open settings: Ctrl+T. Toggle auto-reroll: Ctrl+A. Manual reroll: Ctrl+R.
- Save/load state: Z/X + 1-5.
- Configure filters: dual tags, voucher, pack (two shop slots), Joker
  (searchable list + location), souls, observatory, Perkeo.
- Configure Erratic Deck filters when searching for opening hands by face-card
  count, no faces, or suit concentration.
- Use "Reset All" in the Brainstorm tab to restore filter and Erratic deck
  settings to defaults.

## Troubleshooting
- Missing DLL or wrong build: rerun `mise run build` and
  `TARGET=/mnt/c/Users/Krish/AppData/Roaming/Balatro/Mods/Brainstorm mise run deploy`.
