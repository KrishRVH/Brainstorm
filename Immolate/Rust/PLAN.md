# Immolate Rust Current State

Last updated: 2026-06-10

This document records the current Rust DLL architecture and maintenance rules.
The Rust rewrite is complete. Brainstorm now has one Rust implementation, built
by default as `Immolate.dll`. The C++ source remains buildable only as the
behavior oracle for parity and benchmarking.

For benchmark operation details, use `Immolate/Rust/BENCH.md`.

## Current Status

- `mise run build` and `mise run build-rust` build the Rust DLL and copy it to the repo
  root as `Immolate.dll`.
- `mise run build-cpp` builds the C++ oracle to `target/cpp/Immolate.dll`.
- C++ oracle source lives under `Immolate/CPP/`; Rust source lives under
  `Immolate/Rust/`.
- There are no older Rust build features, mise tasks, or source modules.
- `mise run compare` validates C++ vs Rust through the Windows ABI under Wine.
- `mise run bench-compare` benchmarks C++ vs Rust through the same ABI harness.
- Benchmark comparison fails on result mismatch. A faster wrong seed is a
  failure.

## Runtime Contract

Lua loads `Brainstorm.PATH .. "/Immolate.dll"` and calls:

```c
const char* brainstorm_search(
    const char* seed_start,
    const char* voucher_key,
    const char* pack_key,
    const char* tag1_key,
    const char* tag2_key,
    const char* joker_name,
    const char* joker_location,
    double souls,
    bool observatory,
    bool perkeo,
    const char* deck_key,
    bool erratic,
    bool no_faces,
    int min_face_cards,
    double suit_ratio,
    long long num_seeds,
    int threads
);

void immolate_set_log_path(const char* path);
void free_result(const char* result);
```

FFI rules:

- Null C string pointers are treated as empty strings.
- Balatro keys are accepted for tags, vouchers, packs, and decks.
- Joker input accepts the UI-facing joker name and known internal aliases.
- No result returns a null pointer.
- A non-empty result is returned as an owned C string and must be released with
  `free_result`.
- `immolate_set_log_path` is a no-op while logging remains disabled.
- Rust exports only `brainstorm_search`, `free_result`, and
  `immolate_set_log_path`.

## Source Layout

- `Immolate/CPP/`: C++ oracle source, including `brainstorm.cpp` as the DLL
  entry implementation for oracle builds.
- `Immolate/Rust/src/ffi.rs`: C ABI boundary and string/result ownership.
- `Immolate/Rust/src/lib.rs`: public Rust API and unit tests.
- `Immolate/Rust/src/filters.rs`: raw UI/FFI filter parsing into `FilterConfig`.
- `Immolate/Rust/src/item.rs`: item, joker, pack, voucher, deck, and tag tables.
- `Immolate/Rust/src/rng.rs`: bit-compatible Lua/Balatro RNG primitives.
- `Immolate/Rust/src/seed.rs`: seed space and seed ordering.
- `Immolate/Rust/src/search.rs`: shared budget/thread helpers.
- `Immolate/Rust/src/engine/`: optimized search implementation.
- `Immolate/Rust/src/bench_cases.rs`: shared benchmark fixture catalog.
- `Immolate/Rust/src/bin/immolate_dll_harness.rs`: Windows ABI comparison and
  benchmark harness.
- `Immolate/Rust/src/bin/brainstorm_bench.rs`: native Rust-only profiling
  helper.

The public Rust entry point is `immolate::brainstorm_search_core`. The engine
uses specialized kernels where possible and a generic correctness fallback for
unsupported filter combinations.

## Search Architecture

The current implementation compiles a `FilterConfig` into a `CompiledFilter`
with a `KernelShape`. Shapes include:

- `NoMatch`
- `NoFilter`
- `TagOnly`
- `VoucherOnly`
- `PackOnly`
- `Observatory`
- `ShopJoker`
- `PackJoker`
- `AnyJoker`
- `Souls`
- `Perkeo`
- `Erratic`
- `Composite`
- `Generic`

Dedicated kernels live in `engine/kernels.rs`. `Composite` handles real UI
combinations such as tag+voucher+pack, voucher+pack, tag+joker, Souls+pack, and
Erratic+tag without falling all the way back to the old generic instance path.
`Generic` remains as a correctness fallback for unsupported combinations.

Parallel search preserves earliest-seed semantics. For parallel runs, Rust scans
the first chunk serially, then searches remaining chunks in parallel while
tracking the earliest hit by seed offset. This avoids returning a later seed
just because another worker found it first.

`threads=0` means "auto" and is what the Lua auto-reroll UI passes to the DLL.
Use `BENCH_THREADS=0` for actual user-experience benchmarking. Use
`BENCH_THREADS=1` for single-thread implementation comparisons.

## Correctness Invariants

The Rust implementation must match the observable C++ oracle unless an explicit
product decision says otherwise.

Important invariants:

- Preserve Balatro/Lua RNG behavior exactly; do not replace it with Rust RNGs or
  approximate floating-point math.
- Preserve seed ordering and wrapping over the full seed space.
- Preserve the FFI signature and result ownership rules.
- Preserve first-shop behavior for vouchers, pack slots, Buffoon pack joker
  searches, Souls, Observatory, Perkeo, and Erratic Deck filters.
- Preserve earliest matching seed behavior for both single-thread and parallel
  search.
- Treat static impossible filters as `NoMatch` only when they are impossible
  from the UI/contract, not merely unlikely.

## Build And Validation

Routine validation:

```bash
mise run check-rust
```

That runs Rust formatting, clippy, unit tests, DLL export/import validation,
C++ vs Rust parity, and a small benchmark smoke test.

Strict full-suite performance gate:

```bash
mise run bench-full
```

Actual Lua UI UX gate:

```bash
mise run bench-ux
```

Pretty full-suite dashboard:

```bash
mise run bench-pretty
```

## Latest Local Validation State

The latest post-structure-change validation passed:

- `cargo fmt --manifest-path Immolate/Rust/Cargo.toml --check`
- `cargo check --manifest-path Immolate/Rust/Cargo.toml --all-targets --target x86_64-pc-windows-gnu`
- `cargo test --manifest-path Immolate/Rust/Cargo.toml`
- `cargo clippy --manifest-path Immolate/Rust/Cargo.toml --all-targets -- -D warnings`
- `mise run build-cpp`
- `mise run build-rust`
- `mise run build-harness`
- direct `immolate_dll_harness.exe compare --cpp ... --rust ... --threads 1`
- `mise run check-rust-dll`

The direct harness comparison enforces C++/Rust result parity. Full performance
gates remain `mise run bench-full` and `mise run bench-ux`; run them before
claiming benchmark state for a release.

## Release Notes For Maintainers

- Release and deploy consume the top-level `Immolate.dll`.
- Dev prereleases are managed by `.github/workflows/dev-release.yml` at tag
  `dev-release` with release title `dev release`.
- Do not commit `release/` artifacts.
- Note `Immolate.dll` in PRs whenever it changes.
- Keep `BalatroSource/` out of git; use it only as the local source of truth for
  game behavior.
- Update `Immolate/Rust/BENCH.md` whenever benchmark fixtures, output format,
  gates, or latest validated performance results change.
