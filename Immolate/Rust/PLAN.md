# Rust Immolate Rewrite Plan

Last updated: 2026-06-09

## Mission

Rewrite the native `Immolate/` C++ source as a Rust implementation that builds
to `Immolate.dll` and works as a drop-in replacement for the DLL currently
loaded by `Brainstorm.lua`.

This is not a line-by-line translation project. The Rust implementation should
be idiomatic, allocation-conscious, benchmarked, and aggressively optimized
where that does not change observable behavior. The user-facing contract is:

- Lua still loads `Brainstorm.PATH .. "/Immolate.dll"`.
- Lua still calls `brainstorm_search(...)` with the current FFI signature.
- Non-empty results are still returned as owned C strings and freed through
  `free_result`.
- Null means no matching seed was found in the scanned budget.
- Current first-shop search behavior remains compatible with the mod.

The rewrite keeps the C++ implementation buildable as a legacy oracle while
Rust is the default DLL build. Candidate promotion still requires parity and
performance gates before release/deploy work.

## Current Repo Facts

- Native C++ source lives in `Immolate/`.
- Rust implementation lives in `Immolate/Rust/` as the `immolate` crate. It builds both
  an `rlib` for tests/harnesses and a Windows `cdylib` for the shipped DLL.
- `Immolate/Rust/Cargo.toml` uses Rust 2024, `rust-version = "1.96.0"`, release LTO,
  `panic = "abort"`, and a calibrated clippy profile for bit-exact numeric
  compatibility.
- Current `Makefile` builds `Immolate.dll` from Rust by default. The C++ build
  remains available as `make build-cpp` and writes the oracle artifact to
  `target/cpp/Immolate.dll`.
- `make release` and `make deploy` both consume the top-level `Immolate.dll`.
- The existing C++ DLL exports 187 names because of `--export-all-symbols`.
  Lua only relies on three: `brainstorm_search`, `immolate_set_log_path`, and
  `free_result`.
- Local toolchain discovered during planning:
  - `cargo 1.96.0`
  - `rustc 1.96.0`, host `x86_64-unknown-linux-gnu`
  - installed Rust target: `x86_64-unknown-linux-gnu`
  - available Rust Windows targets include `x86_64-pc-windows-gnu`
  - `x86_64-w64-mingw32-g++`, `wine`, `objdump`, and
    `x86_64-w64-mingw32-objdump` are available
  - this workspace has `x86_64-pc-windows-gnu` installed; fresh environments
    may still need `rustup target add x86_64-pc-windows-gnu`

## Compatibility Contract

The ABI is fixed by `Brainstorm.lua`:

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

Rust must export these exact unmangled names with the C ABI. With Rust 2024,
use explicit unsafe export attributes at the boundary, for example
`#[unsafe(no_mangle)] pub extern "C" fn ...`.

Use Windows C-compatible Rust types at the boundary:

```rust
use std::os::raw::{c_char, c_double, c_int, c_longlong};

#[unsafe(no_mangle)]
pub extern "C" fn brainstorm_search(
    seed_start: *const c_char,
    voucher_key: *const c_char,
    pack_key: *const c_char,
    tag1_key: *const c_char,
    tag2_key: *const c_char,
    joker_name: *const c_char,
    joker_location: *const c_char,
    souls: c_double,
    observatory: bool,
    perkeo: bool,
    deck_key: *const c_char,
    erratic: bool,
    no_faces: bool,
    min_face_cards: c_int,
    suit_ratio: c_double,
    num_seeds: c_longlong,
    threads: c_int,
) -> *const c_char {
    // FFI boundary only; delegate to safe core.
}

#[unsafe(no_mangle)]
pub extern "C" fn immolate_set_log_path(path: *const c_char) {
    // No-op while logging remains disabled.
}

#[unsafe(no_mangle)]
pub extern "C" fn free_result(result: *const c_char) {
    // No-op on null; otherwise reclaim with CString::from_raw(result.cast_mut()).
}
```

Do not use `c_long` for `num_seeds`; Windows `long` is 32-bit, while the Lua
FFI contract is C `long long`.
For this 64-bit Windows C ABI, the `bool` arguments map to Rust `bool`; keep the
argument order exactly as declared by Lua.

FFI rules:

- Treat null C string pointers as empty strings, matching C++.
- Parse all C strings once at the FFI boundary into internal enums/config.
- Return `std::ptr::null()` for no result or invalid internal failure.
- Return successful seeds with `CString::into_raw`.
- Implement `free_result` with `CString::from_raw` and only call it for
  pointers returned by the same Rust DLL.
- Do not free Rust-owned strings with C `free`, and do not free C++ pointers
  with Rust `free_result`.
- Do not let panics cross FFI. Because the intended release profile uses
  `panic = "abort"`, the implementation should avoid panic paths rather than
  relying on `catch_unwind` to recover.
- Avoid `unwrap`/`expect` in exported functions. Convert UTF-8/interior-NUL or
  allocation failures into null returns.
- `immolate_set_log_path` can remain a no-op unless logging is explicitly
  re-enabled; logging is currently disabled in both Lua and C++.

Build/export rules:

- Build a PE x86-64 Windows DLL for `x86_64-pc-windows-gnu`.
- The final artifact copied to repo root must be exactly `Immolate.dll`.
- Rust should export only the three public ABI symbols plus unavoidable runtime
  support. Add an export allowlist test; do not reproduce C++'s accidental C++
  symbol leakage.
- Check imports as well as exports. Release/deploy only package
  `Immolate.dll`, so the Rust artifact must not require unshipped sidecars such
  as `libgcc_s_seh-1.dll` or `libwinpthread-1.dll`.
- Verify with `x86_64-w64-mingw32-objdump -p Immolate.dll`.

## Source Of Truth And Tie-Breaking

Use three levels of truth, in this order:

1. The Lua ABI and mod behavior in `Brainstorm.lua` and `UI.lua`.
2. Current C++ DLL behavior for broad parity and regression comparison.
3. `BalatroSource/` and `BalatroSource_Guide.md` when a suspected C++ model
   mismatch appears.

Default policy: match current C++ behavior first. If a trace shows C++ disagrees
with `BalatroSource/`, do not silently "fix" it in Rust. Record the mismatch,
add a failing/expected test, and make an explicit decision before changing
behavior.

Important Balatro source references:

- RNG and seeds: `BalatroSource/functions/misc_functions.lua`
- Run initialization: `BalatroSource/game.lua`
- Pools, tags, vouchers, packs, card creation:
  `BalatroSource/functions/common_events.lua`
- Shop card creation: `BalatroSource/functions/UI_definitions.lua`
- Booster contents: `BalatroSource/card.lua`
- Erratic deck flag: `BalatroSource/back.lua`
- Summary guide: `BalatroSource_Guide.md`

## Non-Negotiable Correctness Invariants

### RNG and floating point

This is the highest-risk part of the rewrite. Rust must reproduce the C++/Lua
numeric behavior exactly.

Port and lock down golden vectors for:

- `pseudohash`
- `pseudohash_from`
- `pseudostep`
- `fract`
- `round13`
- `LuaRandom` seed expansion
- `LuaRandom` ten-call warmup
- `random`
- `randint`
- `Instance::get_node`
- `randchoice`
- `randweightedchoice`

Do not replace the RNG with Rust's RNG, `rand`, `fastrand`, `SmallRng`, or an
"equivalent" floating-point formula. Do not use fast-math or approximated modulo
logic. Preserve IEEE-754 bit reinterpretation and wrapping `u64` shifts.

`Instance::get_node` is stateful. Every RNG key has mutable per-seed state:

1. Initial node value is based on `pseudohash_from(id, seed.pseudohash(id.len()))`.
2. Every call advances the node with
   `round13(fract(node * 1.72431234 + 2.134453429141))`.
3. The returned seed is `(node + hashed_seed) / 2`.

Call order is behavior. Tests must catch reordered or skipped calls.

Additional low-level traps to preserve:

- `anteToString` is not a general integer formatter. For values above 99, the
  C++ implementation emits two characters derived from `a / 10` and `a % 10`.
  If resample indices ever exceed 99, Rust must match this behavior rather than
  using `to_string()`.
- `nextSpectral` checks the same `soul_Spectral<ante>` RNG key twice when a
  card is soulable: once for `The_Soul`, then again for `Black_Hole`. The second
  call advances the same node and must not be collapsed.
- `SPECTRALS` contains `RETRY` placeholders for Soul and Black Hole. Resampling
  behavior around those placeholders is observable and must be covered by tests.
- `shop_has_joker` consumes both shop slots even if the first slot matches.
  Keep that behavior because it may affect later same-key shop state in traces.

### Seed ordering

The seed space and ordering are part of the DLL contract:

- Character set: `123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`
- `O` is included in this C++ sequence because it is present in
  `kSeedChars`; do not confuse this with Balatro's random string generator
  excluding `O`.
- Maximum length: 8.
- Seed space: `2_318_107_019_761`.
- Iteration order starts:
  `<empty>`, `1`, `11`, `111`, ..., `11111111`, `21111111`, ...
- `Seed(string).getID()` is used as the numeric start offset.
- `Search::search()` returns an empty string when no seed is found; the FFI
  maps that empty result to null.

Golden tests must cover:

- `""`
- `"1"`
- `"11111111"`
- carry behavior
- nonzero `seed_start`
- lowercase/invalid characters as current C++ handles them
- budgets `1`, `2`, `35`, `36`, `1000`

### Search scheduling

Current search behavior:

- Default seed budget for `num_seeds <= 0`: `100_000_000`.
- Thread count:
  - if `threads > 0`, clamp to `1..=4`
  - if `threads <= 0`, use hardware concurrency capped at `4`
- Block size: `1_000_000`.
- Each worker claims blocks using an atomic block counter.
- `exitOnFind` is true for Brainstorm search.
- With multiple threads, "first found" is scheduling-dependent if more than one
  block contains a match.

Testing rule:

- Exact result parity must use `threads=1`.
- Threaded runs should verify returned seeds are valid matches and measure
  throughput/stability, not require the same lowest seed as single-threaded
  runs.

### Item and pool ordering

Treat `Immolate/items.hpp` as test data, not just definitions. The order of enum
variants and static arrays affects RNG results.

Rust item tables must preserve:

- `CARDS` order and rank/suit decoding used by Erratic filters.
- `TAROTS`, `PLANETS`, `SPECTRALS`.
- `COMMON_JOKERS`, `UNCOMMON_JOKERS`, `RARE_JOKERS`,
  `LEGENDARY_JOKERS`.
- Legacy `*_100` joker pools for version-specific behavior, even if the
  current version normally uses 1.0.1c pools.
- `VOUCHERS`, `TAGS`, `BOSSES`.
- `PACKS` weighted order and total sentinel behavior from C++.

Add tests that hash or snapshot the order of each table so accidental reordering
is caught immediately.

### First-shop filter call order

Current C++ only simulates the pieces needed by active filters. Preserve this
unless an explicit correctness decision says otherwise.

At ante 1:

1. `initLocks(ante=1, freshProfile=false, freshRun=false)`.
2. `setDeck(deck)`.
3. Generate tags only if `tag1` or `tag2` is requested.
4. Generate first voucher only if voucher or observatory is requested.
5. Generate two pack slots only if pack, observatory, Perkeo, souls, or
   pack-Joker search needs packs.
6. Check voucher/pack/observatory filters.
7. Generate shop Joker slots only if a shop/any Joker filter is requested.
8. Generate Buffoon pack contents only if pack/any Joker search needs them.
9. Generate soulable pack contents only if souls or Perkeo are requested.
10. Run Erratic deck filters last.

This conditional structure is part of C++ parity because it controls RNG key
state. Do not "simplify" by always generating a full first shop until trace
tests prove and approve the resulting behavior.

Known C++ vs BalatroSource risk: current search calls
`initLocks(1, false, false)`, so it does not apply fresh-profile or fresh-run
locks. That may allow dependency vouchers, profile-locked Jokers,
enhancement-gated Jokers, and undiscovered planets that Balatro's
`get_current_pool` would exclude. Preserve this for C++ parity first; investigate
and explicitly approve any game-parity correction later.

### Pack behavior

Current C++ pack behavior:

- `nextPack(ante)` forces the first ante <= 2 pack to `Buffoon_Pack` when
  `generatedFirstPack` is false and version > 10099.
- The second pack is weighted by `shop_pack1`.
- The UI passes a list of variant keys for a pack family; Lua currently sends
  the first key in that list. C++ normalizes variant suffixes like
  `p_spectral_mega_1` to a family-level item.
- The search checks both shop pack slots.
- Pack contents temporarily lock generated items to avoid duplicates inside the
  same pack, then unlock them after the pack.
- Arcana/Spectral packs are soulable. Celestial/Standard/Buffoon packs are not
  counted for `souls`.
- Buffoon pack sizes/choices:
  - Normal: size 2, choices 1
  - Jumbo: size 4, choices 1
  - Mega: size 4, choices 2

BalatroSource nuance:

- The game's first shop Buffoon special case chooses
  `p_buffoon_normal_1` or `p_buffoon_normal_2` with `math.random(1, 2)`.
- C++ models this at the family level as `Buffoon_Pack`. Preserve the current
  family-level behavior unless a separate game-parity fix is approved.

### Soul and Perkeo behavior

Current behavior to preserve:

- Soul checks only apply to Arcana and Spectral packs in the current shop slots.
- Pack filter restriction applies before counting souls or trying Perkeo.
- Spectral can roll both Soul and Black Hole checks.
- Perkeo requires The Soul first, then the Soul legendary pool must yield
  `Perkeo`.
- `soul_yields_perkeo` intentionally calls `inst.random("rarity1sou")` before
  `nextJoker(ItemSource::Soul, ante, false)`. This looks redundant because
  Soul forces legendary rarity, but it likely preserves game call order. Copy it
  until a BalatroSource/in-game trace proves it should be removed.
- For Mega packs, Perkeo attempts are limited by `packInfo(pack).choices`, not
  by every Soul card generated.

### Erratic deck behavior

Current behavior:

- Erratic filters only apply when `erratic` is true.
- If `erratic` is true but `min_face_cards <= 0` and `suit_ratio <= 0`, the
  filter is effectively skipped.
- The analysis samples 52 cards with RNG key `erratic`.
- `no_faces` removes face ranks before counting totals and suits.
- `min_face_cards` requires at least that many Jack/Queen/King cards.
- `suit_ratio` is clamped to `<= 1.0` at parse time and checks the ratio of the
  two most common suits over remaining total cards.

Card table order matters here. C++ decodes rank/suit from `CARDS` index; Rust
must preserve that exact layout or replace it with a tested explicit mapping.

## Rust Architecture

Use a two-layer design:

1. A tiny, auditable unsafe FFI layer.
2. A safe or mostly-safe search/simulation core optimized around fixed data and
   reusable per-thread state.

Recommended module layout:

```text
Immolate/Rust/
  Cargo.toml
  src/
    lib.rs
    ffi.rs
    api.rs
    config.rs
    seed.rs
    rng.rs
    item.rs
    tables.rs
    instance.rs
    filters.rs
    search.rs
    trace.rs
    tests/
```

Suggested responsibilities:

- `ffi.rs`: raw pointer conversion, unsafe boundary, exported functions, result
  ownership.
- `api.rs`: Rust-callable public API used by tests/bench harness.
- `config.rs`: parse string keys into enums and normalized filter config.
- `seed.rs`: seed storage, ID conversion, incremental next, seed hash cache.
- `rng.rs`: bit-exact Lua RNG and pseudoseed helpers.
- `item.rs`: `#[repr(u16)]` or `#[repr(u32)]` item enum plus compact helper
  predicates.
- `tables.rs`: static arrays, weighted packs, pack info, parse maps.
- `instance.rs`: one mutable simulated run state for one seed.
- `filters.rs`: current `apply_filters` equivalent.
- `search.rs`: single-thread and block-based multi-thread search.
- `trace.rs`: test-only trace generation for parity diagnostics.

Parse the FFI inputs into a `CompiledFilter` once before search. The hot loop
should not use trait-object filter dispatch or string lookups. A worker should
call a concrete `fn apply_filters(instance: &mut Instance, cfg: &CompiledFilter)
-> bool` or an equivalent monomorphized path.

### Data structure guidance

The current C++ hot path spends work on `std::map<std::string, double>`,
temporary `std::string` keys, vectors for small packs, and repeated dynamic
dispatch. Rust should improve this without changing behavior.

Preferred hot-path structures:

- `Seed` as fixed arrays:
  - `chars: [i8; 8]`
  - `len: u8`
  - `cache: [[f64; 48]; 8]`, initialized with sentinel `-1.0`
- `Instance` reused per seed/block:
  - `locked: [bool; ITEM_COUNT]` or bitset if benchmarked
  - `node_cache` with fixed key IDs where possible
  - `generated_first_pack: bool`
  - `params` as plain copyable fields
  - small fixed arrays for pack contents
- `Worker` per thread:
  - current `Seed`
  - current `Instance`
  - `CompiledFilter`
  - scratch pack/card buffers
  - local counters for benchmarking/diagnostics
- `PackBuffer<T, const N: usize>` or stack arrays instead of heap `Vec` for
  pack sizes up to 5.
- Internal enums for filter config so string parsing is not in the loop.
- Static slices and const arrays for item pools.
- `Item` as `#[repr(u16)]` or `#[repr(u32)]` with explicit discriminants
  matching C++ enum order. Numeric ordering affects range checks, voucher
  offsets, pack-info indexing, and stake comparisons.

RNG key storage options, in recommended order:

1. Implement fixed key IDs for all known hot keys and generate resample keys
   through compact key descriptors.
2. Use an interned-node cache shaped like:

   ```rust
   struct NodeCache {
       values: [f64; NODE_COUNT],
       seen: BitSet,
   }
   ```

3. Use a small linear cache of `(KeyDescriptor, f64)` entries per `Instance`
   for rare dynamically described keys. The number of keys touched in one
   first-shop simulation is small.
4. Fall back to strings only for rare debug/trace paths.

Do not use `HashMap<String, f64>` in the hot path unless benchmarks prove it is
faster than the fixed/small-cache approach.

Pack generation should have streaming helpers for production paths:

- `count_souls_in_pack_stream`
- `pack_has_joker_stream`
- `analyze_erratic_deck` with counters only

Keep vector-returning pack APIs only for tests, traces, or compatibility
helpers. The C++ code allocates vectors for small packs; Rust should avoid that
in common search paths.

### Threading guidance

- Keep the current cap of 4 worker threads.
- Use `std::thread::scope` or owned threads with no shared mutable simulation
  state.
- Shared state should be limited to:
  - atomic next block index
  - atomic found flag
  - a mutex or atomic result slot for found seed
- Each worker should own:
  - current `Seed`
  - current `Instance`
  - local buffers
- Preserve block size `1_000_000` until benchmarks justify changing it.
- Benchmark thread counts 1, 2, and 4. More threads inside the game process are
  likely to hurt frame pacing and should not be enabled by default.

### Optimization rules

Correctness gates come first. After unit and trace parity are green, optimize in
small measured steps.

High-value optimization targets:

- Remove heap allocation from per-seed filter evaluation.
- Avoid rebuilding seed strings except when returning a result.
- Advance seed state incrementally with `Seed::next()` instead of reconstructing
  from ID per seed.
- Replace string concatenation for RNG keys with compact key descriptors.
- Inline tiny RNG helpers after golden vectors are stable.
- Keep parse results as enums and booleans.
- Short-circuit filters in the same order as C++.
- Specialize common filter combinations if benchmarks prove it helps and parity
  tests cover the specialization.

Avoid:

- Fast-math flags.
- Approximate hashing.
- Global mutable caches shared across threads.
- Panics or allocation failures crossing FFI.
- Rayon or broad dependency additions before a std-thread baseline exists.
- Optimizations that change call order.

### Unsafe and SIMD policy

Do not start with SIMD. The workload is branchy, stateful, and
floating-compatibility-sensitive. The first complete implementation should be
scalar and parity-proven.

Unsafe is justified for the FFI boundary first. Avoid `get_unchecked`,
target-feature SIMD, raw pointer table access, or custom allocators until
profiles show a specific bottleneck and parity tests cover the optimized path.
If SIMD is explored later, hide it behind a feature flag and keep scalar as the
reference implementation.

## Build Plan

### Cargo changes

Update `Immolate/Rust/Cargo.toml`:

```toml
[package]
name = "immolate"
version = "0.1.0"
edition = "2024"
rust-version = "1.96.0"

[lib]
name = "immolate"
crate-type = ["cdylib", "rlib"]
```

Keep the existing strict release profile. Change linting so unsafe is allowed
only where needed:

- Prefer crate-level `unsafe_code = "deny"` if FFI can be isolated with explicit
  allows.
- Otherwise document every unsafe block in `ffi.rs`.
- Keep clippy's `correctness` and `suspicious` denies.
- The exported ABI has many arguments and booleans by contract; add narrow
  `allow` attributes for those functions instead of weakening the whole crate.

### Make targets

Introduce explicit targets:

```text
make build-rust      # builds Rust and copies Immolate.dll to repo root
make build-cpp       # builds legacy C++ oracle DLL
make build           # aliases build-rust after parity gates
make test-rust       # cargo test for Rust crate
make compare         # run C++ vs Rust parity harness
make bench           # run Rust benchmark harness
make bench-compare   # run thresholded C++ vs Rust benchmark harness
make check-rust      # format, clippy, tests, DLL validation, compare, smoke bench
```

Current build shape:

- Keep C++ available through `make build-cpp` as the oracle.
- Rust is the default `make build` path.
- Make `make release` and `make deploy` depend on `make check-rust` so they
  consume the Rust DLL only after parity and benchmark gates pass.
- Preserve repo-root `Immolate.dll` as the mod payload location.

Example build command:

```bash
rustup target add x86_64-pc-windows-gnu
cargo build --manifest-path Immolate/Rust/Cargo.toml --release --target x86_64-pc-windows-gnu
cp Immolate/Rust/target/x86_64-pc-windows-gnu/release/immolate.dll Immolate.dll
```

The exact DLL filename emitted by Cargo may depend on crate naming and target.
Verify and copy/rename to `Immolate.dll`.

## Test Strategy

Build tests before the full Rust search. The rewrite is only shippable when all
layers pass.

### Layer 1: Rust unit tests

Test pure Rust internals without loading DLLs:

- RNG/math golden vectors:
  - `fract`
  - `round13`
  - `pseudohash`
  - `pseudohash_from`
  - `pseudostep`
  - `LuaRandom::random`
  - `LuaRandom::randint`
- Seed tests:
  - string to ID
  - ID to string
  - incremental `next()`
  - cache behavior
  - carry/wrap cases
- Parsing tests:
  - every UI tag key
  - every UI voucher key
  - every pack variant key and normalized family
  - joker names from Balatro center names / C++ `stringToItem`
  - known Joker display-name edge cases such as `Caino` vs `Canio` and
    `Seance` vs any stylized C++ spelling
  - deck keys
  - invalid strings mapping to current defaults
- Table order tests:
  - item enum discriminants
  - static array order
  - weighted pack table
  - pack info sizes/choices
- Filter helper tests:
  - tag pair matching
  - voucher/observatory checks
  - pack slot matching
  - Erratic face/suit stats

### Layer 2: C++ baseline fixture generation

Before replacing behavior, create a baseline dataset from current C++:

- Build the current C++ DLL with `make build-cpp`.
- Run deterministic cases with `threads=1`.
- Store case inputs and expected outputs in a fixture format such as JSONL.
- Keep fixture generation reproducible and separate from fixture verification.

Fixture fields should include:

```json
{
  "case_id": "voucher_telescope_small_budget",
  "seed_start": "",
  "voucher_key": "v_telescope",
  "pack_key": "",
  "tag1_key": "",
  "tag2_key": "",
  "joker_name": "",
  "joker_location": "any",
  "souls": 0,
  "observatory": false,
  "perkeo": false,
  "deck_key": "b_red",
  "erratic": false,
  "no_faces": false,
  "min_face_cards": 0,
  "suit_ratio": 0.0,
  "num_seeds": 100000,
  "threads": 1,
  "expected": "..."
}
```

### Layer 3: DLL parity harness

Use the in-crate Windows runner built by `make build-harness`. It loads two DLL
paths and calls the exact ABI under Wine:

```bash
make compare
```

Runner requirements:

- Load each DLL separately.
- Resolve `brainstorm_search` and `free_result`.
- Call the function with the exact ABI.
- Convert non-null result with C string rules.
- Free result through the same DLL that returned it.
- Record elapsed time for each call.
- Treat process crash as a failed case.
- Use `threads=1` for exact result parity.
- Emit stable tab-separated output for shell diffing and benchmark capture.

The implemented harness lives at `Immolate/Rust/src/bin/immolate_dll_harness.rs` and is
built as a Windows executable by `make build-harness`.

### Layer 4: trace parity

Black-box search results can hide wrong call order. Add a test-only trace path
that exposes deterministic first-shop internals for both C++ and Rust.

Acceptable trace surfaces:

- A CLI compiled from shared core code.
- Feature-gated Rust test APIs plus a C++ helper executable.
- A test-only DLL export that is excluded from release builds.

Keep the production ABI unchanged. If test-only DLL exports are used, gate them
out of release builds and make them return strings freed by `free_result`.
Suggested diagnostic exports or CLI commands:

```text
brainstorm_trace_seed_json(...) -> char*
brainstorm_scan_json(...) -> char*
```

`brainstorm_trace_seed_json` should take the same filter arguments plus one seed
and return stable JSON for that seed's first-shop simulation.
`brainstorm_scan_json` should take the same search arguments and return
diagnostic metadata such as result, scanned count, elapsed time, thread count,
and start seed ID.

Trace events should include enough detail to diagnose drift:

- seed
- parsed filter config
- RNG key requested
- node value after `get_node`
- RNG values with enough precision to debug drift; include raw `u64` state when
  useful
- random float/int result
- generated small/big tag
- first voucher
- pack slot 1 and 2
- shop item slots
- pack contents
- Soul count
- Soul legendary result
- Erratic stats
- final pass/fail result and first failing reason

Trace parity must cover:

- tag-only
- voucher-only
- pack-only
- joker shop
- joker pack
- souls
- Perkeo
- observatory
- Erratic filters
- combinations that skip earlier sections

### Layer 5: BalatroSource spot checks

Use `BalatroSource/` as a small source oracle for known seeds and first-shop
observables:

- tags from `get_next_tag_key`
- voucher from `get_next_voucher_key`
- first two packs from `get_pack('shop_pack')`
- shop card generation from `create_card_for_shop`
- pack contents from `Card:open`
- Erratic deck card generation

These checks should be limited and explicit. They are for detecting C++ model
drift from the real game, not for replacing the C++ parity suite.

## Golden Case Matrix

Create named golden cases for each category below. Use small budgets where
possible and include a few larger budgets for rare filters.

### ABI and defaults

- No filters, budget 1, seed `""`.
- No filters, seed `"1"`.
- Null input strings from the harness, if the harness can express them.
- Empty strings for all optional keys.
- Unknown tag/voucher/pack/deck/joker strings.
- Non-null result is freed successfully.
- Null result is not freed.
- Repeated successful calls do not leak or crash.

### Seed progression

- Budgets `1`, `2`, `35`, `36`, `1000`.
- Start seeds `""`, `"1"`, `"11111111"`, and a mid-space seed.
- Carry transition around all-`Z` suffixes.
- Invalid/lowercase seed inputs, matching current C++ behavior.

### Tags

- Single tag in either blind.
- Two distinct tags in either order.
- Same tag twice requiring both blind tags to match.
- One empty tag and one selected tag.
- UI-supported tags:
  - `tag_uncommon`
  - `tag_rare`
  - `tag_holo`
  - `tag_foil`
  - `tag_polychrome`
  - `tag_investment`
  - `tag_voucher`
  - `tag_boss`
  - `tag_charm`
  - `tag_juggle`
  - `tag_double`
  - `tag_coupon`
  - `tag_economy`
  - `tag_skip`
  - `tag_d_six`
- C++-supported but not currently visible UI tags should also be parser-tested.

### Vouchers

- Every UI voucher key:
  - `v_overstock_norm`
  - `v_clearance_sale`
  - `v_hone`
  - `v_reroll_surplus`
  - `v_crystal_ball`
  - `v_telescope`
  - `v_grabber`
  - `v_wasteful`
  - `v_tarot_merchant`
  - `v_planet_merchant`
  - `v_seed_money`
  - `v_blank`
  - `v_magic_trick`
  - `v_hieroglyph`
  - `v_directors_cut`
  - `v_paint_brush`
- C++ parser support for tier-two vouchers:
  - `v_overstock_plus`
  - `v_liquidation`
  - `v_glow_up`
  - `v_reroll_glut`
  - `v_omen_globe`
  - `v_observatory`
  - `v_nacho_tong`
  - `v_recyclomancy`
  - `v_tarot_tycoon`
  - `v_planet_tycoon`
  - `v_money_tree`
  - `v_antimatter`
  - `v_illusion`
  - `v_petroglyph`
  - `v_retcon`
  - `v_palette`

### Packs

- Every UI pack family/size.
- Variant normalization:
  - `_normal_1`, `_normal_2`, etc. all map to the same family.
- First forced Buffoon slot.
- Second weighted slot.
- Pack filter restricting Joker/Soul/Perkeo checks to the selected family.

### Jokers

- Shop-only location.
- Pack-only location.
- Any location.
- Common, uncommon, rare, and legendary Joker names.
- Invalid Joker name maps to no Joker filter.
- Buffoon pack contents with temporary duplicate locks.
- Pack Joker search respects selected pack filter.

### Souls and Perkeo

- `souls=1`
- `souls=2`
- negative souls normalizes to zero
- Arcana pack Soul count
- Spectral pack Soul count
- Celestial/Standard/Buffoon do not count as soulable
- `perkeo=true`
- combined `souls > 0` and `perkeo=true`
- Mega pack choices limit Perkeo attempts
- Perkeo's extra `rarity1sou` roll

### Decks and Erratic

- Empty deck key defaults to Red Deck.
- `b_red`
- `b_magic` activates Crystal Ball.
- `b_nebula` activates Telescope.
- `b_zodiac` activates Tarot Merchant, Planet Merchant, and Overstock.
- `b_ghost` changes shop spectral rate.
- `b_erratic`
- Unknown deck defaults to Red Deck.
- `erratic=false` ignores face/suit filters.
- `erratic=true`, no face/suit filters, passes.
- `no_faces=true`.
- `min_face_cards` boundaries: `0`, `1`, high impossible value.
- `suit_ratio`: `0`, `0.5`, `0.6`, `0.65`, `0.7`, `0.75`, `0.8`,
  `0.85`, `1.0`, `>1.0`.
- `suit_ratio > 1.0` clamps to `1.0`.

### Budget and thread normalization

- `num_seeds < 0`
- `num_seeds = 0`
- `num_seeds = 1`
- `num_seeds = 100_000`
- `threads < 0`
- `threads = 0`
- `threads = 1`
- `threads = 2`
- `threads = 4`
- `threads = 99`

## Benchmark Strategy

Benchmarks must compare Rust against C++ and also track Rust-only regressions
over time.

Separate correctness from throughput. Early-hit searches are useful, but they
mostly measure where the first match lands. For throughput, prefer known no-hit
windows or add a test-only `count_matches` / `--no-exit` harness path that scans
the whole budget.

### Benchmark harness

Use the same DLL harness for correctness and timing so the measured path
matches the real Lua FFI boundary.

Required output fields:

- case id
- implementation (`cpp` or `rust`)
- DLL path
- result seed or null
- elapsed nanoseconds
- DLL load time separately from scan time
- budget
- threads
- seeds scanned or attempted
- full-scan vs early-exit mode
- seeds/sec
- wall time and CPU time if cheaply available
- process RSS or memory delta if available
- binary size
- export count
- status (`ok`, `mismatch`, `crash`, `timeout`)

Run each case enough times to report:

- p50
- p95
- p99
- min
- max
- standard deviation if cheap

### Benchmark classes

Micro:

- Budget `1`
- Budget `100`
- Budget `10_000`
- Measures FFI overhead and per-seed baseline.

UI-normal:

- Budget `250`
- Budget `1_000`
- Budget `100_000`
- Matches UI SPF options and likely gameplay settings.

Heavy:

- Budget `1_000_000`
- Budget `10_000_000`
- No-hit or no-exit scans for stable seeds/sec.

Rare filters:

- Perkeo
- souls
- Joker in Buffoon pack
- Observatory
- Erratic suit ratio
- high `min_face_cards`

Thread scaling:

- `threads=1`
- `threads=2`
- `threads=4`
- Compare throughput and returned-seed validity.

Memory/FFI stability:

- 10,000 repeated budget-1 calls.
- Repeated successful result allocation/free.
- Repeated null result calls.
- Mixed C++ and Rust DLL calls in the same harness process, never crossing
  ownership between DLLs.

### Performance acceptance gates

Minimum gates before making Rust default:

- 100% Rust unit tests pass.
- 100% single-thread black-box parity against C++ golden fixtures.
- 100% required trace parity, or documented intentional differences.
- Rust export check passes.
- Rust DLL loads under Wine and through the harness.
- Repeated allocation/free tests pass.
- During bring-up, Rust release builds should reach at least 85-90% of C++'s
  geometric mean on full-scan single-thread fixtures before optimization work
  is considered healthy.
- Before making Rust the default, Rust should be at parity or faster than C++
  on the main no-match UI-normal benchmark.
- Rust must be meaningfully faster than C++ on at least one hot benchmark before
  the rewrite is called "optimized".
- No benchmark fixture may regress by more than 20% without an explicit note and
  accepted reason.

Suggested aspirational performance goals:

- At least 1.5x C++ throughput on simple no-match scans.
- At least 1.25x C++ throughput on complex filters.
- No per-seed heap allocation in common filter paths.
- p99 budget-1000 warm-call latency suitable for in-game auto-reroll without
  visible frame spikes.

Do not hard-code aspirational numbers into CI until stable local baselines
exist.

## Implementation Phases

### Phase 0: Baseline preservation

- Add `make build-cpp` while preserving current C++ build command.
- Save C++ DLL outputs under a non-root path such as `target/cpp/Immolate.dll`
  so Rust builds do not overwrite the oracle accidentally.
- Add export inspection command for the current DLL.
- Generate initial C++ golden fixtures with `threads=1`.

Exit criteria:

- Current C++ build remains reproducible.
- Baseline fixtures can be regenerated.
- No Lua behavior changes.

### Phase 1: Rust crate scaffold

- Rename package to `immolate`.
- Add `[lib] crate-type = ["cdylib", "rlib"]`.
- Add `src/lib.rs` and module skeleton.
- Implement FFI exports with stubbed core behavior.
- Add `make build-rust`.
- Add export allowlist check.
- Add Wine/harness load smoke test.

Exit criteria:

- Rust DLL builds for `x86_64-pc-windows-gnu`.
- It copies to top-level `Immolate.dll` when explicitly requested.
- Harness can load it and resolve the three symbols.

### Phase 2: RNG and seed parity

- Port `Seed`.
- Port `fract`, `round13`, `pseudohash`, `pseudohash_from`, `pseudostep`.
- Port `LuaRandom`.
- Add unit tests from C++ generated vectors.
- Add property-style tests for seed round trips and iteration.

Exit criteria:

- RNG golden vectors are bit-exact.
- Seed ordering tests pass.
- No search/filter work proceeds until this is green.

### Phase 3: items, tables, parsing

- Port item enum and table data.
- Port `itemToString` / `stringToItem` equivalents needed by parser/tests.
- Port tag, voucher, pack, deck, Joker location, and Joker name parsing.
- Add table-order snapshots and parser tests.

Exit criteria:

- Every UI key parses correctly.
- Unknown/default behavior matches C++.
- Static table order tests pass.

### Phase 4: instance simulation

- Port lock/unlock/initLocks/initUnlocks.
- Port voucher activation and deck effects.
- Port next tag, voucher, pack, shop item, Joker, Tarot, Planet, Spectral, and
  pack content generation.
- Add trace API/CLI for first-shop events.
- Compare traces against C++ helper output.

Exit criteria:

- Trace parity passes for required first-shop events.
- Known BalatroSource spot checks are either green or documented as current C++
  model differences.

### Phase 5: filters and single-thread search

- Port `FilterConfig`.
- Port `apply_filters` preserving conditional call order.
- Port single-thread search over seed blocks.
- Wire `brainstorm_search` to the Rust core.
- Run black-box DLL parity with `threads=1`.

Exit criteria:

- All golden black-box cases match C++.
- FFI allocation/free tests pass.
- No known crash path from invalid inputs.

### Phase 6: multi-thread search

- Port atomic block scheduler and early exit.
- Preserve thread resolution and cap.
- Add returned-seed validity checks for threaded searches.
- Benchmark 1, 2, and 4 threads.

Exit criteria:

- Threaded searches return valid matching seeds.
- No data races or shared mutable simulation state.
- Throughput scaling is measured and documented.

### Phase 7: optimization pass

Only begin after parity is green.

Candidate optimization sequence:

1. Replace string RNG keys with fixed descriptors.
2. Replace dynamic node map with small fixed/linear cache.
3. Replace small `Vec`s with stack buffers.
4. Inline hot RNG helpers.
5. Specialize filter combinations generated by parsed config flags.
6. Tune block size only if benchmarks show improvement.
7. Consider narrowly scoped unsafe code only when a benchmark proves need and
   tests cover the behavior.

Each optimization must:

- Run the full unit test suite.
- Run relevant trace parity.
- Run black-box golden parity.
- Record before/after benchmark results.

### Phase 8: make Rust default

- `make build` calls `build-rust`.
- Keep `make build-cpp` available.
- Ensure `make release` and `make deploy` still package top-level
  `Immolate.dll`.
- Keep build notes current.
- Keep benchmark fixture generation documented.

Exit criteria:

- Release zip contains Rust-built `Immolate.dll`.
- C++ oracle remains available for future regression investigation.
- In-game smoke test passes on Windows/Balatro.

## Suggested Commands

Setup:

```bash
rustup target add x86_64-pc-windows-gnu
```

C++ baseline:

```bash
make build-cpp
x86_64-w64-mingw32-objdump -p target/cpp/Immolate.dll
```

Rust build:

```bash
cargo build --manifest-path Immolate/Rust/Cargo.toml --release --target x86_64-pc-windows-gnu
```

Rust tests:

```bash
cargo test --manifest-path Immolate/Rust/Cargo.toml
cargo clippy --manifest-path Immolate/Rust/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path Immolate/Rust/Cargo.toml --check
```

DLL compare:

```bash
make compare
```

Benchmarks:

```bash
make bench-compare BENCH_BUDGET=100000 BENCH_REPEAT=3 BENCH_CASE=all BENCH_MIN_RATIO=0.8
```

Export check:

```bash
file Immolate.dll
x86_64-w64-mingw32-objdump -p Immolate.dll | sed -n '/\\[Ordinal\\/Name Pointer\\]/,+80p'
x86_64-w64-mingw32-objdump -p Immolate.dll | rg 'DLL Name:|brainstorm_search|free_result|immolate_set_log_path'
```

Expected export/build shape:

- `file` reports a PE32+ x86-64 DLL.
- exports include `brainstorm_search`, `free_result`, and
  `immolate_set_log_path`.
- imports do not require unshipped runtime sidecars.

## Agent Rules For This Rewrite

- Do not delete or rewrite the C++ oracle during the Rust rewrite.
- Do not change `Brainstorm.lua` FFI unless the user explicitly approves an ABI
  migration.
- Do not commit `release/` artifacts.
- Do not rely on in-game testing alone.
- Do not optimize before RNG, seed, trace, and black-box parity are green.
- Do not change call order casually.
- Do not add dependencies for convenience if std is sufficient.
- Do not use a broad `unsafe` policy. Keep unsafe isolated and documented.
- Do not allow panics across FFI.
- Do not compare exact first result under multi-threaded scheduling.
- Always free returned pointers through the DLL that returned them.
- Treat `BalatroSource/` as readable source of truth but never commit it.
- Keep logging disabled unless explicitly re-enabled.

## Known Risk Register

RNG drift:

- Risk: tiny floating-point differences return different seeds.
- Mitigation: bit-exact golden vectors before filter/search work.

Call-order drift:

- Risk: Rust generates a full shop or reorders checks and mutates RNG keys
  differently.
- Mitigation: trace parity tests for every filter family.

Item order drift:

- Risk: Rust enum/table order differs from C++, changing random selections.
- Mitigation: table-order snapshot tests.

Thread nondeterminism:

- Risk: threaded search returns a different valid seed than C++ or single-thread.
- Mitigation: exact parity uses `threads=1`; threaded tests verify validity.

FFI ownership:

- Risk: cross-DLL frees or borrowed strings cause crashes.
- Mitigation: harness frees through originating DLL and stress-tests repeated
  success/null calls.

Rust target setup:

- Risk: `x86_64-pc-windows-gnu` target missing on a new machine.
- Mitigation: setup command in README/Makefile error message.

Over-optimization:

- Risk: compact key IDs or specializations accidentally skip RNG calls.
- Mitigation: optimize only after trace parity and benchmark each step.

C++ vs Balatro mismatch:

- Risk: Rust copies a C++ bug or "fixes" behavior without approval.
- Known examples: fresh-profile/fresh-run locks are not applied by current
  search; some Joker display names may differ between live Balatro UI names and
  C++ `stringToItem` spellings.
- Mitigation: record mismatch, add test, and require explicit decision.

## Definition Of Done

The Rust rewrite is done when:

- `Immolate.dll` built from Rust loads from `Brainstorm.lua` without Lua changes.
- The three required FFI symbols are present and callable.
- All Rust unit tests pass.
- C++ vs Rust black-box parity passes for the golden suite with `threads=1`.
- Trace parity passes for first-shop mechanics or documented intentional
  differences are approved.
- Threaded searches return valid seeds and do not crash.
- Benchmark report shows Rust matches or beats C++ on the main UI-normal path
  and improves at least one hot path.
- `make build` builds the Rust DLL.
- `make build-cpp` still builds the legacy oracle.
- `make release` and `make deploy` package the Rust DLL correctly.
- A Windows/Balatro smoke test confirms the mod works in-game.
