# Rust Immolate V2 Overhaul Plan

Last updated: 2026-06-09

## Objective

Build a second Rust implementation of Immolate that is a drop-in replacement for
the current `Immolate.dll` ABI, but is designed around search throughput instead
of around mirroring the C++ object model.

The target is not a cleanup pass over `Immolate/Rust/src/instance.rs`. The target is a
search-specialized engine that preserves current observable DLL behavior while
removing avoidable per-seed work: string RNG keys, tiny heap allocations, generic
card structs, unnecessary edition/sticker rolls, broad branchy filter dispatch,
and full Erratic simulation when a candidate can already be rejected.

Default correctness oracle: current C++ and current Rust must match first.
`BalatroSource/` is the game-mechanics source of truth for understanding and
future expansion, but source-correct behavior that differs from the shipped C++
model must be introduced only behind explicit tests and UI contract decisions.

## Deliverable Shape

Create V2 side by side with the current implementation. Do not delete the
current Rust implementation during the overhaul.

Recommended layout:

- `Immolate/Rust/src/v2/mod.rs`: public search entry for V2.
- `Immolate/Rust/src/v2/config.rs`: FFI parse result and compiled filter shape.
- `Immolate/Rust/src/v2/seed.rs`: search seed cursor and hash cache, or a compatible
  wrapper around the current `Seed` until a measured replacement exists.
- `Immolate/Rust/src/v2/rng.rs`: exact RNG functions plus fixed-key node state.
- `Immolate/Rust/src/v2/tables.rs`: static item, pool, pack, card, and bitset tables.
- `Immolate/Rust/src/v2/kernels.rs`: shape-specific search kernels.
- `Immolate/Rust/src/v2/search.rs`: single and multi-thread block orchestration.
- `Immolate/Rust/src/v2/trace.rs`: optional slow trace helpers for parity debugging.

V2 is now exposed through the existing ABI after parity and benchmark gates
passed. The previous Rust core remains available through the `v1-legacy` Cargo
feature and the `make build-rust-v1` comparison artifact.

## Compatibility Contract

The exported DLL functions remain exactly:

- `brainstorm_search(...) -> *const c_char`
- `free_result(*const c_char)`
- `immolate_set_log_path(*const c_char)`

Required ABI behavior:

- Null C string inputs are empty strings.
- Invalid or unknown filter keys fall back the same way current Rust/C++ do.
- Empty result and no match return null.
- Non-empty result is a DLL-owned C string freed by `free_result`.
- Panics must not cross FFI.
- The top-level artifact remains `Immolate.dll`.
- `make compare` and `make bench-compare` remain the release gates.

Single-thread output must match current C++/Rust exactly. Multi-thread current
behavior is block-scheduling dependent; V2 may improve this, but only with a
documented decision and tests.

## Current Implementation Audit

Current flow:

1. `ffi.rs` copies all C strings into owned `String`s.
2. `FilterConfig::from_raw` parses once into enum fields.
3. `search.rs` iterates seeds with one `Instance` per block.
4. `apply_filters` performs ante-1 simulation conditionally:
   tags, voucher, packs, joker, souls/Perkeo, Erratic.
5. `Instance::get_node` maps string RNG keys to mutable per-key node values.
6. Every random call creates a fresh `LuaRandom`, expands state, burns ten raw
   draws, and returns one value.

Good current choices:

- Parser work is already outside the per-seed loop.
- Seed iteration is incremental inside a block.
- Rust replaced C++ `std::map<std::string,double>` with a small linear cache.
- Result string allocation only happens on hit.
- The benchmark and DLL comparison harnesses are already useful.

Main hot costs:

- Per RNG call: string key lookup, sometimes `format!`, node update, full
  `LuaRandom::new`, ten warmup draws.
- Per joker identity check: edition and sticker rolls are computed even though
  the current UI filters only compare joker identity.
- Per pack content check: tiny `Vec` allocations are used when the search only
  needs `contains` or `count`.
- Per candidate: `init_locks` writes many booleans and generic pool functions
  branch through behavior that most filter shapes do not need.
- Erratic filter performs 52 generic card draws before checking final
  thresholds.
- `BLOCK_SIZE = 1_000_000` means `threads=4` does not scale for budgets at or
  below 1,000,000 because only one block exists.

## Evidence Collected

Commands run during this research:

```bash
cargo test --manifest-path Immolate/Rust/Cargo.toml
cargo run --manifest-path Immolate/Rust/Cargo.toml --release --bin brainstorm_bench -- --case all --budget 200000 --repeat 3 --threads 1
cargo run --manifest-path Immolate/Rust/Cargo.toml --release --bin brainstorm_bench -- --case dual-tag --budget 1000000 --repeat 5 --threads 1
cargo run --manifest-path Immolate/Rust/Cargo.toml --release --bin brainstorm_bench -- --case shop-miss --budget 1000000 --repeat 5 --threads 1
cargo run --manifest-path Immolate/Rust/Cargo.toml --release --bin brainstorm_bench -- --case pack-miss --budget 1000000 --repeat 5 --threads 1
cargo run --manifest-path Immolate/Rust/Cargo.toml --release --bin brainstorm_bench -- --case erratic-suit --budget 1000000 --repeat 5 --threads 1
make bench-compare BENCH_CASE=shop-miss BENCH_BUDGET=1000000 BENCH_REPEAT=3 BENCH_WARMUP=1 BENCH_FORMAT=tsv BENCH_COLOR=never
```

Representative Rust-core full-budget costs:

| Case | Cost |
| --- | ---: |
| `pack-miss` | ~164 ns/seed |
| `dual-tag` | ~340 to 345 ns/seed |
| `shop-miss` | ~789 to 793 ns/seed |
| `erratic-suit` | ~2000 to 2050 ns/scanned seed |

Threading signal:

- `shop-miss`, 10,000,000 seeds, 1 thread: ~7.88 s, ~788 ns/seed.
- `shop-miss`, 10,000,000 seeds, 4 threads: ~2.39 s, ~239 ns/seed.
- `shop-miss`, 1,000,000 seeds, 4 threads: no meaningful scaling because the
  current block size is exactly 1,000,000.

DLL C++ vs Rust signal under Wine:

- `shop-miss`, 1,000,000 seeds: C++ ~989 ms, Rust ~948 ms, only ~1.04x faster.
- This proves ordinary translation-level Rust wins are exhausted. V2 needs
  specialized algorithms.

Temporary fixed-key microbench under `/tmp`:

- Replacing current `Instance::random(key)` with a direct fixed byte-key path
  for a single RNG call improved hot keys by only ~1.11x to ~1.12x.
- Conclusion: fixed RNG keys are worthwhile, but not enough alone. The bigger
  wins must come from fused kernels and avoided simulation work.

## Must-Preserve Mechanics

### Seed Space

- Charset: `123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`.
- Max length: 8.
- Order begins `""`, `"1"`, `"11"`.
- Seed space: `2_318_107_019_761`.
- Current Rust normalizes IDs with `rem_euclid(SEED_SPACE)`. Keep current Rust
  behavior unless parity tests explicitly decide otherwise.

### RNG

The RNG is non-negotiable:

- Preserve `pseudohash`, `pseudohash_from`, `pseudostep`, `fract`, `round13`.
- Preserve per-key node state:
  `round13(fract(node * 1.72431234 + 2.134453429141))`.
- Preserve `(node + hashed_seed) / 2.0`.
- Preserve `LuaRandom` state expansion and ten warmup draws.
- Preserve `randint` and threshold comparisons exactly.
- Do not use fast-math, another RNG, approximate modulo, or compacted pool
  distributions.

### Conditional Call Order

The current search intentionally does not simulate everything. It only makes
RNG calls needed by selected filters. V2 must preserve that for C++ parity.

Current `apply_filters` order:

1. Base locks and deck setup.
2. Tags only if either tag filter is requested.
3. Voucher only if voucher or current `observatory` filter is requested.
4. Packs only if pack, observatory, Perkeo, souls, or pack-joker search needs
   them.
5. Voucher/pack/observatory checks.
6. Joker search.
7. Soul/Perkeo search.
8. Erratic deck filters.

Do not replace this with "generate the whole run/shop" unless creating a new
source-correct mode with its own tests.

### Pack Semantics

- Source first shop pack slot is forced to a normal Buffoon pack unless banned.
  Current native collapses concrete variants, so slot 1 is `Buffoon_Pack`.
- Slot 2 uses weighted `shop_pack1`.
- Current UI pack filters normalize concrete keys like `p_spectral_mega_1` to a
  pack family.
- Current native search therefore needs pack family, size, choices, and kind;
  it does not need exact art variant.

### Joker Semantics

- Shop joker search currently scans two shop slots.
- `shop_has_joker` consumes both slots even when the first matches.
- Joker identity comes from rarity and pool roll. Edition and sticker rolls are
  independent of identity for current filters.
- Buffoon packs temporarily lock generated jokers to avoid duplicates inside
  the pack, then unlock them.
- The Soul path uses legendary source `sou` and pool key `Joker4` without an
  ante suffix.
- Perkeo search requires The Soul to appear first, then the Soul legendary roll
  to select Perkeo.

### Souls

- The Soul and Black Hole are excluded from normal pools.
- Soulable Tarot/Spectral/Tarot_Planet can force The Soul with
  `soul_<type>1 > 0.997`.
- Soulable Planet/Spectral can force Black Hole with the same threshold.
- Spectral performs two calls to `soul_Spectral1`; Black Hole can overwrite a
  previous Soul result.
- Perkeo attempts are capped by pack choices, not by total souls beyond choices.

### Erratic

- Erratic uses 52 draws from `G.P_CARDS` through key `erratic`.
- `no_faces` discards face samples after sampling; it does not replace them.
- Current Erratic filters are last, so search-only early rejection can skip
  remaining Erratic draws for losing candidates without changing returned seeds.

### Source vs Native Boundaries

Current native behavior does not model every source detail:

- Native scans two shop joker slots even when source deck/voucher effects could
  increase shop size, such as Zodiac's Overstock.
- Native `observatory` filter means "Telescope voucher plus Mega Celestial pack
  available", not the Observatory voucher's scoring effect.
- Native collapses concrete booster variants to pack families.
- Native does not model unseeded `math.random` variant draws.

V2 should preserve native behavior by default. Source-correct expansions should
be separate, explicit work.

## Accepted Optimization Vectors

These are accepted because current code, source mechanics, and measurements all
support them.

### 1. Compile Filters Into Shape-Specific Kernels

Current `apply_filters` checks many booleans every seed. V2 should compile the
FFI config once into a `CompiledFilter` with a small enum:

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
- combinations that genuinely occur from the UI

Each kernel should make only the RNG calls required by the current behavior.
This is the top-level organization that enables all other optimizations.

Implementation notes:

- Keep a generic slow path for rare combinations until every shape has tests.
- Keep `trace` helpers that can run the old full object path for debugging.
- The compiled config should contain numeric item IDs, pack kinds, pack size,
  target rarity set if known, and precomputed booleans.

Expected effect:

- Removes repeated branch work from every seed.
- Makes early rejection and omitted simulation mechanically obvious.

### 2. Replace Hot Strings With Numeric IDs And Static Tables

Current parsing is already outside the loop. The hot issue is internal RNG keys
and pool membership.

Use static numeric IDs for:

- Items/centers.
- Tags.
- Vouchers.
- Packs.
- Jokers.
- Cards/ranks/suits.
- RNG keys.

Use generated or hand-checked tables:

- `ITEM_KEY`, `ITEM_NAME`, aliases.
- `SET`, `RARITY`, `ORDER`.
- Joker rarity pools.
- Voucher dependency pairs.
- Base ante-1 lock mask.
- Tag availability/min-ante data.
- Pack kind/size/choices.
- Card rank/suit/face flags.
- Cumulative pack weights.
- Cumulative shop type weights for common deck/voucher states.

Do not use `HashMap<String, f64>` in V2 hot paths. Active key sets are small and
mostly fixed; numeric IDs plus fixed arrays are the right shape.

Joker input parsing should accept current display names and source keys/aliases
where practical. Known risk aliases:

- Source `j_caino` vs native name `Canio`.
- Source `j_seance` vs native generated-name spelling for Seance.
- Preserve `tag_skip` -> native `Speed_Tag`.

### 3. Fixed-Key RNG Node State

Replace `Cache { Vec<CacheNode { key: String, value: f64 }> }` with fixed node
slots for known ante-1 keys.

Examples:

- `Tag1`
- `Voucher1`
- `shop_pack1`
- `cdt1`
- `rarity1sho`
- `rarity1buf`
- `edisho1`
- `edibuf1`
- `Joker1sho1`, `Joker2sho1`, `Joker3sho1`
- `Joker1buf1`, `Joker2buf1`, `Joker3buf1`
- `Joker4`
- `soul_Tarot1`, `soul_Spectral1`
- `Tarotar1`, `Spectralar21`, `Spectralspe1`
- `erratic`

Store each node as:

```rust
struct Node {
    initialized_for_seed_generation: u32,
    value: f64,
}
```

Or simply clear a compact fixed array per seed if that benchmarks faster.

The temporary microbench showed only ~1.11x to ~1.12x for one RNG call, so this
is not the whole optimization. It matters because joker and Erratic kernels make
many RNG calls.

### 4. Search-Only Joker Identity Generation

Current `next_joker` always rolls edition and sometimes stickers, but current
filters only need `joker == target`.

Add search-only functions:

- `next_joker_identity(source, ante, target_context) -> Item`
- `shop_slot_has_joker(target) -> bool`
- `buffoon_pack_has_joker(target, size) -> bool`

For current filters, skip:

- `poll_edition('edi...')`
- `etperpoll` / `packetper`
- `ssjr` / `packssjr`
- construction of `JokerData`

Why safe for current search:

- Edition and sticker RNG keys are independent from rarity and joker-pool keys.
- Current filters never inspect edition/sticker.
- Search resets per seed. Skipping independent unused keys does not affect later
  target identity keys in the same seed.

Guardrails:

- Keep full `next_joker` in the trace/compat path.
- If the UI later adds edition/sticker filters, route those configs to a full
  joker kernel.

Expected effect:

- Directly attacks `shop-miss`, the most expensive common non-Erratic miss path
  at ~789 ns/seed in Rust core and ~948 ns/seed through the DLL.

### 5. Streaming Pack Predicates

Replace pack-content APIs that allocate `Vec`s with streaming search functions:

- `count_souls_in_pack_stream(pack, need, perkeo_mode) -> SoulCount`
- `buffoon_pack_contains_joker_stream(pack, target) -> bool`
- `arcana_pack_soul_count_stream(size) -> u8`
- `spectral_pack_soul_count_stream(size) -> u8`

Preserve pack duplicate-lock semantics:

- For Buffoon packs, generated jokers must be temporarily locked until the pack
  is complete.
- For Tarot/Spectral packs, generated consumables are locked until the pack is
  complete.

Early exits:

- If only counting souls and `souls_found >= cfg.souls` and no Perkeo is needed,
  the kernel may stop because no later filter depends on pack-content RNG.
- If searching a Buffoon pack for a target and the target is found, the kernel
  may stop only if no later same-seed logic needs the rest of that pack. Current
  `pack_has_joker` returns immediately after constructing the whole pack, so this
  early stop is safe only in a search-only kernel that has no later dependence
  on pack locks or RNG state. The current `apply_filters` runs souls after joker
  search, but souls ignores Buffoon packs, so this is safe for current UI shapes.

Expected effect:

- Removes tiny heap allocations.
- Reduces work in pack-joker and souls kernels.

### 6. Direct Pack Slot Logic

For current ante-1 family-level search:

- Slot 1 is always `Buffoon_Pack` in the native model.
- Slot 2 is the only weighted `shop_pack1` roll.

Use direct predicates:

- Pack-only target `Buffoon_Pack`: pass immediately after slot 1, no
  `shop_pack1` RNG call needed if no later pack-dependent filter exists.
- Pack-only non-Buffoon target: roll only slot 2.
- Observatory: roll voucher first; if not Telescope, do not roll packs. If
  Telescope passes, check whether slot 2 is `Mega_Celestial_Pack` because slot 1
  is Buffoon.
- Soul searches with no pack filter: slot 1 is Buffoon and not soulable, so only
  slot 2 can matter.

This is source-backed and current-native-backed. It is one of the cleanest
algorithmic wins.

### 7. Specialized Erratic Kernel

Erratic is the slowest measured scanned path at ~2.0 us per seed.

Replace generic:

- `randchoice("erratic", &CARDS)`
- `Item` arithmetic to decode rank/suit
- full 52 draws before rejection

With:

- Direct `randint` over 52 using the fixed `erratic` node.
- Precomputed `CARD_FACE[52]` and `CARD_SUIT[52]`.
- Counters only.

Safe early exits:

- `min_face_cards`: if `face_count + remaining < min_face_cards`, reject.
- `no_faces + suit_ratio`: discarded face samples reduce the denominator. Track
  remaining possible non-face cards conservatively before rejecting.
- `suit_ratio`: compute upper bounds for the top-two suits using remaining
  draws; reject once the target ratio is impossible.
- Success early is allowed only in search-only mode because Erratic is last and
  no later filter observes the remaining `erratic` node state.

Do not use probabilistic shortcuts. Every accepted or rejected seed must match
the exact sampled sequence implied by the current logic.

### 8. Base Lock Bitsets And Temporary Overlays

Current `init_locks` writes many booleans every seed. Replace with:

- A precomputed `BASE_LOCKS_ANTE1` bitset.
- Deck/voucher-derived lock modifications.
- Temporary pack locks tracked in a small stack buffer.

Use `[u64; 8]` or equivalent for 508 item IDs.

Important:

- Unlocking pack-temporary items must not clear base locks.
- Voucher activation locks the active voucher and unlocks the next-tier voucher.
- Showman is currently false in the searched initial state; keep the generic
  bitset path able to support it, but specialize current kernels for no Showman.

Expected effect:

- Moderate by itself, useful when combined with stream pack/joker kernels.

### 9. Seed Cursor And Hash Cache Refinement

Current seed iteration is already incremental and reasonably good.

Only accept changes here after microbenching:

- Generation-stamp cache invalidation instead of resetting rows on carry.
- Store seed length and chars in a layout friendlier to pseudohash.
- Specialized `seed_to_string` only on hit.
- Avoid `Seed::from_id` inside hot loops except at block start.

Do not attempt large seed-space math shortcuts unless there is a proof for the
specific RNG key distribution. The pseudohash recurrence makes direct skipping
nontrivial.

### 10. Dynamic Block Size For Threaded Search

Current 1,000,000-seed blocks are fine for large misses but bad for UI-sized
passes at or below 1,000,000 seeds.

V2 should use a dynamic chunk size such as:

- 16,384 to 65,536 for hit-heavy or UI-sized scans.
- 262,144 to 1,000,000 for long full-budget misses.

Benchmark this by shape. The correct number may differ for Erratic vs pack-only
filters.

Optional improvement:

- Track `found_id` as an atomic minimum and continue scanning already-claimed
  lower ranges to return the earliest matching seed. This would improve user
  semantics, but it is not current C++ behavior for threaded searches, so gate it
  behind a decision.

### 11. FFI Fast Parse

This is low priority for full-budget scans, but useful for repeated early hits.

Possible changes:

- Parse `CStr` bytes directly into enums without allocating `String`s.
- Normalize pack suffixes by byte scan.
- Return seed via a small stack buffer copied into `CString` only on hit.

Do this after hot kernels. It will not move `shop-miss` or Erratic throughput.

## Rejected Or Deferred Ideas

Do not implement these in the first V2 pass:

- Replacing the RNG with any other RNG.
- Using fast-math or approximate float transforms.
- Compacting pools before sampling.
- Generating the whole shop/run for every filter.
- Introducing `HashMap<String, f64>` for RNG nodes.
- GPU offload. Current kernels are branchy, f64-heavy, and have low arithmetic
  intensity per seed. There is no evidence it will beat optimized CPU scalar
  code after transfer and batching overhead.
- SIMD across seeds. Possible later, but not accepted yet. The exact f64
  recurrence, branchy resampling, and per-key state make this a second-stage
  research topic after scalar V2 is measured.
- Precomputing full seed outcomes. The seed space is too large.
- Skipping RNG calls for winning candidates when later filters still depend on
  the same key state. Early exits must be limited to search-only losing paths or
  terminal filters.

## Implementation Phases

### Phase 0: Lock The Baseline

Before code changes:

```bash
cargo test --manifest-path Immolate/Rust/Cargo.toml
make compare
make bench-compare BENCH_CASE=all BENCH_BUDGET=1000000 BENCH_REPEAT=7 BENCH_WARMUP=2 BENCH_THREADS=1
```

Save the benchmark output outside git or in a dedicated benchmark note.

Add missing golden tests before replacing internals:

- `fract`
- `round13`
- `pseudohash_from`
- `pseudostep`
- `get_node` repeated-call sequences for hot keys
- `randchoice` resample behavior
- `randweightedchoice` boundary behavior
- Spectral double soul call
- first forced Buffoon pack
- Perkeo Soul legendary path
- Erratic no-face discard behavior

### Phase 1: V2 Skeleton And Tables

Create V2 modules and table definitions. Wire a V2 core function into tests, not
the DLL export.

Acceptance:

- V2 no-filter, tag, voucher, pack, observatory, souls, Perkeo, joker, and
  Erratic test vectors match current core.
- No V2 hot path allocates except at search start or result hit.

### Phase 2: Fixed RNG Keys

Implement numeric `RngKey` slots and direct fixed-key node updates.

Acceptance:

- Golden node sequences match current `Instance`.
- Microbench fixed-key random calls beat current by at least the measured
  ~1.10x. If not, keep the clearer path.

### Phase 3: Pack And Joker Search Kernels

Implement direct pack slot logic, joker identity generation, and streaming pack
predicates.

Acceptance:

- `shop-miss`, `pack-joker`, `any-joker`, `pack-miss`, `souls-arcana`, and
  `perkeo` match current search outputs.
- `shop-miss` improves materially. A sub-5% gain is not enough for this phase,
  because the point is to remove edition/sticker and object-model work.

### Phase 4: Erratic Kernel

Implement direct card-index Erratic sampling and safe early exits.

Acceptance:

- Erratic test vectors and randomized comparison against current core pass.
- `erratic-suit` improves materially. This is the single most expensive scanned
  path, so a small win is not enough.

### Phase 5: Threading

Tune dynamic chunk sizes by filter shape.

Acceptance:

- Large miss cases scale with 4 threads.
- 100,000 to 1,000,000 seed UI-sized scans get useful parallelism when threads
  are requested.
- Single-thread output remains exact.
- Threaded output behavior is documented.

### Phase 6: DLL Switch

Completed after V2 proved better across the benchmark matrix:

- `brainstorm_search` routes to V2 by default.
- The previous core remains available behind the `v1-legacy` feature and
  `make build-rust-v1`.
- Full release gates remain the required proof before release/deploy work.

## Benchmark Gates

Use single-thread for implementation comparisons:

```bash
make compare
make build-rust-v1 build-rust-v2
make compare RUST_BASE_DLL=target/rust-v1/Immolate.dll RUST_CANDIDATE_DLL=target/rust-v2/Immolate.dll
make bench-compare RUST_BASE_DLL=target/rust-v1/Immolate.dll RUST_CANDIDATE_DLL=target/rust-v2/Immolate.dll BENCH_CASE=all BENCH_BUDGET=1000000 BENCH_REPEAT=7 BENCH_WARMUP=2 BENCH_THREADS=1 BENCH_CANDIDATE_MIN_RATIO=10 BENCH_CANDIDATE_MIN_SCAN_PCT=0.95
```

Then test threaded behavior separately:

```bash
cargo run --manifest-path Immolate/Rust/Cargo.toml --release --bin brainstorm_bench -- --engine v2 --case shop-miss --budget 10000000 --repeat 3 --threads 1
cargo run --manifest-path Immolate/Rust/Cargo.toml --release --bin brainstorm_bench -- --engine v2 --case shop-miss --budget 10000000 --repeat 3 --threads 4
```

Minimum acceptance guidance:

- No correctness regressions in `make compare`.
- No benchmark group regresses by more than the established A/A noise floor.
- `shop-miss` should improve substantially after joker identity kernels.
- `erratic-suit` should improve substantially after Erratic specialization.
- Pack/soul cases should not regress while stream kernels are introduced.

Do not claim wins below ~1% without A/A evidence. The existing harness has shown
sub-1% noise.

## Useful Source Facts For Table Generation

From `BalatroSource/`:

- `pseudorandom_element` sorts entries by `sort_id` if present, otherwise key.
- `P_CENTER_POOLS` are sorted by `order` for most center pools.
- `G.P_CARDS` selections use key sorting, which current native `CARDS` matches
  as `C/D/H/S`.
- Boss/card keyed-table order must not be replaced with center `order`.
- `get_current_pool` keeps placeholders and resamples.
- Source booster centers are 32 concrete keys; native family aggregation is
  current behavior.
- Source Spectral excludes Soul and Black Hole from ordinary pools; soulable
  checks inject them.
- Deck-granted vouchers are applied through `Card.apply_to_run`.

Keep `BalatroSource_Guide.md` as the compact mechanics reference. Keep this file
as the implementation plan.

## Open Decisions Before Implementation

1. Should V2 preserve current native behavior for Zodiac shop size, or expand to
   source-correct shop slot count? Default: preserve native.
2. Should threaded search return the first seed in scan order or preserve current
   block-scheduling behavior? Default: preserve current unless UI wants stable
   earliest results.
3. Should the FFI accept source `j_*` joker keys in addition to display names?
   Recommended: yes, backward compatible.
4. Should source-correct concrete booster variant filters ever be exposed?
   Default: no, current UI filters are family-level.
