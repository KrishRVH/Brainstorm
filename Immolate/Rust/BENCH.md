# Immolate Rust Benchmarks

This is the operating guide for benchmarking the Rust DLL against the legacy
C++ oracle DLL.

## Benchmark Philosophy

Correctness and speed are separate questions. Run `mise run compare` first to
prove that the C++ oracle and Rust DLL return the same answers through the
Windows ABI. Run `mise run bench-compare` only after parity passes.

For performance work:

1. Prove C++ vs Rust parity with `mise run compare`.
2. Run the same benchmark command before and after the change.
3. Treat small wins skeptically unless they clear the reported coefficient of
   variation (`cv`).
4. Keep the change only if the relevant fixture improves and parity still
   passes.

Keep `BENCH_THREADS=1` for implementation comparisons. That isolates the hot
path and avoids confusing algorithm/runtime differences with scheduling noise.
Use `BENCH_THREADS=0` when measuring Lua auto-reroll UX, because the Lua UI
passes `threads=0` to the DLL.

The harness fails on result mismatches. A fast wrong seed is a benchmark
failure.

## Canonical Commands

Build the Rust DLL used by Brainstorm:

```bash
mise run build-rust
```

Build and compare C++ vs Rust:

```bash
mise run compare
```

Run the complete C++ vs Rust dashboard. This is the full-suite pretty command
and fails if Rust is not faster than C++ on any case or if any result differs:

```bash
mise run bench-pretty
```

Run the full benchmark catalog as a strict per-case gate:

```bash
mise run bench-full
```

Run the actual Lua UI UX gate. This uses the reset/default 100,000 SPF budget,
lets the DLL choose the thread count exactly as the UI does, requires Rust to
beat C++ on every UX case, and fails if any result differs:

```bash
mise run bench-ux
```

Run the Rust DLL benchmark only:

```bash
BENCH_BUDGET=100000 BENCH_REPEAT=3 BENCH_CASE=all mise run bench
```

Run one profiling group:

```bash
BENCH_CASE=jokers BENCH_BUDGET=100000 BENCH_REPEAT=5 mise run bench-compare
```

Run script-friendly TSV output:

```bash
BENCH_FORMAT=tsv BENCH_COLOR=never \
  BENCH_BUDGET=100000 BENCH_REPEAT=3 BENCH_CASE=all \
  mise run bench-compare
```

Run a thorough TSV archive for later parsing:

```bash
BENCH_CASE=all \
  BENCH_BUDGET=1000000 \
  BENCH_REPEAT=7 \
  BENCH_WARMUP=2 \
  BENCH_THREADS=1 \
  BENCH_FORMAT=tsv \
  BENCH_COLOR=never \
  mise run bench-compare > bench.tsv
```

Run the full Rust validation gate, including a small benchmark smoke test:

```bash
mise run check-rust
```

## Requirements

Install the same tools used by the Rust rewrite gate:

- Rust with the Windows GNU target:
  `rustup target add x86_64-pc-windows-gnu`
- MinGW-w64 for building and inspecting Windows DLLs.
- Wine for running the Windows DLL harness.

Wine may print a `wine32 is missing` warning on Linux. That warning is not a
failure for this project as long as the 64-bit harness continues and exits
successfully.

## Benchmark Knobs

The mise tasks read these environment variables:

- `BENCH_CASE=all|baseline|tags|vouchers|packs|jokers|souls|deck|ux|CASE_NAME`
- `BENCH_BUDGET=1000000`
- `BENCH_REPEAT=5`
- `BENCH_WARMUP=1`
- `BENCH_THREADS=1`
- `BENCH_MIN_RATIO=1.0`
- `BENCH_FORMAT=pretty|tsv`
- `BENCH_COLOR=auto|always|never`

`BENCH_BUDGET` is the search budget passed to the DLL as `num_seeds`.
`BENCH_REPEAT` controls repeated measurements for each case. Use at least
`BENCH_REPEAT=3` for local comparisons and at least `BENCH_BUDGET=100000` when
looking for meaningful regressions. Larger runs, for example
`BENCH_BUDGET=1000000 BENCH_REPEAT=5`, are better for optimization decisions.
`BENCH_WARMUP` controls discarded warmup calls before the measured samples.

Treat small deltas against the coefficient of variation (`cv`). For very small
changes, raise both `BENCH_BUDGET` and `BENCH_REPEAT`.

## Pretty Dashboard

`BENCH_FORMAT=pretty` is the default. When stdout is an interactive terminal,
the harness shows a live status line for the active DLL call with elapsed time.
The final report excludes rendering time and includes:

- per-case C++ and Rust throughput
- scanned percentage, so early-hit fixtures are obvious
- mean latency, p50/p95/p99 latency, min/max latency, and stdev
- `ns/seed`, which is often the clearest hot-path metric
- coefficient of variation (`cv`) to flag noisy measurements
- per-run sparklines for C++ and Rust sample stability
- Rust/C++ speedup ratio and pass/fail coloring against `BENCH_MIN_RATIO`
- geometric-mean speedups per profiling group
- ranked Rust ahead/behind sections with fixture notes
- a high-variance section when either implementation has CV above 5%

Use `BENCH_COLOR=always` when piping through a terminal renderer that preserves
ANSI color. Use `BENCH_COLOR=never` for plain logs.

## Fixture Groups

The benchmark suite is shared by the DLL comparison harness and the native
Rust-only helper.

- `baseline-hit`: no filters; isolates ABI/result overhead.
- `tag-hit`, `dual-tag`: blind tag checks.
- `voucher-hit`: ante-1 voucher roll.
- `pack-hit`, `observatory`: pack slots plus voucher/pack coupling.
- `shop-hit`, `shop-miss`, `pack-joker`, `any-joker`: joker generation across
  shop and Buffoon pack paths.
- `pack-miss`, `souls-arcana`, `perkeo`: Soul counting and legendary-pool paths.
- `erratic`, `erratic-suit`: Erratic Deck opening-card filters.
- `ux-*`: UI-reachable combinations derived from the Lua controls.

No-match/full-budget cases are the most useful for raw throughput. Early-hit
cases are still valuable because they catch overhead, result handling, and
short-circuit behavior. The `scan` column tells you which kind of fixture you
are looking at.

## TSV Output

`BENCH_FORMAT=tsv` keeps automation simple. It prints rows with this shape:

```text
kind    impl|status  case  group  shape  budget  scanned  scan_pct  threads  sample  elapsed_ms  seeds_per_sec  ns_per_seed  min_ms  p50_ms  p95_ms  p99_ms  max_ms  stdev_ms  cv_pct  result
run     ...
summary ...
compare ...
```

For `compare` rows, the `impl` column carries the row status (`ok`,
`regression`, or `result-mismatch`). The relation is stored in the `result`
field as semicolon-delimited details such as `ratio`, `target_ratio`, `lhs`,
`rhs`, `lhs_sps`, `rhs_sps`, `lhs_ms`, `rhs_ms`, `lhs_result`, and
`rhs_result`.

`rust-vs-cpp` controls the benchmark process exit status through
`BENCH_MIN_RATIO`. Result mismatches also fail the process.

## Optional Native Rust-Only Benchmark

For quick Linux-side profiling of the Rust core without the Windows DLL ABI,
use the native helper. It runs only the Rust implementation.

```bash
cargo run --manifest-path Immolate/Rust/Cargo.toml --release --bin brainstorm_bench -- \
  --case all --budget 1000000 --threads 1 --repeat 5
```

For UI-style profiling:

```bash
cargo run --manifest-path Immolate/Rust/Cargo.toml --release --bin brainstorm_bench -- \
  --case ux --budget 100000 --threads 0 --repeat 5 --warmup 2
```

Do not use this helper to claim drop-in DLL parity or C++ superiority. It avoids
Wine and the C ABI, so it is best for inner-loop profiling only.

## Agent Workflow

Before changing hot-path code:

1. Run `mise run compare` and make sure C++ and Rust are in parity.
2. Run the complete dashboard with `BENCH_CASE=all`,
   `BENCH_BUDGET=1000000`, `BENCH_REPEAT=7`, `BENCH_WARMUP=2`, and
   `BENCH_MIN_RATIO=1.0`.
3. Run the UX gate with `BENCH_CASE=ux`, `BENCH_BUDGET=100000`,
   `BENCH_THREADS=0`, and `BENCH_MIN_RATIO=1.0`.
4. Identify the weakest Rust groups from "Rust Behind C++", "Group Speedups",
   and any "High Variance" warnings.
5. Make the smallest performance-oriented change that preserves parity.
6. Rerun the exact same benchmark command.
7. Keep the change only if the relevant fixture improves beyond measurement
   noise or the tradeoff is explicitly justified.
8. Finish with `mise run check-rust` before release or deployment work.

When adding a benchmark fixture, update `Immolate/Rust/src/bench_cases.rs`. Both
`Immolate/Rust/src/bin/immolate_dll_harness.rs` and
`Immolate/Rust/src/bin/brainstorm_bench.rs` read from that shared catalog.
