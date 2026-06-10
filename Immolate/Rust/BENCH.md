# Immolate Rust Benchmarks

This is the operating guide for benchmarking the current Rust DLL against the
legacy C++ DLL oracle.

## Benchmark Philosophy

Correctness and speed are separate questions. Use `mise run compare` first to prove
that the C++ oracle and Rust DLL return the same answers through the Windows ABI.
Use `mise run bench-compare` only after parity passes.

For performance work, default to this loop:

1. Prove C++ vs current Rust parity with `mise run compare`.
2. Establish the noise floor with an A/A run where `rust-candidate` points at
   the same DLL as `rust-base`.
3. Validate the real candidate with `RUST_CANDIDATE_DLL=... mise run compare`.
4. Benchmark C++ vs current Rust vs the candidate using the same case set,
   budget, repeat count, warmup count, and thread count.
5. Treat small wins skeptically unless they clear the A/A drift and the reported
   coefficient of variation (`cv`).

Keep `BENCH_THREADS=1` for implementation comparisons. That isolates the hot
path and avoids confusing algorithm/runtime differences with scheduling noise.
Use `BENCH_THREADS=0` when measuring Lua auto-reroll UX, because the Lua UI
passes `threads=0` to the DLL.

The harness fails on result mismatches. A fast wrong seed is a benchmark failure.

## Canonical Commands

Build the current Rust DLL used by Brainstorm:

```bash
mise run build-rust
```

Build and compare C++ vs current Rust:

```bash
mise run compare
```

Run the complete C++ vs current Rust dashboard. This is the full-suite pretty
command and fails if Rust is not faster than C++ on any case or if any result
differs:

```bash
mise run bench-pretty
```

Run the full benchmark catalog as a strict per-case gate. This fails if Rust is
not faster than C++ on any case or if any result differs:

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

Use `bench-compare` for performance claims. It builds the C++ oracle and the
current Rust DLL, labels the current Rust implementation as `rust-base`, runs
both through the same Windows ABI harness under Wine, and fails if `rust-base`
drops below the configured C++ throughput threshold.

## Candidate DLL Workflow

The repo now has one Rust implementation. `mise run build-rust` writes the default
Brainstorm DLL to both `target/rust/Immolate.dll` and `Immolate.dll`.

To compare a future optimization candidate, build or copy that candidate DLL to
a separate path and pass it to the existing harness:

```bash
RUST_CANDIDATE_DLL=/path/to/candidate/Immolate.dll mise run compare

RUST_CANDIDATE_DLL=/path/to/candidate/Immolate.dll mise run bench-pretty
```

`RUST_BASE_DLL` is still available when you need to benchmark against a saved
artifact instead of the freshly built current Rust DLL:

```bash
RUST_BASE_DLL=/path/to/base/Immolate.dll \
  RUST_CANDIDATE_DLL=/path/to/candidate/Immolate.dll \
  BENCH_CASE=all \
  BENCH_BUDGET=1000000 \
  BENCH_REPEAT=7 \
  BENCH_WARMUP=2 \
  BENCH_THREADS=1 \
  mise run bench-compare
```

For a serious candidate comparison, run the same command twice: once as A/A
with `RUST_CANDIDATE_DLL=target/rust/Immolate.dll`, then once with the real
candidate. If the real candidate only moves by about the same amount as the A/A
run, keep investigating before claiming an improvement.

Candidate per-case ratio rows are informational unless you set candidate gates.
The candidate result must still match C++; result mismatch is always a failure.
Set `BENCH_CANDIDATE_MIN_RATIO` to make `bench-compare` fail unless the
candidate/base non-hit full-budget geometric mean reaches that target.

## Latest Validated Local Run

The latest local documentation-sync pass validated the current Rust DLL with:

```bash
mise run check-rust
mise run bench-full
mise run bench-ux
```

Those gates passed with result parity enforced across C++ and Rust. Rust beat
C++ on every full-suite fixture and every UI-reachable UX fixture in those runs.
The narrowest observed full-suite ratio was `1.124x` on
`ux-tag-voucher-pack`; the narrowest observed UX ratio was `1.611x` on
`ux-voucher-pack`.

Treat these ratios as local measurements, not permanent constants. The named
gates are the source of truth: they fail if any compared result differs or if
`rust-base` drops below the configured C++ throughput threshold.

`mise run check-rust` includes only a benchmark smoke test
(`BENCH_BUDGET=1000 BENCH_REPEAT=1` by default in the mise task), not an
optimization-quality measurement.

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
- `BENCH_CANDIDATE_MIN_RATIO=0`
- `BENCH_CANDIDATE_MIN_SCAN_PCT=0.95`
- `BENCH_FORMAT=pretty|tsv`
- `BENCH_COLOR=auto|always|never`
- `RUST_BASE_DLL=/path/to/base/Immolate.dll`
- `RUST_CANDIDATE_DLL=/path/to/candidate/Immolate.dll`

`BENCH_BUDGET` is the search budget passed to the DLL as `num_seeds`.
`BENCH_REPEAT` controls repeated measurements for each case. Use at least
`BENCH_REPEAT=3` for local comparisons and at least `BENCH_BUDGET=100000` when
looking for meaningful regressions. Larger runs, for example
`BENCH_BUDGET=1000000 BENCH_REPEAT=5`, are better for optimization decisions.
`BENCH_WARMUP` controls discarded warmup calls before the measured samples. Keep
at least one warmup for candidate comparisons unless you are specifically
measuring cold-call behavior.

Treat small deltas against the coefficient of variation (`cv`). If an A/A run
where `rust-candidate` points at `target/rust/Immolate.dll` shows
`rust-candidate`/`rust-base` around `1.000x` with sub-1% CV, then a later
candidate needs to beat that noise floor before it is evidence of a real win.
For very small changes, raise both `BENCH_BUDGET` and `BENCH_REPEAT`.

Keep `BENCH_THREADS=1` as the default for implementation comparisons. Use
`BENCH_THREADS=0` for user-experience benchmarking because that is what the Lua
UI passes during auto-reroll.

## Pretty Dashboard

`BENCH_FORMAT=pretty` is the default. When stdout is an interactive terminal,
the harness shows a live status line for the active DLL call with elapsed time.
The final report excludes rendering time and includes:

- per-case C++, `rust-base`, and optional `rust-candidate` throughput
- scanned percentage, so early-hit fixtures are obvious
- mean latency, p50/p95/p99 latency, min/max latency, and stdev
- `ns/seed`, which is often the clearest hot-path metric
- coefficient of variation (`cv`) to flag noisy measurements
- per-run sparklines for C++ and Rust sample stability
- `rust-base`/C++ speedup ratio and pass/fail coloring against `BENCH_MIN_RATIO`
- optional `rust-candidate`/C++ and `rust-candidate`/`rust-base` ratios
- optional `rust-candidate` non-hit full-budget geometric mean
- geometric-mean speedups per profiling group
- ranked `rust-base` ahead/behind sections with fixture notes
- optional `rust-candidate` gained/lost sections against `rust-base`
- a high-variance section when any implementation has CV above 5%

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
- `ux-*`: UI-reachable combinations derived from the Lua controls, including
  tags, vouchers, pack selection, joker location, Souls, instant Observatory,
  instant Perkeo, and Erratic Deck filters.

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
aggregate ...
```

For `compare` and `aggregate` rows, the `impl` column carries the row status
(`ok`, `below-target`, `regression`, or `result-mismatch`). The relation is
stored in the `result` field as semicolon-delimited details such as `ratio`,
`target_ratio`, `lhs`, `rhs`, `lhs_sps`, `rhs_sps`, `lhs_ms`, `rhs_ms`,
`lhs_result`, and `rhs_result`.

`rust-base-vs-cpp` controls the benchmark process exit status through
`BENCH_MIN_RATIO`. Result mismatches also fail the process. Candidate per-case
rows are informational unless the candidate result differs from C++; that is a
failure. The aggregate row controls exit status when
`BENCH_CANDIDATE_MIN_RATIO` is greater than zero.

## Optional Native Rust-Only Benchmark

For quick Linux-side profiling of the Rust core without the Windows DLL ABI,
use the native helper. It runs only the current Rust implementation.

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

1. Run `mise run compare` and make sure C++ and `rust-base` are in parity.
2. Run an A/A benchmark with `RUST_CANDIDATE_DLL=target/rust/Immolate.dll` and
   save the candidate/base ratio plus CV values.
3. Run the complete dashboard with `BENCH_CASE=all`,
   `BENCH_BUDGET=1000000`, `BENCH_REPEAT=7`, `BENCH_WARMUP=2`, and
   `BENCH_MIN_RATIO=1.0`.
4. Run the UX gate with `BENCH_CASE=ux`, `BENCH_BUDGET=100000`,
   `BENCH_THREADS=0`, and `BENCH_MIN_RATIO=1.0`.
5. Identify the weakest Rust groups from "Rust-base Behind C++",
   "Group Speedups", and any "High Variance" warnings.
6. Make the smallest performance-oriented change that preserves parity.
7. Build the experiment as a candidate DLL and run
   `RUST_CANDIDATE_DLL=/path/to/candidate/Immolate.dll mise run compare`.
8. Rerun the exact same `bench-compare` command with `RUST_CANDIDATE_DLL=...`.
9. Keep the change only if the relevant fixture improves beyond the A/A noise or
   the tradeoff is explicitly justified.
10. Finish with `mise run check-rust` before release or deployment work.

When adding a benchmark fixture, update `Immolate/Rust/src/bench_cases.rs`. Both
`Immolate/Rust/src/bin/immolate_dll_harness.rs` and
`Immolate/Rust/src/bin/brainstorm_bench.rs` read from that shared catalog.
