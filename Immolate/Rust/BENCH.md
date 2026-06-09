# Immolate Rust Benchmarks

This is the operating guide for benchmarking the Rust rewrite against the
legacy C++ DLL oracle.

## Benchmark Philosophy

Correctness and speed are separate questions. Use `make compare` first to prove
the DLLs return the same answers through the Windows ABI. Use `make bench-compare`
only after parity passes.

For performance work, default to this loop:

1. Prove C++ vs `rust-base` parity with `make compare`.
2. Establish the noise floor with an A/A run where `rust-candidate` points at
   the same DLL as `rust-base`.
3. Validate the real candidate with `make compare RUST_CANDIDATE_DLL=...`.
4. Benchmark C++ vs `rust-base` vs `rust-candidate` using the same case set,
   budget, repeat count, warmup count, and thread count.
5. Treat small wins skeptically unless they clear the A/A drift and the reported
   coefficient of variation (`cv`).

Keep `BENCH_THREADS=1` for implementation comparisons. That isolates the hot
path and avoids confusing algorithm/runtime differences with scheduling noise.
Raise threads only when you are specifically profiling parallel search behavior.

## Canonical Commands

Prettiest, most complete C++ vs Rust-base dashboard:

```bash
make bench-compare \
  BENCH_CASE=all \
  BENCH_BUDGET=1000000 \
  BENCH_REPEAT=7 \
  BENCH_WARMUP=2 \
  BENCH_THREADS=1 \
  BENCH_FORMAT=pretty \
  BENCH_COLOR=always
```

Prettiest, most complete V2 baseline vs V3 candidate comparison:

```bash
make build-rust-v2 build-rust-v3
make compare \
  RUST_BASE_DLL=target/rust-v2/Immolate.dll \
  RUST_CANDIDATE_DLL=target/rust-v3/Immolate.dll

make bench-compare \
  RUST_BASE_DLL=target/rust-v2/Immolate.dll \
  RUST_CANDIDATE_DLL=target/rust-v3/Immolate.dll \
  BENCH_CASE=all \
  BENCH_BUDGET=1000000 \
  BENCH_REPEAT=7 \
  BENCH_WARMUP=2 \
  BENCH_THREADS=1 \
  BENCH_FORMAT=pretty \
  BENCH_COLOR=always \
  BENCH_CANDIDATE_MIN_RATIO=10
```

Prettiest, most complete legacy V1 vs V3 comparison:

```bash
make build-rust-v1 build-rust-v3
make compare \
  RUST_BASE_DLL=target/rust-v1/Immolate.dll \
  RUST_CANDIDATE_DLL=target/rust-v3/Immolate.dll

make bench-compare \
  RUST_BASE_DLL=target/rust-v1/Immolate.dll \
  RUST_CANDIDATE_DLL=target/rust-v3/Immolate.dll \
  BENCH_CASE=all \
  BENCH_BUDGET=1000000 \
  BENCH_REPEAT=7 \
  BENCH_WARMUP=2 \
  BENCH_THREADS=1 \
  BENCH_FORMAT=pretty \
  BENCH_COLOR=always
```

Run those directly in an interactive terminal for the live status animation.
`BENCH_COLOR=always` is useful for terminal recordings or tools that preserve
ANSI color. Use `BENCH_COLOR=auto` if you only care about local terminal output.

Run the Rust DLL benchmark only:

```bash
make bench BENCH_BUDGET=100000 BENCH_REPEAT=3 BENCH_CASE=all
```

Run the C++ vs Rust-base benchmark dashboard:

```bash
make bench-compare BENCH_BUDGET=100000 BENCH_REPEAT=3 BENCH_CASE=all
```

Run C++ vs Rust-base vs a future Rust-candidate DLL:

```bash
make bench-compare RUST_CANDIDATE_DLL=target/rust-v3/Immolate.dll \
  BENCH_BUDGET=100000 BENCH_REPEAT=3 BENCH_CASE=all
```

Run an A/A noise-floor check by intentionally pointing `rust-candidate` at the
same DLL as `rust-base`:

```bash
make bench-compare RUST_CANDIDATE_DLL=target/rust/Immolate.dll \
  BENCH_CASE=pack-miss BENCH_BUDGET=100000 BENCH_REPEAT=7 BENCH_WARMUP=1 \
  BENCH_FORMAT=tsv BENCH_COLOR=never BENCH_MIN_RATIO=0.1
```

That command is intentionally not a candidate proof. It measures how much drift
appears when the same Rust DLL is loaded as both `rust-base` and
`rust-candidate`.

Run one profiling group:

```bash
make bench-compare BENCH_CASE=jokers BENCH_BUDGET=100000 BENCH_REPEAT=5
```

Run script-friendly TSV output:

```bash
make bench-compare BENCH_FORMAT=tsv BENCH_COLOR=never \
  BENCH_BUDGET=100000 BENCH_REPEAT=3 BENCH_CASE=all
```

Run a thorough TSV archive for later parsing:

```bash
make bench-compare \
  RUST_CANDIDATE_DLL=/path/to/candidate/Immolate.dll \
  BENCH_CASE=all \
  BENCH_BUDGET=1000000 \
  BENCH_REPEAT=7 \
  BENCH_WARMUP=2 \
  BENCH_THREADS=1 \
  BENCH_FORMAT=tsv \
  BENCH_COLOR=never > bench.tsv
```

Run the full Rust validation gate, including a small benchmark smoke test:

```bash
make check-rust
```

Use `bench-compare` for performance claims. It builds the C++ oracle and the
current Rust DLL, labels the current Rust implementation as `rust-base`, runs
both through the same Windows ABI harness under Wine, and fails if `rust-base`
drops below the configured C++ throughput threshold. If `RUST_CANDIDATE_DLL` is
set, the candidate is measured as a third competitor but does not replace the
`rust-base` gate.

## Latest Validated Local Run

The latest local V2 baseline vs V3 candidate proof on June 9, 2026 used:

```bash
make build-rust-v2 build-rust-v3
make compare \
  RUST_BASE_DLL=/tmp/brainstorm-v3-proof/rust-v2-baseline.dll \
  RUST_CANDIDATE_DLL=/tmp/brainstorm-v3-proof/rust-v3-candidate.dll

make bench-compare \
  RUST_BASE_DLL=/tmp/brainstorm-v3-proof/rust-v2-baseline.dll \
  RUST_CANDIDATE_DLL=/tmp/brainstorm-v3-proof/rust-v3-candidate.dll \
  BENCH_CASE=all BENCH_BUDGET=1000000 BENCH_REPEAT=7 BENCH_WARMUP=2 \
  BENCH_THREADS=1 BENCH_FORMAT=tsv BENCH_COLOR=never BENCH_MIN_RATIO=0.1 \
  BENCH_CANDIDATE_MIN_RATIO=10 BENCH_CANDIDATE_MIN_SCAN_PCT=0.95
```

The candidate aggregate passed:

| comparison | ratio | target | cases |
| --- | ---: | ---: | ---: |
| V3/V2 non-hit full-budget geometric mean | 58.418x | 10.000x | 3 |

The benchmark summary results also matched across C++, V2, and V3 for the same
run. This proves the harness-defined aggregate target; it does not mean every
individual hit or mixed case is 10x faster.

The latest local A/A sanity run on June 9, 2026 used:

```bash
make bench-compare RUST_CANDIDATE_DLL=target/rust/Immolate.dll \
  BENCH_BUDGET=100000 BENCH_REPEAT=7 BENCH_WARMUP=1 \
  BENCH_CASE=pack-miss BENCH_FORMAT=tsv BENCH_COLOR=never BENCH_MIN_RATIO=0.1
```

The important summary was:

| implementation | seeds/s | ns/seed | cv |
| --- | ---: | ---: | ---: |
| C++ | 4,804,089 | 208.156 | 0.455% |
| rust-base | 5,833,785 | 171.415 | 0.461% |
| rust-candidate | 5,789,483 | 172.727 | 0.384% |

Ratios from that run:

- `rust-base`/C++ was `1.214x`.
- `rust-candidate`/C++ was `1.205x`.
- `rust-candidate`/`rust-base` was `0.992x`.

Because `rust-candidate` was intentionally the same DLL as `rust-base`, the
`0.992x` candidate/base result is measured noise, not a real regression. Use
that as a rough floor: on this fixture and machine, a sub-1% candidate delta is
not enough evidence to call a win or loss.

`make check-rust` also passed after the benchmark overhaul. Its embedded
benchmark is only a smoke test (`BENCH_BUDGET=1000 BENCH_REPEAT=1` by default
in the Makefile target), not an optimization-quality measurement.

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

The Makefile exposes these variables:

- `BENCH_CASE=all|baseline|tags|vouchers|packs|jokers|souls|deck|CASE_NAME`
- `BENCH_BUDGET=1000000`
- `BENCH_REPEAT=5`
- `BENCH_WARMUP=1`
- `BENCH_THREADS=1`
- `BENCH_MIN_RATIO=0.8`
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

Keep `BENCH_THREADS=1` as the default for implementation comparisons. Increase
it only when evaluating parallel search behavior specifically.

Set `BENCH_CANDIDATE_MIN_RATIO` to make `bench-compare` fail unless the
candidate/base non-hit full-budget geometric mean reaches that target. The
aggregate includes only non-hit fixtures where both Rust implementations report
`<null>` and `scan_pct >= BENCH_CANDIDATE_MIN_SCAN_PCT`, which keeps early-hit
and late-hit overhead cases out of the headline throughput number.

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
(`ok`, `below-target`, or `regression`). The relation is stored in the `result`
field as semicolon-delimited details such as `ratio`, `target_ratio`, `lhs`,
`rhs`, `lhs_sps`, `rhs_sps`, `lhs_ms`, and `rhs_ms`.

Only `rust-base-vs-cpp` controls the benchmark process exit status through
`BENCH_MIN_RATIO`. Candidate per-case rows are informational. The aggregate row
controls exit status only when `BENCH_CANDIDATE_MIN_RATIO` is greater than zero.
A `rust-candidate-vs-base` row can report `below-target` during an A/A run and
still exit successfully because that row is documenting drift, not enforcing a
release gate.

## Rust Candidate Workflow

V3 is the default Rust DLL built by `make build-rust`. The preserved V2 baseline
DLL is built by `make build-rust-v2`, which writes
`target/rust-v2/Immolate.dll`. The legacy V1 DLL remains available for
comparison with `make build-rust-v1`, which writes
`target/rust-v1/Immolate.dll`.

Set `RUST_BASE_DLL=target/rust-v2/Immolate.dll` when proving V3 against the
previous fastest Rust implementation. Set
`RUST_CANDIDATE_DLL=target/rust-v3/Immolate.dll` for the in-repo V3 candidate,
or point it at a separate DLL from another optimization branch, alternate build
profile, or experimental worktree.

Before trusting candidate performance, validate candidate functionality too:

```bash
make build-rust-v2 build-rust-v3
make compare \
  RUST_BASE_DLL=target/rust-v2/Immolate.dll \
  RUST_CANDIDATE_DLL=target/rust-v3/Immolate.dll
```

That compares C++, V2 as `rust-base`, and V3 as `rust-candidate` across the functional
fixture suite, parser matrix, threaded search fixture, wraparound fixture, and
alloc/free stress check. Only benchmark a candidate after this passes.

For a serious optimization comparison, run the same command twice: once as A/A
with `RUST_CANDIDATE_DLL=target/rust/Immolate.dll`, then once with the real
candidate. If the real candidate only moves by about the same amount as the A/A
run, keep investigating before claiming an improvement.

To prove a 10x candidate/base improvement on non-hit full-budget fixtures:

```bash
make build-rust-v2 build-rust-v3

make bench-compare \
  RUST_BASE_DLL=target/rust-v2/Immolate.dll \
  RUST_CANDIDATE_DLL=target/rust-v3/Immolate.dll \
  BENCH_CASE=all \
  BENCH_BUDGET=1000000 \
  BENCH_REPEAT=7 \
  BENCH_WARMUP=2 \
  BENCH_THREADS=1 \
  BENCH_FORMAT=tsv \
  BENCH_COLOR=never \
  BENCH_MIN_RATIO=0.1 \
  BENCH_CANDIDATE_MIN_RATIO=10 \
  BENCH_CANDIDATE_MIN_SCAN_PCT=0.95
```

## Optional Native Rust-Only Benchmark

For quick Linux-side profiling of the Rust core without the Windows DLL ABI,
use the native helper. It defaults to V3, matching the current exported Rust
DLL path. Pass `--engine v2` for the preserved V2 source core, `--engine base`
for the legacy V1 source core, or `--engine both` for an in-process source-level
comparison.

```bash
cargo run --manifest-path Immolate/Rust/Cargo.toml --release --bin brainstorm_bench -- \
  --case all --budget 1000000 --threads 1 --repeat 5
```

Do not use this helper to claim drop-in DLL parity or C++ superiority. It avoids
Wine and the C ABI, so it is best for inner-loop profiling only.

## Agent Workflow

Before changing hot-path code:

1. Run `make compare` and make sure C++ and `rust-base` are in parity.
2. Run an A/A benchmark with `RUST_CANDIDATE_DLL=target/rust/Immolate.dll` and
   save the candidate/base ratio plus CV values, or set `RUST_BASE_DLL` to an
   explicit legacy/artifact DLL when comparing against an older baseline.
3. Run the prettiest complete dashboard with `BENCH_CASE=all`,
   `BENCH_BUDGET=1000000`, `BENCH_REPEAT=7`, and `BENCH_WARMUP=2`.
4. Identify the weakest Rust groups from "Rust-base Behind C++",
   "Group Speedups", and any "High Variance" warnings.
5. Make the smallest performance-oriented change that preserves parity.
6. Build the experiment as a candidate DLL and run
   `make compare RUST_BASE_DLL=target/rust-v2/Immolate.dll RUST_CANDIDATE_DLL=target/rust-v3/Immolate.dll`.
7. Rerun the exact same `bench-compare` command with `RUST_CANDIDATE_DLL=...`.
8. Keep the change only if the relevant fixture improves beyond the A/A noise or
   the tradeoff is explicitly justified.
9. Finish with `make check-rust` before release or deployment work.

When adding a benchmark fixture, update `Immolate/Rust/src/bench_cases.rs`. Both
`Immolate/Rust/src/bin/immolate_dll_harness.rs` and `Immolate/Rust/src/bin/brainstorm_bench.rs`
read from that shared catalog.
