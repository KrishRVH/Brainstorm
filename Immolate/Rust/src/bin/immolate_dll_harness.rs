#![allow(unsafe_code)]

#[cfg(not(windows))]
fn main() {
    eprintln!("immolate_dll_harness must be built for Windows and run under Windows or Wine");
    std::process::exit(2);
}

#[cfg(windows)]
fn main() {
    windows_harness::main();
}

#[cfg(windows)]
#[path = "../bench_cases.rs"]
mod bench_cases;

#[cfg(windows)]
mod windows_harness {
    use std::cmp::Ordering as CmpOrdering;
    use std::env;
    use std::ffi::{CStr, CString, OsStr};
    use std::io::{self, IsTerminal, Write};
    use std::os::raw::{c_char, c_double, c_int, c_longlong, c_void};
    use std::os::windows::ffi::OsStrExt;
    use std::ptr;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
    use std::thread;
    use std::time::{Duration, Instant};

    use immolate::seed::Seed;

    use super::bench_cases::{self as bench, BenchCase, BenchGroup, BenchShape};

    type HModule = *mut c_void;
    type FarProc = *mut c_void;
    type BrainstormSearch = unsafe extern "C" fn(
        *const c_char,
        *const c_char,
        *const c_char,
        *const c_char,
        *const c_char,
        *const c_char,
        *const c_char,
        c_double,
        bool,
        bool,
        *const c_char,
        bool,
        bool,
        c_int,
        c_double,
        c_longlong,
        c_int,
    ) -> *const c_char;
    type FreeResult = unsafe extern "C" fn(*const c_char);

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryW(path: *const u16) -> HModule;
        fn GetProcAddress(module: HModule, name: *const c_char) -> FarProc;
        fn FreeLibrary(module: HModule) -> i32;
    }

    #[derive(Clone)]
    struct Case {
        name: &'static str,
        group: BenchGroup,
        shape: BenchShape,
        note: &'static str,
        seed_start: Option<&'static str>,
        voucher: Option<&'static str>,
        pack: Option<&'static str>,
        tag1: Option<&'static str>,
        tag2: Option<&'static str>,
        joker: Option<&'static str>,
        joker_location: Option<&'static str>,
        souls: f64,
        observatory: bool,
        perkeo: bool,
        deck: Option<&'static str>,
        erratic: bool,
        no_faces: bool,
        min_face_cards: i32,
        suit_ratio: f64,
        num_seeds: i64,
        threads: i32,
    }

    struct Dll {
        handle: HModule,
        search: BrainstormSearch,
        free_result: FreeResult,
    }

    impl Dll {
        fn load(path: &str) -> Result<Self, String> {
            let mut wide: Vec<u16> = OsStr::new(path).encode_wide().collect();
            wide.push(0);
            let handle = unsafe { LoadLibraryW(wide.as_ptr()) };
            if handle.is_null() {
                return Err(format!("failed to load DLL: {path}"));
            }

            let search_name = CString::new("brainstorm_search").map_err(|err| format!("{err}"))?;
            let free_name = CString::new("free_result").map_err(|err| format!("{err}"))?;
            let search_ptr = unsafe { GetProcAddress(handle, search_name.as_ptr()) };
            let free_ptr = unsafe { GetProcAddress(handle, free_name.as_ptr()) };
            if search_ptr.is_null() || free_ptr.is_null() {
                unsafe {
                    FreeLibrary(handle);
                }
                return Err(format!(
                    "missing required exports in {path}: brainstorm_search/free_result",
                ));
            }

            Ok(Self {
                handle,
                search: unsafe { std::mem::transmute::<FarProc, BrainstormSearch>(search_ptr) },
                free_result: unsafe { std::mem::transmute::<FarProc, FreeResult>(free_ptr) },
            })
        }

        fn run(&self, case: &Case) -> Result<Option<String>, String> {
            let seed_start = CArg::new(case.seed_start)?;
            let voucher = CArg::new(case.voucher)?;
            let pack = CArg::new(case.pack)?;
            let tag1 = CArg::new(case.tag1)?;
            let tag2 = CArg::new(case.tag2)?;
            let joker = CArg::new(case.joker)?;
            let joker_location = CArg::new(case.joker_location)?;
            let deck = CArg::new(case.deck)?;

            let result = unsafe {
                (self.search)(
                    seed_start.as_ptr(),
                    voucher.as_ptr(),
                    pack.as_ptr(),
                    tag1.as_ptr(),
                    tag2.as_ptr(),
                    joker.as_ptr(),
                    joker_location.as_ptr(),
                    case.souls,
                    case.observatory,
                    case.perkeo,
                    deck.as_ptr(),
                    case.erratic,
                    case.no_faces,
                    case.min_face_cards,
                    case.suit_ratio,
                    case.num_seeds,
                    case.threads,
                )
            };
            if result.is_null() {
                return Ok(None);
            }
            let out = unsafe { CStr::from_ptr(result) }
                .to_string_lossy()
                .into_owned();
            unsafe {
                (self.free_result)(result);
            }
            Ok(Some(out))
        }
    }

    impl Drop for Dll {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe {
                    FreeLibrary(self.handle);
                }
            }
        }
    }

    struct CArg {
        value: Option<CString>,
    }

    impl CArg {
        fn new(value: Option<&str>) -> Result<Self, String> {
            value
                .map(|value| CString::new(value).map_err(|err| format!("{err}")))
                .transpose()
                .map(|value| Self { value })
        }

        fn as_ptr(&self) -> *const c_char {
            self.value
                .as_ref()
                .map_or(ptr::null(), |value| value.as_ptr())
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum OutputFormat {
        Pretty,
        Tsv,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum ColorMode {
        Auto,
        Always,
        Never,
    }

    #[derive(Clone, Copy, Debug)]
    struct OutputOptions {
        format: OutputFormat,
        color: ColorMode,
    }

    impl Default for OutputOptions {
        fn default() -> Self {
            Self {
                format: OutputFormat::Pretty,
                color: ColorMode::Auto,
            }
        }
    }

    impl OutputOptions {
        fn use_color(self) -> bool {
            match self.color {
                ColorMode::Always => true,
                ColorMode::Never => false,
                ColorMode::Auto => io::stdout().is_terminal(),
            }
        }

        fn animate(self) -> bool {
            self.format == OutputFormat::Pretty && io::stdout().is_terminal()
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct BenchSettings<'a> {
        selected_case: &'a str,
        budget: i64,
        threads: i32,
        repeat: usize,
        warmup: usize,
        output: OutputOptions,
    }

    enum Command {
        Compare {
            cpp: String,
            rust_base: String,
            rust_candidate: Option<String>,
            threads: i32,
        },
        Bench {
            dll: String,
            case: String,
            budget: i64,
            threads: i32,
            repeat: usize,
            warmup: usize,
            output: OutputOptions,
        },
        BenchCompare {
            cpp: String,
            rust_base: String,
            rust_candidate: Option<String>,
            case: String,
            budget: i64,
            threads: i32,
            repeat: usize,
            warmup: usize,
            min_ratio: f64,
            candidate_min_ratio: f64,
            candidate_min_scan_pct: f64,
            output: OutputOptions,
        },
    }

    pub fn main() {
        match parse_command(env::args().skip(1).collect()) {
            Ok(Command::Compare {
                cpp,
                rust_base,
                rust_candidate,
                threads,
            }) => {
                if let Err(err) = compare(&cpp, &rust_base, rust_candidate.as_deref(), threads) {
                    eprintln!("{err}");
                    std::process::exit(1);
                }
            },
            Ok(Command::Bench {
                dll,
                case,
                budget,
                threads,
                repeat,
                warmup,
                output,
            }) => {
                let settings = BenchSettings {
                    selected_case: &case,
                    budget,
                    threads,
                    repeat,
                    warmup,
                    output,
                };
                if let Err(err) = bench(&dll, settings) {
                    eprintln!("{err}");
                    std::process::exit(1);
                }
            },
            Ok(Command::BenchCompare {
                cpp,
                rust_base,
                rust_candidate,
                case,
                budget,
                threads,
                repeat,
                warmup,
                min_ratio,
                candidate_min_ratio,
                candidate_min_scan_pct,
                output,
            }) => {
                let settings = BenchSettings {
                    selected_case: &case,
                    budget,
                    threads,
                    repeat,
                    warmup,
                    output,
                };
                if let Err(err) = bench_compare(
                    &cpp,
                    &rust_base,
                    rust_candidate.as_deref(),
                    settings,
                    min_ratio,
                    candidate_min_ratio,
                    candidate_min_scan_pct,
                ) {
                    eprintln!("{err}");
                    std::process::exit(1);
                }
            },
            Err(err) => {
                eprintln!("{err}");
                eprintln!(
                    "usage:\n  immolate_dll_harness compare --cpp PATH --rust-base PATH [--rust-candidate PATH] [--threads N]\n  immolate_dll_harness bench --dll PATH [--case all|GROUP|NAME] [--budget N] [--threads N] [--repeat N] [--warmup N] [--format pretty|tsv] [--color auto|always|never]\n  immolate_dll_harness bench-compare --cpp PATH --rust-base PATH [--rust-candidate PATH] [--case all|GROUP|NAME] [--budget N] [--threads N] [--repeat N] [--warmup N] [--min-ratio N] [--candidate-min-ratio N] [--candidate-min-scan-pct N] [--format pretty|tsv] [--color auto|always|never]"
                );
                std::process::exit(2);
            },
        }
    }

    fn compare(
        cpp_path: &str,
        rust_base_path: &str,
        rust_candidate_path: Option<&str>,
        threads: i32,
    ) -> Result<(), String> {
        let cpp = Dll::load(cpp_path)?;
        let rust_base = Dll::load(rust_base_path)?;
        let rust_candidate = rust_candidate_path.map(Dll::load).transpose()?;
        let mut failed = false;
        if rust_candidate.is_some() {
            println!("status\tcase\tcpp\trust-base\trust-candidate");
        } else {
            println!("status\tcase\tcpp\trust-base");
        }
        for mut case in compare_cases() {
            if threads != 1 {
                case.threads = threads;
            }
            let cpp_result = cpp.run(&case)?;
            let rust_base_result = rust_base.run(&case)?;
            let rust_candidate_result = rust_candidate
                .as_ref()
                .map(|dll| dll.run(&case))
                .transpose()?;
            let status = if cpp_result == rust_base_result
                && rust_candidate_result
                    .as_ref()
                    .is_none_or(|candidate| candidate == &cpp_result)
            {
                "ok"
            } else {
                failed = true;
                "mismatch"
            };
            if let Some(candidate_result) = rust_candidate_result.as_ref() {
                println!(
                    "{status}\t{}\t{}\t{}\t{}",
                    case.name,
                    display_result(cpp_result.as_deref()),
                    display_result(rust_base_result.as_deref()),
                    display_result(candidate_result.as_deref()),
                );
            } else {
                println!(
                    "{status}\t{}\t{}\t{}",
                    case.name,
                    display_result(cpp_result.as_deref()),
                    display_result(rust_base_result.as_deref()),
                );
            }
        }
        let mut stress_dlls = vec![("cpp", &cpp), ("rust-base", &rust_base)];
        if let Some(candidate) = rust_candidate.as_ref() {
            stress_dlls.push(("rust-candidate", candidate));
        }
        for (name, dll) in stress_dlls {
            let stress = repeated_alloc_free(dll, 100)?;
            if stress {
                println!("ok\talloc_free_stress_100_{name}\t1\t1");
            } else {
                failed = true;
                println!("mismatch\talloc_free_stress_100_{name}\t1\t<null>");
            }
        }
        if failed {
            Err("DLL compare failed".to_owned())
        } else {
            Ok(())
        }
    }

    fn bench(dll_path: &str, settings: BenchSettings<'_>) -> Result<(), String> {
        if settings.budget <= 0 {
            return Err("--budget must be positive".to_owned());
        }
        if settings.repeat == 0 {
            return Err("--repeat must be positive".to_owned());
        }
        let dll = Dll::load(dll_path)?;
        let cases =
            selected_bench_cases(settings.selected_case, settings.budget, settings.threads)?;

        if settings.output.format == OutputFormat::Tsv {
            print_tsv_header();
        } else {
            print_run_header("Brainstorm DLL Benchmark", settings, cases.len());
        }

        let mut summaries = Vec::with_capacity(cases.len());
        for case in &cases {
            summaries.push(measure_bench_case(
                &dll,
                case,
                settings.repeat,
                settings.warmup,
                "dll",
                settings.output,
            )?);
        }
        if settings.output.format == OutputFormat::Pretty {
            print_single_bench_report(&summaries, settings.output);
        }
        Ok(())
    }

    fn bench_compare(
        cpp_path: &str,
        rust_base_path: &str,
        rust_candidate_path: Option<&str>,
        settings: BenchSettings<'_>,
        min_ratio: f64,
        candidate_min_ratio: f64,
        candidate_min_scan_pct: f64,
    ) -> Result<(), String> {
        if settings.budget <= 0 {
            return Err("--budget must be positive".to_owned());
        }
        if settings.repeat == 0 {
            return Err("--repeat must be positive".to_owned());
        }
        if min_ratio <= 0.0 {
            return Err("--min-ratio must be positive".to_owned());
        }
        if candidate_min_ratio < 0.0 {
            return Err("--candidate-min-ratio must be non-negative".to_owned());
        }
        if candidate_min_ratio > 0.0 && rust_candidate_path.is_none() {
            return Err("--candidate-min-ratio requires --rust-candidate".to_owned());
        }
        if !(0.0..=1.0).contains(&candidate_min_scan_pct) {
            return Err("--candidate-min-scan-pct must be between 0 and 1".to_owned());
        }
        let cpp = Dll::load(cpp_path)?;
        let rust_base = Dll::load(rust_base_path)?;
        let rust_candidate = rust_candidate_path.map(Dll::load).transpose()?;
        let cases =
            selected_bench_cases(settings.selected_case, settings.budget, settings.threads)?;
        if settings.output.format == OutputFormat::Tsv {
            print_tsv_header();
        } else {
            print_run_header(
                if rust_candidate.is_some() {
                    "Brainstorm DLL Benchmark: C++ vs Rust-base vs Rust-candidate"
                } else {
                    "Brainstorm DLL Benchmark: C++ vs Rust-base"
                },
                settings,
                cases.len(),
            );
        }

        let mut failed = false;
        let mut comparisons = Vec::with_capacity(cases.len());
        for case in &cases {
            let cpp_summary = measure_bench_case(
                &cpp,
                case,
                settings.repeat,
                settings.warmup,
                "cpp",
                settings.output,
            )?;
            let rust_base_summary = measure_bench_case(
                &rust_base,
                case,
                settings.repeat,
                settings.warmup,
                "rust-base",
                settings.output,
            )?;
            let rust_candidate_summary = rust_candidate
                .as_ref()
                .map(|dll| {
                    measure_bench_case(
                        dll,
                        case,
                        settings.repeat,
                        settings.warmup,
                        "rust-candidate",
                        settings.output,
                    )
                })
                .transpose()?;
            let comparison = BenchComparison {
                cpp: cpp_summary,
                rust_base: rust_base_summary,
                rust_candidate: rust_candidate_summary,
            };
            if !comparison.base_matches_cpp() || comparison.base_vs_cpp_ratio() < min_ratio {
                failed = true;
            }
            if !comparison.candidate_matches_cpp() {
                failed = true;
            }
            if settings.output.format == OutputFormat::Tsv {
                print_tsv_compare(&comparison, min_ratio);
            }
            comparisons.push(comparison);
        }
        if settings.output.format == OutputFormat::Pretty {
            print_compare_report(&comparisons, min_ratio, settings.output);
        }
        if rust_candidate.is_some() {
            let aggregate = candidate_full_budget_gmean(&comparisons, candidate_min_scan_pct);
            if settings.output.format == OutputFormat::Tsv {
                print_tsv_candidate_aggregate(
                    aggregate.as_ref(),
                    settings,
                    candidate_min_ratio,
                    candidate_min_scan_pct,
                );
            } else {
                print_candidate_aggregate(
                    aggregate.as_ref(),
                    candidate_min_ratio,
                    candidate_min_scan_pct,
                    settings.output,
                );
            }
            if candidate_min_ratio > 0.0
                && aggregate
                    .as_ref()
                    .is_none_or(|aggregate| aggregate.ratio < candidate_min_ratio)
            {
                failed = true;
            }
        }
        if failed {
            Err("benchmark regression threshold failed".to_owned())
        } else {
            Ok(())
        }
    }

    struct BenchRun {
        run: usize,
        elapsed: Duration,
        scanned: i64,
        seeds_per_sec: f64,
        ns_per_seed: f64,
        result: String,
    }

    struct BenchSummary {
        implementation: &'static str,
        case_name: &'static str,
        group: BenchGroup,
        shape: BenchShape,
        note: &'static str,
        budget: i64,
        threads: i32,
        repeat: usize,
        runs: Vec<BenchRun>,
        mean_elapsed: Duration,
        min_elapsed: Duration,
        max_elapsed: Duration,
        p50_elapsed: Duration,
        p95_elapsed: Duration,
        p99_elapsed: Duration,
        stdev_elapsed: Duration,
        coefficient_variation: f64,
        mean_scanned: f64,
        scanned_pct: f64,
        seeds_per_sec: f64,
        ns_per_seed: f64,
        result: String,
    }

    struct BenchComparison {
        cpp: BenchSummary,
        rust_base: BenchSummary,
        rust_candidate: Option<BenchSummary>,
    }

    struct CandidateAggregate {
        ratio: f64,
        cases: usize,
    }

    impl BenchComparison {
        fn base_vs_cpp_ratio(&self) -> f64 {
            self.rust_base.seeds_per_sec / self.cpp.seeds_per_sec
        }

        fn candidate_vs_cpp_ratio(&self) -> Option<f64> {
            self.rust_candidate
                .as_ref()
                .map(|candidate| candidate.seeds_per_sec / self.cpp.seeds_per_sec)
        }

        fn candidate_vs_base_ratio(&self) -> Option<f64> {
            self.rust_candidate
                .as_ref()
                .map(|candidate| candidate.seeds_per_sec / self.rust_base.seeds_per_sec)
        }

        fn candidate(&self) -> Option<&BenchSummary> {
            self.rust_candidate.as_ref()
        }

        fn base_matches_cpp(&self) -> bool {
            self.rust_base.result == self.cpp.result
        }

        fn candidate_matches_cpp(&self) -> bool {
            self.rust_candidate
                .as_ref()
                .is_none_or(|candidate| candidate.result == self.cpp.result)
        }
    }

    fn candidate_full_budget_gmean(
        comparisons: &[BenchComparison],
        min_scan_pct: f64,
    ) -> Option<CandidateAggregate> {
        let mut cases = 0;
        let mut log_sum = 0.0;
        for comparison in comparisons {
            let Some(candidate) = comparison.candidate() else {
                continue;
            };
            if comparison.rust_base.shape == BenchShape::Hit {
                continue;
            }
            if comparison.rust_base.result != "<null>" || candidate.result != "<null>" {
                continue;
            }
            if comparison.rust_base.scanned_pct < min_scan_pct
                || candidate.scanned_pct < min_scan_pct
            {
                continue;
            }
            let ratio = candidate.seeds_per_sec / comparison.rust_base.seeds_per_sec;
            if ratio > 0.0 && ratio.is_finite() {
                cases += 1;
                log_sum += ratio.ln();
            }
        }
        (cases > 0).then(|| CandidateAggregate {
            ratio: (log_sum / cases as f64).exp(),
            cases,
        })
    }

    fn measure_bench_case(
        dll: &Dll,
        case: &Case,
        repeat: usize,
        warmup: usize,
        implementation: &'static str,
        output: OutputOptions,
    ) -> Result<BenchSummary, String> {
        run_warmups(dll, case, warmup, implementation, output)?;
        let mut runs = Vec::with_capacity(repeat);
        let mut scanned_counts = Vec::with_capacity(repeat);
        for run in 1..=repeat {
            let status = format!(
                "{implementation} {:<18} run {run}/{repeat}  budget {}  threads {}",
                case.name,
                format_integer(case.num_seeds),
                case.threads,
            );
            let ticker = RunTicker::start(output.animate(), status, output.use_color());
            let started = Instant::now();
            let result = dll.run(case);
            let elapsed = started.elapsed();
            ticker.finish();
            let result = result?;
            let scanned = scanned_count(case, result.as_deref());
            let elapsed_secs = elapsed.as_secs_f64();
            let seeds_per_sec = scanned as f64 / elapsed_secs;
            let ns_per_seed = elapsed_secs * 1_000_000_000.0 / scanned as f64;
            scanned_counts.push(scanned);
            let bench_run = BenchRun {
                run,
                elapsed,
                scanned,
                seeds_per_sec,
                ns_per_seed,
                result: display_result(result.as_deref()).to_owned(),
            };
            if output.format == OutputFormat::Tsv {
                print_tsv_run(implementation, case, &bench_run);
            }
            runs.push(bench_run);
        }
        let mut durations: Vec<_> = runs.iter().map(|run| run.elapsed).collect();
        durations.sort_by(compare_duration);
        let mean_elapsed = mean_duration(&durations);
        let min_elapsed = durations[0];
        let max_elapsed = durations[durations.len() - 1];
        let p50_elapsed = percentile(&durations, 0.50);
        let p95_elapsed = percentile(&durations, 0.95);
        let p99_elapsed = percentile(&durations, 0.99);
        let stdev_elapsed = stdev_duration(&durations, mean_elapsed);
        let coefficient_variation = stdev_elapsed.as_secs_f64() / mean_elapsed.as_secs_f64();
        let mean_scanned = scanned_counts
            .iter()
            .map(|value| *value as f64)
            .sum::<f64>()
            / repeat as f64;
        let seeds_per_sec = mean_scanned / mean_elapsed.as_secs_f64();
        let ns_per_seed = mean_elapsed.as_secs_f64() * 1_000_000_000.0 / mean_scanned;
        let scanned_pct = mean_scanned / case.num_seeds as f64;
        let result = runs
            .last()
            .map_or_else(|| "<none>".to_owned(), |run| run.result.clone());
        let summary = BenchSummary {
            implementation,
            case_name: case.name,
            group: case.group,
            shape: case.shape,
            note: case.note,
            budget: case.num_seeds,
            threads: case.threads,
            repeat,
            runs,
            mean_elapsed,
            min_elapsed,
            max_elapsed,
            p50_elapsed,
            p95_elapsed,
            p99_elapsed,
            stdev_elapsed,
            coefficient_variation,
            mean_scanned,
            scanned_pct,
            seeds_per_sec,
            ns_per_seed,
            result,
        };
        if output.format == OutputFormat::Tsv {
            print_tsv_summary(&summary);
        }
        Ok(summary)
    }

    fn run_warmups(
        dll: &Dll,
        case: &Case,
        warmup: usize,
        implementation: &str,
        output: OutputOptions,
    ) -> Result<(), String> {
        for run in 1..=warmup {
            let status = format!(
                "{implementation} {:<18} warmup {run}/{warmup}  budget {}  threads {}",
                case.name,
                format_integer(case.num_seeds),
                case.threads,
            );
            let ticker = RunTicker::start(output.animate(), status, output.use_color());
            let result = dll.run(case);
            ticker.finish();
            result?;
        }
        Ok(())
    }

    fn parse_command(args: Vec<String>) -> Result<Command, String> {
        let Some(mode) = args.first() else {
            return Err("missing command".to_owned());
        };
        match mode.as_str() {
            "compare" => {
                let mut cpp = None;
                let mut rust_base = None;
                let mut rust_candidate = None;
                let mut threads = 1;
                parse_flags(&args[1..], |flag, value| match flag {
                    "--cpp" => {
                        cpp = Some(value.to_owned());
                        Ok(())
                    },
                    "--rust" | "--rust-base" => {
                        rust_base = Some(value.to_owned());
                        Ok(())
                    },
                    "--rust-candidate" => {
                        rust_candidate = Some(value.to_owned());
                        Ok(())
                    },
                    "--threads" => {
                        threads = parse_value(value, "--threads")?;
                        Ok(())
                    },
                    _ => Err(format!("unknown compare flag: {flag}")),
                })?;
                Ok(Command::Compare {
                    cpp: cpp.ok_or_else(|| "missing --cpp".to_owned())?,
                    rust_base: rust_base.ok_or_else(|| "missing --rust-base".to_owned())?,
                    rust_candidate,
                    threads,
                })
            },
            "bench" => {
                let mut dll = None;
                let mut case = "all".to_owned();
                let mut budget = 1_000_000;
                let mut threads = 1;
                let mut repeat = 5;
                let mut warmup = 1;
                let mut output = OutputOptions::default();
                parse_flags(&args[1..], |flag, value| match flag {
                    "--dll" => {
                        dll = Some(value.to_owned());
                        Ok(())
                    },
                    "--case" => {
                        value.clone_into(&mut case);
                        Ok(())
                    },
                    "--budget" => {
                        budget = parse_value(value, "--budget")?;
                        Ok(())
                    },
                    "--threads" => {
                        threads = parse_value(value, "--threads")?;
                        Ok(())
                    },
                    "--repeat" => {
                        repeat = parse_value(value, "--repeat")?;
                        Ok(())
                    },
                    "--warmup" => {
                        warmup = parse_value(value, "--warmup")?;
                        Ok(())
                    },
                    "--format" => {
                        output.format = parse_output_format(value)?;
                        Ok(())
                    },
                    "--color" => {
                        output.color = parse_color_mode(value)?;
                        Ok(())
                    },
                    _ => Err(format!("unknown bench flag: {flag}")),
                })?;
                Ok(Command::Bench {
                    dll: dll.ok_or_else(|| "missing --dll".to_owned())?,
                    case,
                    budget,
                    threads,
                    repeat,
                    warmup,
                    output,
                })
            },
            "bench-compare" => {
                let mut cpp = None;
                let mut rust_base = None;
                let mut rust_candidate = None;
                let mut case = "all".to_owned();
                let mut budget = 1_000_000;
                let mut threads = 1;
                let mut repeat = 5;
                let mut warmup = 1;
                let mut min_ratio = 0.8;
                let mut candidate_min_ratio = 0.0;
                let mut candidate_min_scan_pct = 0.95;
                let mut output = OutputOptions::default();
                parse_flags(&args[1..], |flag, value| match flag {
                    "--cpp" => {
                        cpp = Some(value.to_owned());
                        Ok(())
                    },
                    "--rust" | "--rust-base" => {
                        rust_base = Some(value.to_owned());
                        Ok(())
                    },
                    "--rust-candidate" => {
                        rust_candidate = Some(value.to_owned());
                        Ok(())
                    },
                    "--case" => {
                        value.clone_into(&mut case);
                        Ok(())
                    },
                    "--budget" => {
                        budget = parse_value(value, "--budget")?;
                        Ok(())
                    },
                    "--threads" => {
                        threads = parse_value(value, "--threads")?;
                        Ok(())
                    },
                    "--repeat" => {
                        repeat = parse_value(value, "--repeat")?;
                        Ok(())
                    },
                    "--warmup" => {
                        warmup = parse_value(value, "--warmup")?;
                        Ok(())
                    },
                    "--min-ratio" => {
                        min_ratio = parse_value(value, "--min-ratio")?;
                        Ok(())
                    },
                    "--candidate-min-ratio" => {
                        candidate_min_ratio = parse_value(value, "--candidate-min-ratio")?;
                        Ok(())
                    },
                    "--candidate-min-scan-pct" => {
                        candidate_min_scan_pct = parse_value(value, "--candidate-min-scan-pct")?;
                        Ok(())
                    },
                    "--format" => {
                        output.format = parse_output_format(value)?;
                        Ok(())
                    },
                    "--color" => {
                        output.color = parse_color_mode(value)?;
                        Ok(())
                    },
                    _ => Err(format!("unknown bench-compare flag: {flag}")),
                })?;
                Ok(Command::BenchCompare {
                    cpp: cpp.ok_or_else(|| "missing --cpp".to_owned())?,
                    rust_base: rust_base.ok_or_else(|| "missing --rust-base".to_owned())?,
                    rust_candidate,
                    case,
                    budget,
                    threads,
                    repeat,
                    warmup,
                    min_ratio,
                    candidate_min_ratio,
                    candidate_min_scan_pct,
                    output,
                })
            },
            _ => Err(format!("unknown command: {mode}")),
        }
    }

    fn parse_flags<F>(args: &[String], mut visit: F) -> Result<(), String>
    where
        F: FnMut(&str, &str) -> Result<(), String>,
    {
        let mut idx = 0;
        while idx < args.len() {
            let flag = &args[idx];
            let value = args
                .get(idx + 1)
                .ok_or_else(|| format!("missing value for {flag}"))?;
            visit(flag, value)?;
            idx += 2;
        }
        Ok(())
    }

    fn parse_value<T>(value: &str, flag: &str) -> Result<T, String>
    where
        T: std::str::FromStr,
    {
        value
            .parse::<T>()
            .map_err(|_| format!("invalid {flag}: {value}"))
    }

    fn parse_output_format(value: &str) -> Result<OutputFormat, String> {
        match value {
            "pretty" => Ok(OutputFormat::Pretty),
            "tsv" => Ok(OutputFormat::Tsv),
            _ => Err(format!("invalid --format: {value}")),
        }
    }

    fn parse_color_mode(value: &str) -> Result<ColorMode, String> {
        match value {
            "auto" => Ok(ColorMode::Auto),
            "always" => Ok(ColorMode::Always),
            "never" => Ok(ColorMode::Never),
            _ => Err(format!("invalid --color: {value}")),
        }
    }

    fn display_result(result: Option<&str>) -> &str {
        result.unwrap_or("<null>")
    }

    fn scanned_count(case: &Case, result: Option<&str>) -> i64 {
        let Some(result) = result else {
            return case.num_seeds;
        };
        if result.is_empty() {
            return 1;
        }
        let start = case.seed_start.unwrap_or("");
        (Seed::from_str(result).id() - Seed::from_str(start).id() + 1).clamp(1, case.num_seeds)
    }

    fn compare_duration(a: &Duration, b: &Duration) -> CmpOrdering {
        a.partial_cmp(b).unwrap_or(CmpOrdering::Equal)
    }

    fn percentile(values: &[Duration], pct: f64) -> Duration {
        let idx = ((values.len().saturating_sub(1)) as f64 * pct).ceil() as usize;
        values[idx.min(values.len() - 1)]
    }

    fn mean_duration(values: &[Duration]) -> Duration {
        let total = values.iter().map(Duration::as_secs_f64).sum::<f64>();
        Duration::from_secs_f64(total / values.len() as f64)
    }

    fn stdev_duration(values: &[Duration], mean: Duration) -> Duration {
        let mean_secs = mean.as_secs_f64();
        let variance = values
            .iter()
            .map(|value| {
                let delta = value.as_secs_f64() - mean_secs;
                delta * delta
            })
            .sum::<f64>()
            / values.len() as f64;
        Duration::from_secs_f64(variance.sqrt())
    }

    struct RunTicker {
        enabled: bool,
        stop: Arc<AtomicBool>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl RunTicker {
        fn start(enabled: bool, message: String, color: bool) -> Self {
            let stop = Arc::new(AtomicBool::new(false));
            if !enabled {
                return Self {
                    enabled,
                    stop,
                    handle: None,
                };
            }

            let started = Instant::now();
            let thread_stop = Arc::clone(&stop);
            let handle = thread::spawn(move || {
                const FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
                let mut frame = 0_usize;
                while !thread_stop.load(AtomicOrdering::Relaxed) {
                    let spinner = paint(color, ANSI_CYAN, FRAMES[frame % FRAMES.len()]);
                    print!(
                        "\r\x1b[2K  {spinner} {message}  elapsed {}",
                        format_status_duration(started.elapsed())
                    );
                    let _ = io::stdout().flush();
                    frame += 1;
                    thread::sleep(Duration::from_millis(90));
                }
            });

            Self {
                enabled,
                stop,
                handle: Some(handle),
            }
        }

        fn finish(mut self) {
            if !self.enabled {
                return;
            }
            self.stop.store(true, AtomicOrdering::Relaxed);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
            print!("\r\x1b[2K");
            let _ = io::stdout().flush();
        }
    }

    const ANSI_RESET: &str = "\x1b[0m";
    const ANSI_DIM: &str = "\x1b[2m";
    const ANSI_RED: &str = "\x1b[31m";
    const ANSI_GREEN: &str = "\x1b[32m";
    const ANSI_YELLOW: &str = "\x1b[33m";
    const ANSI_BLUE: &str = "\x1b[34m";
    const ANSI_CYAN: &str = "\x1b[36m";

    fn paint(enabled: bool, code: &str, text: &str) -> String {
        if enabled {
            format!("{code}{text}{ANSI_RESET}")
        } else {
            text.to_owned()
        }
    }

    fn print_run_header(title: &str, settings: BenchSettings<'_>, case_count: usize) {
        let color = settings.output.use_color();
        let rule = "═".repeat(78);
        println!("{}", paint(color, ANSI_CYAN, &format!("╔{rule}╗")));
        println!("{}", paint(color, ANSI_CYAN, &format!("║ {title:<76} ║")));
        println!("{}", paint(color, ANSI_CYAN, &format!("╚{rule}╝")));
        println!(
            "  case selector {:<14} budget {:>12}   repeats {:>3}   warmups {:>2}   threads {:>2}   cases {:>2}",
            settings.selected_case,
            format_integer(settings.budget),
            settings.repeat,
            settings.warmup,
            settings.threads,
            case_count,
        );
        println!(
            "  groups: {}",
            paint(color, ANSI_DIM, &bench::bench_group_keys().join(", "))
        );
        if settings.output.animate() {
            println!(
                "  {}",
                paint(
                    color,
                    ANSI_DIM,
                    "live status shows the active DLL call and elapsed time; final numbers below exclude rendering"
                )
            );
        }
        println!();
    }

    fn print_single_bench_report(summaries: &[BenchSummary], output: OutputOptions) {
        let color = output.use_color();
        print_section("Case Summary", color);
        println!(
            "{:<18} {:<9} {:<6} {:>7} {:>11} {:>9} {:>9} {:>10} {:>7} {:<12} samples",
            "case",
            "group",
            "shape",
            "scan",
            "seeds/s",
            "mean ms",
            "p95 ms",
            "ns/seed",
            "cv",
            "result",
        );
        println!("{}", paint(color, ANSI_DIM, &"─".repeat(126)));
        for summary in summaries {
            let cv = format!("{:.1}%", summary.coefficient_variation * 100.0);
            let cv = paint(
                color,
                cv_color(summary.coefficient_variation),
                &format!("{cv:>7}"),
            );
            println!(
                "{:<18} {:<9} {:<6} {:>7} {:>11} {:>9.3} {:>9.3} {:>10} {} {:<12} {}",
                summary.case_name,
                summary.group.label(),
                summary.shape.label(),
                format!("{:.1}%", summary.scanned_pct * 100.0),
                format_rate(summary.seeds_per_sec),
                ms(summary.mean_elapsed),
                ms(summary.p95_elapsed),
                format_ns(summary.ns_per_seed),
                cv,
                short_result(&summary.result, 12),
                sparkline(&summary.runs),
            );
        }
    }

    fn print_compare_report(
        comparisons: &[BenchComparison],
        min_ratio: f64,
        output: OutputOptions,
    ) {
        let color = output.use_color();
        let has_candidate = comparisons
            .iter()
            .any(|comparison| comparison.candidate().is_some());
        print_section("Case Comparison", color);
        if has_candidate {
            println!(
                "{:<18} {:<9} {:<6} {:>7} {:>11} {:>11} {:>11} {:>10} {:>10} {:>10} {:>18} samples",
                "case",
                "group",
                "shape",
                "scan",
                "base/s",
                "cand/s",
                "cpp/s",
                "base/cpp",
                "cand/cpp",
                "cand/base",
                "ns/seed B/K/C",
            );
            println!("{}", paint(color, ANSI_DIM, &"─".repeat(170)));
            for comparison in comparisons {
                let candidate = comparison
                    .candidate()
                    .expect("candidate presence checked before report");
                let base_ratio = comparison.base_vs_cpp_ratio();
                let candidate_cpp_ratio = comparison
                    .candidate_vs_cpp_ratio()
                    .expect("candidate presence checked before report");
                let candidate_base_ratio = comparison
                    .candidate_vs_base_ratio()
                    .expect("candidate presence checked before report");
                println!(
                    "{:<18} {:<9} {:<6} {:>7} {:>11} {:>11} {:>11} {} {} {} {:>18} B{} K{} C{}",
                    comparison.rust_base.case_name,
                    comparison.rust_base.group.label(),
                    comparison.rust_base.shape.label(),
                    format!("{:.1}%", comparison.rust_base.scanned_pct * 100.0),
                    format_rate(comparison.rust_base.seeds_per_sec),
                    format_rate(candidate.seeds_per_sec),
                    format_rate(comparison.cpp.seeds_per_sec),
                    paint(
                        color,
                        ratio_color(base_ratio, min_ratio),
                        &format!("{base_ratio:>10.3}x"),
                    ),
                    paint(
                        color,
                        ratio_color(candidate_cpp_ratio, min_ratio),
                        &format!("{candidate_cpp_ratio:>10.3}x"),
                    ),
                    paint(
                        color,
                        ratio_color(candidate_base_ratio, 1.0),
                        &format!("{candidate_base_ratio:>10.3}x"),
                    ),
                    format!(
                        "{}/{}/{}",
                        format_ns(comparison.rust_base.ns_per_seed),
                        format_ns(candidate.ns_per_seed),
                        format_ns(comparison.cpp.ns_per_seed)
                    ),
                    sparkline(&comparison.rust_base.runs),
                    sparkline(&candidate.runs),
                    sparkline(&comparison.cpp.runs),
                );
            }
        } else {
            println!(
                "{:<18} {:<9} {:<6} {:>7} {:>11} {:>11} {:>10} {:>15} {:>17} {:>11} samples",
                "case",
                "group",
                "shape",
                "scan",
                "rust-base/s",
                "cpp/s",
                "base/cpp",
                "mean ms B/C",
                "ns/seed B/C",
                "cv B/C",
            );
            println!("{}", paint(color, ANSI_DIM, &"─".repeat(150)));
            for comparison in comparisons {
                let base_ratio = comparison.base_vs_cpp_ratio();
                let ratio = paint(
                    color,
                    ratio_color(base_ratio, min_ratio),
                    &format!("{base_ratio:>10.3}x"),
                );
                let cv_pair = format!(
                    "{:.1}/{:.1}%",
                    comparison.rust_base.coefficient_variation * 100.0,
                    comparison.cpp.coefficient_variation * 100.0,
                );
                println!(
                    "{:<18} {:<9} {:<6} {:>7} {:>11} {:>11} {} {:>15} {:>17} {:>11} B{} C{}",
                    comparison.rust_base.case_name,
                    comparison.rust_base.group.label(),
                    comparison.rust_base.shape.label(),
                    format!("{:.1}%", comparison.rust_base.scanned_pct * 100.0),
                    format_rate(comparison.rust_base.seeds_per_sec),
                    format_rate(comparison.cpp.seeds_per_sec),
                    ratio,
                    format!(
                        "{:.3}/{:.3}",
                        ms(comparison.rust_base.mean_elapsed),
                        ms(comparison.cpp.mean_elapsed)
                    ),
                    format!(
                        "{}/{}",
                        format_ns(comparison.rust_base.ns_per_seed),
                        format_ns(comparison.cpp.ns_per_seed)
                    ),
                    cv_pair,
                    sparkline(&comparison.rust_base.runs),
                    sparkline(&comparison.cpp.runs),
                );
            }
        }
        print_group_report(comparisons, min_ratio, color, has_candidate);
        print_ranked_report(comparisons, min_ratio, color, has_candidate);
        print_noise_report(comparisons, color);
    }

    fn print_candidate_aggregate(
        aggregate: Option<&CandidateAggregate>,
        target_ratio: f64,
        min_scan_pct: f64,
        output: OutputOptions,
    ) {
        let color = output.use_color();
        print_section("Rust-candidate Non-hit Full-budget Gmean", color);
        let target = if target_ratio > 0.0 {
            format!("{target_ratio:.3}x target")
        } else {
            "informational".to_owned()
        };
        match aggregate {
            Some(aggregate) => {
                let color_code = if target_ratio > 0.0 {
                    ratio_color(aggregate.ratio, target_ratio)
                } else {
                    ANSI_CYAN
                };
                println!(
                    "  {} across {} cases with scan >= {:.1}% ({})",
                    paint(color, color_code, &format!("{:.3}x", aggregate.ratio)),
                    aggregate.cases,
                    min_scan_pct * 100.0,
                    target,
                );
            },
            None => println!(
                "  no non-hit candidate/base cases met scan >= {:.1}% ({})",
                min_scan_pct * 100.0,
                target,
            ),
        }
    }

    fn print_group_report(
        comparisons: &[BenchComparison],
        min_ratio: f64,
        color: bool,
        has_candidate: bool,
    ) {
        print_section("Group Speedups", color);
        if has_candidate {
            println!(
                "{:<10} {:>5} {:>11} {:>11} {:>11} {:<20} {:<20} meter",
                "group", "cases", "base/cpp", "cand/cpp", "cand/base", "best cand", "worst cand",
            );
            println!("{}", paint(color, ANSI_DIM, &"─".repeat(118)));
        } else {
            println!(
                "{:<10} {:>5} {:>9} {:>12} {:<20} {:<20} meter",
                "group", "cases", "base wins", "gmean", "best", "worst",
            );
            println!("{}", paint(color, ANSI_DIM, &"─".repeat(98)));
        }
        for group in bench_group_order() {
            let group_comparisons: Vec<_> = comparisons
                .iter()
                .filter(|comparison| comparison.rust_base.group == group)
                .collect();
            if group_comparisons.is_empty() {
                continue;
            }
            if has_candidate {
                let base_gmean = geometric_mean(
                    &group_comparisons
                        .iter()
                        .map(|comparison| comparison.base_vs_cpp_ratio())
                        .collect::<Vec<_>>(),
                );
                let candidate_cpp_gmean = geometric_mean(
                    &group_comparisons
                        .iter()
                        .filter_map(|comparison| comparison.candidate_vs_cpp_ratio())
                        .collect::<Vec<_>>(),
                );
                let candidate_base_gmean = geometric_mean(
                    &group_comparisons
                        .iter()
                        .filter_map(|comparison| comparison.candidate_vs_base_ratio())
                        .collect::<Vec<_>>(),
                );
                let mut best = group_comparisons[0];
                let mut worst = group_comparisons[0];
                for comparison in &group_comparisons {
                    if comparison.candidate_vs_base_ratio().unwrap_or(0.0)
                        > best.candidate_vs_base_ratio().unwrap_or(0.0)
                    {
                        best = comparison;
                    }
                    if comparison
                        .candidate_vs_base_ratio()
                        .unwrap_or(f64::INFINITY)
                        < worst.candidate_vs_base_ratio().unwrap_or(f64::INFINITY)
                    {
                        worst = comparison;
                    }
                }
                println!(
                    "{:<10} {:>5} {} {} {} {:<20} {:<20} {}",
                    group.label(),
                    group_comparisons.len(),
                    paint(
                        color,
                        ratio_color(base_gmean, min_ratio),
                        &format!("{base_gmean:>11.3}x"),
                    ),
                    paint(
                        color,
                        ratio_color(candidate_cpp_gmean, min_ratio),
                        &format!("{candidate_cpp_gmean:>11.3}x"),
                    ),
                    paint(
                        color,
                        ratio_color(candidate_base_gmean, 1.0),
                        &format!("{candidate_base_gmean:>11.3}x"),
                    ),
                    format!(
                        "{} {:.2}x",
                        best.rust_base.case_name,
                        best.candidate_vs_base_ratio().unwrap_or(0.0)
                    ),
                    format!(
                        "{} {:.2}x",
                        worst.rust_base.case_name,
                        worst.candidate_vs_base_ratio().unwrap_or(0.0)
                    ),
                    ratio_meter(candidate_base_gmean, color),
                );
            } else {
                let base_wins = group_comparisons
                    .iter()
                    .filter(|comparison| comparison.base_vs_cpp_ratio() >= 1.0)
                    .count();
                let gmean = geometric_mean(
                    &group_comparisons
                        .iter()
                        .map(|comparison| comparison.base_vs_cpp_ratio())
                        .collect::<Vec<_>>(),
                );
                let mut best = group_comparisons[0];
                let mut worst = group_comparisons[0];
                for comparison in &group_comparisons {
                    if comparison.base_vs_cpp_ratio() > best.base_vs_cpp_ratio() {
                        best = comparison;
                    }
                    if comparison.base_vs_cpp_ratio() < worst.base_vs_cpp_ratio() {
                        worst = comparison;
                    }
                }
                println!(
                    "{:<10} {:>5} {:>4}/{:<4} {} {:<20} {:<20} {}",
                    group.label(),
                    group_comparisons.len(),
                    base_wins,
                    group_comparisons.len(),
                    paint(
                        color,
                        ratio_color(gmean, min_ratio),
                        &format!("{gmean:>12.3}x")
                    ),
                    format!(
                        "{} {:.2}x",
                        best.rust_base.case_name,
                        best.base_vs_cpp_ratio()
                    ),
                    format!(
                        "{} {:.2}x",
                        worst.rust_base.case_name,
                        worst.base_vs_cpp_ratio()
                    ),
                    ratio_meter(gmean, color),
                );
            }
        }
    }

    fn print_ranked_report(
        comparisons: &[BenchComparison],
        min_ratio: f64,
        color: bool,
        has_candidate: bool,
    ) {
        let mut behind: Vec<_> = comparisons
            .iter()
            .filter(|comparison| comparison.base_vs_cpp_ratio() < 1.0)
            .collect();
        behind.sort_by(|a, b| {
            a.base_vs_cpp_ratio()
                .partial_cmp(&b.base_vs_cpp_ratio())
                .unwrap_or(CmpOrdering::Equal)
        });

        print_section("Rust-base Behind C++", color);
        if behind.is_empty() {
            println!("  none in this selection");
        } else {
            for comparison in behind.iter().take(5) {
                let base_ratio = comparison.base_vs_cpp_ratio();
                let ratio = paint(
                    color,
                    ratio_color(base_ratio, min_ratio),
                    &format!("{base_ratio:.3}x"),
                );
                println!(
                    "  {:<18} {:>11}  C++ faster by {:>6.1}%  {}",
                    comparison.rust_base.case_name,
                    ratio,
                    (1.0 - base_ratio) * 100.0,
                    paint(color, ANSI_DIM, comparison.rust_base.note),
                );
            }
        }

        let mut ahead: Vec<_> = comparisons
            .iter()
            .filter(|comparison| comparison.base_vs_cpp_ratio() >= 1.0)
            .collect();
        ahead.sort_by(|a, b| {
            b.base_vs_cpp_ratio()
                .partial_cmp(&a.base_vs_cpp_ratio())
                .unwrap_or(CmpOrdering::Equal)
        });

        print_section("Rust-base Ahead C++", color);
        if ahead.is_empty() {
            println!("  none in this selection");
        } else {
            for comparison in ahead.iter().take(5) {
                let base_ratio = comparison.base_vs_cpp_ratio();
                let ratio = paint(color, ANSI_GREEN, &format!("{base_ratio:.3}x"));
                println!(
                    "  {:<18} {:>11}  Rust faster by {:>6.1}%  {}",
                    comparison.rust_base.case_name,
                    ratio,
                    (base_ratio - 1.0) * 100.0,
                    paint(color, ANSI_DIM, comparison.rust_base.note),
                );
            }
        }

        if has_candidate {
            print_candidate_delta_report(comparisons, color);
        }
    }

    fn print_candidate_delta_report(comparisons: &[BenchComparison], color: bool) {
        let mut slower: Vec<_> = comparisons
            .iter()
            .filter(|comparison| {
                comparison
                    .candidate_vs_base_ratio()
                    .is_some_and(|ratio| ratio < 1.0)
            })
            .collect();
        slower.sort_by(|a, b| {
            a.candidate_vs_base_ratio()
                .partial_cmp(&b.candidate_vs_base_ratio())
                .unwrap_or(CmpOrdering::Equal)
        });

        print_section("Rust-candidate Lost To Base", color);
        if slower.is_empty() {
            println!("  none in this selection");
        } else {
            for comparison in slower.iter().take(5) {
                let ratio = comparison.candidate_vs_base_ratio().unwrap_or(0.0);
                println!(
                    "  {:<18} {}  candidate slower by {:>6.1}%  {}",
                    comparison.rust_base.case_name,
                    paint(color, ratio_color(ratio, 1.0), &format!("{ratio:>7.3}x")),
                    (1.0 - ratio) * 100.0,
                    paint(color, ANSI_DIM, comparison.rust_base.note),
                );
            }
        }

        let mut faster: Vec<_> = comparisons
            .iter()
            .filter(|comparison| {
                comparison
                    .candidate_vs_base_ratio()
                    .is_some_and(|ratio| ratio >= 1.0)
            })
            .collect();
        faster.sort_by(|a, b| {
            b.candidate_vs_base_ratio()
                .partial_cmp(&a.candidate_vs_base_ratio())
                .unwrap_or(CmpOrdering::Equal)
        });

        print_section("Rust-candidate Beat Base", color);
        if faster.is_empty() {
            println!("  none in this selection");
        } else {
            for comparison in faster.iter().take(5) {
                let ratio = comparison.candidate_vs_base_ratio().unwrap_or(0.0);
                println!(
                    "  {:<18} {}  candidate faster by {:>6.1}%  {}",
                    comparison.rust_base.case_name,
                    paint(color, ANSI_GREEN, &format!("{ratio:>7.3}x")),
                    (ratio - 1.0) * 100.0,
                    paint(color, ANSI_DIM, comparison.rust_base.note),
                );
            }
        }
    }

    fn print_noise_report(comparisons: &[BenchComparison], color: bool) {
        let noisy: Vec<_> = comparisons
            .iter()
            .filter(|comparison| {
                comparison.rust_base.coefficient_variation > 0.05
                    || comparison.cpp.coefficient_variation > 0.05
                    || comparison
                        .candidate()
                        .is_some_and(|candidate| candidate.coefficient_variation > 0.05)
            })
            .collect();
        if noisy.is_empty() {
            return;
        }
        print_section("High Variance", color);
        for comparison in noisy {
            if let Some(candidate) = comparison.candidate() {
                println!(
                    "  {:<18} base cv {:>5.1}%   cand cv {:>5.1}%   cpp cv {:>5.1}%   repeat or raise budget before trusting small deltas",
                    comparison.rust_base.case_name,
                    comparison.rust_base.coefficient_variation * 100.0,
                    candidate.coefficient_variation * 100.0,
                    comparison.cpp.coefficient_variation * 100.0,
                );
            } else {
                println!(
                    "  {:<18} base cv {:>5.1}%   cpp cv {:>5.1}%   repeat or raise budget before trusting small deltas",
                    comparison.rust_base.case_name,
                    comparison.rust_base.coefficient_variation * 100.0,
                    comparison.cpp.coefficient_variation * 100.0,
                );
            }
        }
    }

    fn print_section(title: &str, color: bool) {
        println!();
        println!(
            "{}",
            paint(color, ANSI_BLUE, &format!("╭─ {title} {}", "─".repeat(60)))
        );
    }

    fn print_tsv_header() {
        println!(
            "kind\timpl\tcase\tgroup\tshape\tbudget\tscanned\tscan_pct\tthreads\tsample\telapsed_ms\tseeds_per_sec\tns_per_seed\tmin_ms\tp50_ms\tp95_ms\tp99_ms\tmax_ms\tstdev_ms\tcv_pct\tresult"
        );
    }

    fn print_tsv_run(implementation: &str, case: &Case, run: &BenchRun) {
        println!(
            "run\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{}\t{:.3}\t{:.0}\t{:.3}\t\t\t\t\t\t\t\t{}",
            implementation,
            case.name,
            case.group.key(),
            case.shape.label(),
            case.num_seeds,
            run.scanned,
            run.scanned as f64 / case.num_seeds as f64,
            case.threads,
            run.run,
            ms(run.elapsed),
            run.seeds_per_sec,
            run.ns_per_seed,
            run.result,
        );
    }

    fn print_tsv_summary(summary: &BenchSummary) {
        println!(
            "summary\t{}\t{}\t{}\t{}\t{}\t{:.0}\t{:.6}\t{}\t{}\t{:.3}\t{:.0}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}",
            summary.implementation,
            summary.case_name,
            summary.group.key(),
            summary.shape.label(),
            summary.budget,
            summary.mean_scanned,
            summary.scanned_pct,
            summary.threads,
            summary.repeat,
            ms(summary.mean_elapsed),
            summary.seeds_per_sec,
            summary.ns_per_seed,
            ms(summary.min_elapsed),
            ms(summary.p50_elapsed),
            ms(summary.p95_elapsed),
            ms(summary.p99_elapsed),
            ms(summary.max_elapsed),
            ms(summary.stdev_elapsed),
            summary.coefficient_variation * 100.0,
            summary.result,
        );
    }

    fn print_tsv_compare(comparison: &BenchComparison, min_ratio: f64) {
        print_tsv_ratio(
            "rust-base-vs-cpp",
            &comparison.rust_base,
            &comparison.cpp,
            comparison.base_vs_cpp_ratio(),
            min_ratio,
        );
        if let Some(candidate) = comparison.candidate() {
            print_tsv_ratio(
                "rust-candidate-vs-cpp",
                candidate,
                &comparison.cpp,
                comparison.candidate_vs_cpp_ratio().unwrap_or(0.0),
                min_ratio,
            );
            print_tsv_ratio(
                "rust-candidate-vs-base",
                candidate,
                &comparison.rust_base,
                comparison.candidate_vs_base_ratio().unwrap_or(0.0),
                1.0,
            );
        }
    }

    fn print_tsv_candidate_aggregate(
        aggregate: Option<&CandidateAggregate>,
        settings: BenchSettings<'_>,
        target_ratio: f64,
        min_scan_pct: f64,
    ) {
        let (status, ratio, cases) = match aggregate {
            Some(aggregate) if target_ratio <= 0.0 || aggregate.ratio >= target_ratio => {
                ("ok", aggregate.ratio, aggregate.cases)
            },
            Some(aggregate) => ("below-target", aggregate.ratio, aggregate.cases),
            None => ("below-target", 0.0, 0),
        };
        println!(
            "aggregate\t{}\trust-candidate-vs-base-non-hit-full-budget-gmean\tall\tfull-budget\t{}\t0\t{:.6}\t{}\t{}\t\t\t\t\t\t\t\t\t\t\tratio={:.3};target_ratio={:.3};cases={};scan_pct_min={:.6};lhs=rust-candidate;rhs=rust-base",
            status,
            settings.budget,
            min_scan_pct,
            settings.threads,
            settings.repeat,
            ratio,
            target_ratio,
            cases,
            min_scan_pct,
        );
    }

    fn print_tsv_ratio(
        relation: &str,
        lhs: &BenchSummary,
        rhs: &BenchSummary,
        ratio: f64,
        target_ratio: f64,
    ) {
        let status = if lhs.result != rhs.result {
            "result-mismatch"
        } else if ratio >= target_ratio {
            "ok"
        } else if relation == "rust-base-vs-cpp" {
            "regression"
        } else {
            "below-target"
        };
        println!(
            "compare\t{}\t{}\t{}\t{}\t{}\t{:.0}\t{:.6}\t{}\t{}\t{:.3}\t{:.0}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\tratio={:.3};target_ratio={:.3};lhs={};rhs={};lhs_sps={:.0};rhs_sps={:.0};lhs_ms={:.3};rhs_ms={:.3};lhs_result={};rhs_result={}",
            status,
            lhs.case_name,
            lhs.group.key(),
            lhs.shape.label(),
            lhs.budget,
            lhs.mean_scanned,
            lhs.scanned_pct,
            lhs.threads,
            lhs.repeat,
            ms(lhs.mean_elapsed),
            lhs.seeds_per_sec,
            lhs.ns_per_seed,
            ms(lhs.min_elapsed),
            ms(lhs.p50_elapsed),
            ms(lhs.p95_elapsed),
            ms(lhs.p99_elapsed),
            ms(lhs.max_elapsed),
            ms(lhs.stdev_elapsed),
            lhs.coefficient_variation * 100.0,
            ratio,
            target_ratio,
            relation.split("-vs-").next().unwrap_or(relation),
            relation.split("-vs-").nth(1).unwrap_or("unknown"),
            lhs.seeds_per_sec,
            rhs.seeds_per_sec,
            ms(lhs.mean_elapsed),
            ms(rhs.mean_elapsed),
            lhs.result,
            rhs.result,
        );
    }

    fn bench_group_order() -> [BenchGroup; 8] {
        [
            BenchGroup::Baseline,
            BenchGroup::Tags,
            BenchGroup::Vouchers,
            BenchGroup::Packs,
            BenchGroup::Jokers,
            BenchGroup::Souls,
            BenchGroup::Deck,
            BenchGroup::Ux,
        ]
    }

    fn geometric_mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean_ln = values.iter().map(|value| value.ln()).sum::<f64>() / values.len() as f64;
        mean_ln.exp()
    }

    fn ratio_color(ratio: f64, min_ratio: f64) -> &'static str {
        if ratio < min_ratio {
            ANSI_RED
        } else if ratio < 1.0 {
            ANSI_YELLOW
        } else {
            ANSI_GREEN
        }
    }

    fn cv_color(coefficient_variation: f64) -> &'static str {
        if coefficient_variation > 0.05 {
            ANSI_RED
        } else if coefficient_variation > 0.02 {
            ANSI_YELLOW
        } else {
            ANSI_GREEN
        }
    }

    fn ratio_meter(ratio: f64, color: bool) -> String {
        const WIDTH: usize = 18;
        let normalized = ((ratio.log2() + 1.0) / 2.0).clamp(0.0, 1.0);
        let filled = (normalized * WIDTH as f64).round() as usize;
        let meter = format!("{}{}", "█".repeat(filled), "░".repeat(WIDTH - filled));
        paint(color, ratio_color(ratio, 1.0), &meter)
    }

    fn sparkline(runs: &[BenchRun]) -> String {
        const LEVELS: &[&str] = &["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
        if runs.is_empty() {
            return String::new();
        }
        if runs.len() == 1 {
            return "▅".to_owned();
        }
        let values: Vec<_> = runs.iter().map(|run| run.elapsed.as_secs_f64()).collect();
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if (max - min).abs() < f64::EPSILON {
            return "▅".repeat(values.len());
        }
        values
            .iter()
            .map(|value| {
                let idx =
                    (((*value - min) / (max - min)) * (LEVELS.len() - 1) as f64).round() as usize;
                LEVELS[idx.min(LEVELS.len() - 1)]
            })
            .collect()
    }

    fn ms(duration: Duration) -> f64 {
        duration.as_secs_f64() * 1000.0
    }

    fn format_status_duration(duration: Duration) -> String {
        let millis = duration.as_millis();
        if millis < 1_000 {
            format!("{millis}ms")
        } else {
            format!("{:.1}s", duration.as_secs_f64())
        }
    }

    fn format_rate(value: f64) -> String {
        format!("{}/s", format_compact(value))
    }

    fn format_compact(value: f64) -> String {
        if value >= 1_000_000_000.0 {
            format!("{:.2}B", value / 1_000_000_000.0)
        } else if value >= 1_000_000.0 {
            format!("{:.2}M", value / 1_000_000.0)
        } else if value >= 1_000.0 {
            format!("{:.2}K", value / 1_000.0)
        } else {
            format!("{value:.0}")
        }
    }

    fn format_ns(value: f64) -> String {
        if value >= 1_000_000.0 {
            format!("{:.2}ms", value / 1_000_000.0)
        } else if value >= 1_000.0 {
            format!("{:.2}us", value / 1_000.0)
        } else {
            format!("{value:.1}ns")
        }
    }

    fn format_integer(value: i64) -> String {
        let negative = value < 0;
        let mut chars: Vec<_> = value.abs().to_string().chars().rev().collect();
        let mut out = String::new();
        for idx in 0..chars.len() {
            if idx > 0 && idx % 3 == 0 {
                out.push(',');
            }
            out.push(chars[idx]);
        }
        if negative {
            out.push('-');
        }
        chars.clear();
        out.chars().rev().collect()
    }

    fn short_result(value: &str, width: usize) -> String {
        if value.chars().count() <= width {
            return value.to_owned();
        }
        let mut out: String = value.chars().take(width.saturating_sub(1)).collect();
        out.push('…');
        out
    }

    const TAG_KEYS: &[&str] = &[
        "tag_uncommon",
        "tag_rare",
        "tag_negative",
        "tag_foil",
        "tag_holo",
        "tag_polychrome",
        "tag_investment",
        "tag_voucher",
        "tag_boss",
        "tag_standard",
        "tag_charm",
        "tag_meteor",
        "tag_buffoon",
        "tag_handy",
        "tag_garbage",
        "tag_ethereal",
        "tag_coupon",
        "tag_double",
        "tag_juggle",
        "tag_d_six",
        "tag_top_up",
        "tag_skip",
        "tag_orbital",
        "tag_economy",
    ];

    const VOUCHER_KEYS: &[&str] = &[
        "v_overstock_norm",
        "v_overstock_plus",
        "v_clearance_sale",
        "v_liquidation",
        "v_hone",
        "v_glow_up",
        "v_reroll_surplus",
        "v_reroll_glut",
        "v_crystal_ball",
        "v_omen_globe",
        "v_telescope",
        "v_observatory",
        "v_grabber",
        "v_nacho_tong",
        "v_wasteful",
        "v_recyclomancy",
        "v_tarot_merchant",
        "v_tarot_tycoon",
        "v_planet_merchant",
        "v_planet_tycoon",
        "v_seed_money",
        "v_money_tree",
        "v_blank",
        "v_antimatter",
        "v_magic_trick",
        "v_illusion",
        "v_hieroglyph",
        "v_petroglyph",
        "v_directors_cut",
        "v_paint_brush",
        "v_retcon",
        "v_palette",
    ];

    const PACK_KEYS: &[&str] = &[
        "p_arcana_normal_1",
        "p_arcana_jumbo_1",
        "p_arcana_mega_1",
        "p_celestial_normal_1",
        "p_celestial_jumbo_1",
        "p_celestial_mega_1",
        "p_standard_normal_1",
        "p_standard_jumbo_1",
        "p_standard_mega_1",
        "p_buffoon_normal_1",
        "p_buffoon_jumbo_1",
        "p_buffoon_mega_1",
        "p_spectral_normal_1",
        "p_spectral_jumbo_1",
        "p_spectral_mega_1",
    ];

    const DECK_KEYS: &[&str] = &[
        "b_red",
        "b_blue",
        "b_yellow",
        "b_green",
        "b_black",
        "b_magic",
        "b_nebula",
        "b_ghost",
        "b_abandoned",
        "b_checkered",
        "b_zodiac",
        "b_painted",
        "b_anaglyph",
        "b_plasma",
        "b_erratic",
        "b_challenge",
    ];

    fn base_case(name: &'static str) -> Case {
        Case {
            name,
            group: BenchGroup::Baseline,
            shape: BenchShape::Mixed,
            note: "correctness fixture",
            seed_start: Some(""),
            voucher: Some(""),
            pack: Some(""),
            tag1: Some(""),
            tag2: Some(""),
            joker: Some(""),
            joker_location: Some("any"),
            souls: 0.0,
            observatory: false,
            perkeo: false,
            deck: Some("b_red"),
            erratic: false,
            no_faces: false,
            min_face_cards: 0,
            suit_ratio: 0.0,
            num_seeds: 1,
            threads: 1,
        }
    }

    fn compare_cases() -> Vec<Case> {
        let mut cases = Vec::new();

        let mut nulls = base_case("null_strings_no_filter_1");
        nulls.seed_start = None;
        nulls.voucher = None;
        nulls.pack = None;
        nulls.tag1 = None;
        nulls.tag2 = None;
        nulls.joker = None;
        nulls.joker_location = None;
        nulls.deck = None;
        cases.push(nulls);

        cases.push(base_case("empty_no_filter_1"));

        let mut one = base_case("1_no_filter_1");
        one.seed_start = Some("1");
        cases.push(one);

        let mut no_match = base_case("no_match_tiny");
        no_match.tag1 = Some("tag_charm");
        cases.push(no_match);

        let mut tag = base_case("tag_charm_10000");
        tag.tag1 = Some("tag_charm");
        tag.num_seeds = 10_000;
        cases.push(tag);

        let mut voucher = base_case("v_telescope_10000");
        voucher.voucher = Some("v_telescope");
        voucher.num_seeds = 10_000;
        cases.push(voucher);

        let mut pack = base_case("pack_spectral_10000");
        pack.pack = Some("p_spectral_mega_1");
        pack.num_seeds = 10_000;
        cases.push(pack);

        let mut observatory = base_case("observatory_100000");
        observatory.observatory = true;
        observatory.num_seeds = 100_000;
        cases.push(observatory);

        let mut erratic = base_case("erratic_faces_10000");
        erratic.deck = Some("b_erratic");
        erratic.erratic = true;
        erratic.min_face_cards = 12;
        erratic.num_seeds = 10_000;
        cases.push(erratic);

        let mut joker_shop = base_case("joker_shop_burnt_50000");
        joker_shop.joker = Some("Burnt Joker");
        joker_shop.joker_location = Some("shop");
        joker_shop.num_seeds = 50_000;
        cases.push(joker_shop);

        let mut joker_pack = base_case("joker_pack_reserved_parking_10000");
        joker_pack.joker = Some("Reserved Parking");
        joker_pack.joker_location = Some("pack");
        joker_pack.num_seeds = 10_000;
        cases.push(joker_pack);

        let mut joker_any_pack_filter = base_case("joker_any_blueprint_buffoon_50000");
        joker_any_pack_filter.joker = Some("Blueprint");
        joker_any_pack_filter.joker_location = Some("any");
        joker_any_pack_filter.pack = Some("p_buffoon_mega_1");
        joker_any_pack_filter.num_seeds = 50_000;
        cases.push(joker_any_pack_filter);

        let mut souls = base_case("souls_one_50000");
        souls.souls = 1.0;
        souls.num_seeds = 50_000;
        cases.push(souls);

        let mut souls_pack = base_case("souls_one_spectral_50000");
        souls_pack.pack = Some("p_spectral_mega_1");
        souls_pack.souls = 1.0;
        souls_pack.num_seeds = 50_000;
        cases.push(souls_pack);

        let mut perkeo = base_case("perkeo_20000");
        perkeo.perkeo = true;
        perkeo.num_seeds = 20_000;
        cases.push(perkeo);

        let mut threaded_no_match = base_case("threaded_shop_miss_5000");
        threaded_no_match.joker = Some("Perkeo");
        threaded_no_match.joker_location = Some("shop");
        threaded_no_match.num_seeds = 5_000;
        threaded_no_match.threads = 2;
        cases.push(threaded_no_match);

        let mut wrap = base_case("wrap_end_no_filter_2");
        wrap.seed_start = Some("ZZZZZZZZ");
        wrap.num_seeds = 2;
        cases.push(wrap);

        add_parser_matrix_cases(&mut cases);

        cases
    }

    fn repeated_alloc_free(dll: &Dll, repeats: usize) -> Result<bool, String> {
        let mut case = base_case("alloc_free_stress");
        case.seed_start = Some("1");
        for _ in 0..repeats {
            if dll.run(&case)?.as_deref() != Some("1") {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn add_parser_matrix_cases(cases: &mut Vec<Case>) {
        for &key in TAG_KEYS {
            let mut case = base_case("parser_tag");
            case.name = key;
            case.tag1 = Some(key);
            cases.push(case);
        }
        for &key in VOUCHER_KEYS {
            let mut case = base_case("parser_voucher");
            case.name = key;
            case.voucher = Some(key);
            cases.push(case);
        }
        for &key in PACK_KEYS {
            let mut case = base_case("parser_pack");
            case.name = key;
            case.pack = Some(key);
            cases.push(case);
        }
        for &key in DECK_KEYS {
            let mut case = base_case("parser_deck");
            case.name = key;
            case.deck = Some(key);
            case.erratic = key == "b_erratic";
            case.min_face_cards = if case.erratic { 12 } else { 0 };
            cases.push(case);
        }
    }

    fn selected_bench_cases(
        selected_case: &str,
        budget: i64,
        threads: i32,
    ) -> Result<Vec<Case>, String> {
        bench::selected_bench_cases(selected_case).map(|cases| {
            cases
                .into_iter()
                .map(|case| case_from_bench_case(case, budget, threads))
                .collect()
        })
    }

    fn case_from_bench_case(case: BenchCase, budget: i64, threads: i32) -> Case {
        Case {
            name: case.name,
            group: case.group,
            shape: case.shape,
            note: case.note,
            seed_start: Some(case.seed_start),
            voucher: Some(case.voucher),
            pack: Some(case.pack),
            tag1: Some(case.tag1),
            tag2: Some(case.tag2),
            joker: Some(case.joker),
            joker_location: Some(case.joker_location),
            souls: case.souls,
            observatory: case.observatory,
            perkeo: case.perkeo,
            deck: Some(case.deck),
            erratic: case.erratic,
            no_faces: case.no_faces,
            min_face_cards: case.min_face_cards,
            suit_ratio: case.suit_ratio,
            num_seeds: budget,
            threads,
        }
    }
}
