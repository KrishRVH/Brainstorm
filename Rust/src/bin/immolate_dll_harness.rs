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
mod windows_harness {
    use std::cmp::Ordering;
    use std::env;
    use std::ffi::{CStr, CString, OsStr};
    use std::os::raw::{c_char, c_double, c_int, c_longlong, c_void};
    use std::os::windows::ffi::OsStrExt;
    use std::ptr;
    use std::time::{Duration, Instant};

    use immolate::seed::Seed;

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

    enum Command {
        Compare {
            cpp: String,
            rust: String,
            threads: i32,
        },
        Bench {
            dll: String,
            case: String,
            budget: i64,
            threads: i32,
            repeat: usize,
        },
        BenchCompare {
            cpp: String,
            rust: String,
            case: String,
            budget: i64,
            threads: i32,
            repeat: usize,
            min_ratio: f64,
        },
    }

    pub fn main() {
        match parse_command(env::args().skip(1).collect()) {
            Ok(Command::Compare { cpp, rust, threads }) => {
                if let Err(err) = compare(&cpp, &rust, threads) {
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
            }) => {
                if let Err(err) = bench(&dll, &case, budget, threads, repeat) {
                    eprintln!("{err}");
                    std::process::exit(1);
                }
            },
            Ok(Command::BenchCompare {
                cpp,
                rust,
                case,
                budget,
                threads,
                repeat,
                min_ratio,
            }) => {
                if let Err(err) =
                    bench_compare(&cpp, &rust, &case, budget, threads, repeat, min_ratio)
                {
                    eprintln!("{err}");
                    std::process::exit(1);
                }
            },
            Err(err) => {
                eprintln!("{err}");
                eprintln!(
                    "usage:\n  immolate_dll_harness compare --cpp PATH --rust PATH [--threads N]\n  immolate_dll_harness bench --dll PATH [--case all|NAME] [--budget N] [--threads N] [--repeat N]\n  immolate_dll_harness bench-compare --cpp PATH --rust PATH [--case all|NAME] [--budget N] [--threads N] [--repeat N] [--min-ratio N]"
                );
                std::process::exit(2);
            },
        }
    }

    fn compare(cpp_path: &str, rust_path: &str, threads: i32) -> Result<(), String> {
        let cpp = Dll::load(cpp_path)?;
        let rust = Dll::load(rust_path)?;
        let mut failed = false;
        println!("status\tcase\tcpp\trust");
        for mut case in compare_cases() {
            if threads != 1 {
                case.threads = threads;
            }
            let cpp_result = cpp.run(&case)?;
            let rust_result = rust.run(&case)?;
            let status = if cpp_result == rust_result {
                "ok"
            } else {
                failed = true;
                "mismatch"
            };
            println!(
                "{status}\t{}\t{}\t{}",
                case.name,
                display_result(cpp_result.as_deref()),
                display_result(rust_result.as_deref()),
            );
        }
        for (name, dll) in [("cpp", &cpp), ("rust", &rust)] {
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

    fn bench(
        dll_path: &str,
        selected_case: &str,
        budget: i64,
        threads: i32,
        repeat: usize,
    ) -> Result<(), String> {
        if budget <= 0 {
            return Err("--budget must be positive".to_owned());
        }
        if repeat == 0 {
            return Err("--repeat must be positive".to_owned());
        }
        let dll = Dll::load(dll_path)?;
        let cases = selected_bench_cases(selected_case)?;

        println!("kind\tcase\tbudget\tscanned\tthreads\trepeat\telapsed_ms\tseeds_per_sec\tresult");
        for mut case in cases {
            case.num_seeds = budget;
            case.threads = threads;
            let mut durations = Vec::with_capacity(repeat);
            let mut scanned_counts = Vec::with_capacity(repeat);
            for run in 1..=repeat {
                let started = Instant::now();
                let result = dll.run(&case)?;
                let elapsed = started.elapsed();
                let scanned = scanned_count(&case, result.as_deref());
                durations.push(elapsed);
                scanned_counts.push(scanned);
                println!(
                    "run\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.0}\t{}",
                    case.name,
                    budget,
                    scanned,
                    threads,
                    run,
                    elapsed.as_secs_f64() * 1000.0,
                    scanned as f64 / elapsed.as_secs_f64(),
                    display_result(result.as_deref()),
                );
            }
            durations.sort_by(compare_duration);
            let mean_scanned = scanned_counts
                .iter()
                .map(|value| *value as f64)
                .sum::<f64>()
                / repeat as f64;
            println!(
                "summary\t{}\t{}\t{:.0}\t{}\t{}\t{:.3}\t{:.0}\tp50={:.3};p95={:.3};p99={:.3}",
                case.name,
                budget,
                mean_scanned,
                threads,
                repeat,
                mean_duration(&durations).as_secs_f64() * 1000.0,
                mean_scanned / mean_duration(&durations).as_secs_f64(),
                percentile(&durations, 0.50).as_secs_f64() * 1000.0,
                percentile(&durations, 0.95).as_secs_f64() * 1000.0,
                percentile(&durations, 0.99).as_secs_f64() * 1000.0,
            );
        }
        Ok(())
    }

    fn bench_compare(
        cpp_path: &str,
        rust_path: &str,
        selected_case: &str,
        budget: i64,
        threads: i32,
        repeat: usize,
        min_ratio: f64,
    ) -> Result<(), String> {
        if budget <= 0 {
            return Err("--budget must be positive".to_owned());
        }
        if repeat == 0 {
            return Err("--repeat must be positive".to_owned());
        }
        if min_ratio <= 0.0 {
            return Err("--min-ratio must be positive".to_owned());
        }
        let cpp = Dll::load(cpp_path)?;
        let rust = Dll::load(rust_path)?;
        let mut cases = selected_bench_cases(selected_case)?;
        println!(
            "kind\timpl\tcase\tbudget\tscanned\tthreads\trepeat\telapsed_ms\tseeds_per_sec\tresult"
        );
        let mut failed = false;
        for case in &mut cases {
            case.num_seeds = budget;
            case.threads = threads;
            let cpp_summary = measure_bench_case(&cpp, case, repeat, "cpp")?;
            let rust_summary = measure_bench_case(&rust, case, repeat, "rust")?;
            let ratio = rust_summary.seeds_per_sec / cpp_summary.seeds_per_sec;
            let status = if ratio >= min_ratio {
                "ok"
            } else {
                failed = true;
                "regression"
            };
            println!(
                "compare\t{status}\t{}\t{}\t{:.0}\t{}\t{}\t{:.3}\t{:.3}\tratio={:.3};min_ratio={:.3};cpp_sps={:.0};rust_sps={:.0}",
                case.name,
                budget,
                rust_summary.mean_scanned,
                threads,
                repeat,
                rust_summary.mean_elapsed.as_secs_f64() * 1000.0,
                cpp_summary.mean_elapsed.as_secs_f64() * 1000.0,
                ratio,
                min_ratio,
                cpp_summary.seeds_per_sec,
                rust_summary.seeds_per_sec,
            );
        }
        if failed {
            Err("benchmark regression threshold failed".to_owned())
        } else {
            Ok(())
        }
    }

    struct BenchSummary {
        mean_elapsed: Duration,
        mean_scanned: f64,
        seeds_per_sec: f64,
    }

    fn measure_bench_case(
        dll: &Dll,
        case: &Case,
        repeat: usize,
        implementation: &str,
    ) -> Result<BenchSummary, String> {
        let mut durations = Vec::with_capacity(repeat);
        let mut scanned_counts = Vec::with_capacity(repeat);
        for run in 1..=repeat {
            let started = Instant::now();
            let result = dll.run(case)?;
            let elapsed = started.elapsed();
            let scanned = scanned_count(case, result.as_deref());
            durations.push(elapsed);
            scanned_counts.push(scanned);
            println!(
                "run\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.0}\t{}",
                implementation,
                case.name,
                case.num_seeds,
                scanned,
                case.threads,
                run,
                elapsed.as_secs_f64() * 1000.0,
                scanned as f64 / elapsed.as_secs_f64(),
                display_result(result.as_deref()),
            );
        }
        durations.sort_by(compare_duration);
        let mean_elapsed = mean_duration(&durations);
        let mean_scanned = scanned_counts
            .iter()
            .map(|value| *value as f64)
            .sum::<f64>()
            / repeat as f64;
        let seeds_per_sec = mean_scanned / mean_elapsed.as_secs_f64();
        println!(
            "summary\t{}\t{}\t{}\t{:.0}\t{}\t{}\t{:.3}\t{:.0}\tp50={:.3};p95={:.3};p99={:.3}",
            implementation,
            case.name,
            case.num_seeds,
            mean_scanned,
            case.threads,
            repeat,
            mean_elapsed.as_secs_f64() * 1000.0,
            seeds_per_sec,
            percentile(&durations, 0.50).as_secs_f64() * 1000.0,
            percentile(&durations, 0.95).as_secs_f64() * 1000.0,
            percentile(&durations, 0.99).as_secs_f64() * 1000.0,
        );
        Ok(BenchSummary {
            mean_elapsed,
            mean_scanned,
            seeds_per_sec,
        })
    }

    fn parse_command(args: Vec<String>) -> Result<Command, String> {
        let Some(mode) = args.first() else {
            return Err("missing command".to_owned());
        };
        match mode.as_str() {
            "compare" => {
                let mut cpp = None;
                let mut rust = None;
                let mut threads = 1;
                parse_flags(&args[1..], |flag, value| match flag {
                    "--cpp" => {
                        cpp = Some(value.to_owned());
                        Ok(())
                    },
                    "--rust" => {
                        rust = Some(value.to_owned());
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
                    rust: rust.ok_or_else(|| "missing --rust".to_owned())?,
                    threads,
                })
            },
            "bench" => {
                let mut dll = None;
                let mut case = "all".to_owned();
                let mut budget = 1_000_000;
                let mut threads = 1;
                let mut repeat = 5;
                parse_flags(&args[1..], |flag, value| match flag {
                    "--dll" => {
                        dll = Some(value.to_owned());
                        Ok(())
                    },
                    "--case" => {
                        case = value.to_owned();
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
                    _ => Err(format!("unknown bench flag: {flag}")),
                })?;
                Ok(Command::Bench {
                    dll: dll.ok_or_else(|| "missing --dll".to_owned())?,
                    case,
                    budget,
                    threads,
                    repeat,
                })
            },
            "bench-compare" => {
                let mut cpp = None;
                let mut rust = None;
                let mut case = "all".to_owned();
                let mut budget = 1_000_000;
                let mut threads = 1;
                let mut repeat = 5;
                let mut min_ratio = 0.8;
                parse_flags(&args[1..], |flag, value| match flag {
                    "--cpp" => {
                        cpp = Some(value.to_owned());
                        Ok(())
                    },
                    "--rust" => {
                        rust = Some(value.to_owned());
                        Ok(())
                    },
                    "--case" => {
                        case = value.to_owned();
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
                    "--min-ratio" => {
                        min_ratio = parse_value(value, "--min-ratio")?;
                        Ok(())
                    },
                    _ => Err(format!("unknown bench-compare flag: {flag}")),
                })?;
                Ok(Command::BenchCompare {
                    cpp: cpp.ok_or_else(|| "missing --cpp".to_owned())?,
                    rust: rust.ok_or_else(|| "missing --rust".to_owned())?,
                    case,
                    budget,
                    threads,
                    repeat,
                    min_ratio,
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

    fn compare_duration(a: &Duration, b: &Duration) -> Ordering {
        a.partial_cmp(b).unwrap_or(Ordering::Equal)
    }

    fn percentile(values: &[Duration], pct: f64) -> Duration {
        let idx = ((values.len().saturating_sub(1)) as f64 * pct).ceil() as usize;
        values[idx.min(values.len() - 1)]
    }

    fn mean_duration(values: &[Duration]) -> Duration {
        let total = values.iter().map(Duration::as_secs_f64).sum::<f64>();
        Duration::from_secs_f64(total / values.len() as f64)
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

    fn bench_cases() -> Vec<Case> {
        let mut shop = base_case("shop-miss");
        shop.joker = Some("Perkeo");
        shop.joker_location = Some("shop");

        let mut pack = base_case("pack-miss");
        pack.pack = Some("p_spectral_normal_1");
        pack.souls = 3.0;

        let mut tag = base_case("tag-hit");
        tag.tag1 = Some("tag_charm");

        let mut erratic = base_case("erratic");
        erratic.deck = Some("b_erratic");
        erratic.erratic = true;
        erratic.min_face_cards = 12;

        vec![shop, pack, tag, erratic]
    }

    fn selected_bench_cases(selected_case: &str) -> Result<Vec<Case>, String> {
        let mut cases = bench_cases();
        if selected_case == "all" {
            return Ok(cases);
        }
        cases.retain(|case| case.name == selected_case);
        if cases.is_empty() {
            Err(format!("unknown benchmark case: {selected_case}"))
        } else {
            Ok(cases)
        }
    }
}
