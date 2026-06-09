use std::env;
use std::process;
use std::time::Instant;

use immolate::filters::FilterConfig;
use immolate::search::brainstorm_search_core;
use immolate::seed::Seed;

#[derive(Clone)]
struct BenchCase {
    name: &'static str,
    seed_start: &'static str,
    voucher: &'static str,
    pack: &'static str,
    tag1: &'static str,
    tag2: &'static str,
    joker: &'static str,
    joker_location: &'static str,
    souls: f64,
    observatory: bool,
    perkeo: bool,
    deck: &'static str,
    erratic: bool,
    no_faces: bool,
    min_face_cards: i32,
    suit_ratio: f64,
}

impl BenchCase {
    fn config(&self) -> FilterConfig {
        FilterConfig::from_raw(
            self.voucher,
            self.pack,
            self.tag1,
            self.tag2,
            self.joker,
            self.joker_location,
            self.souls,
            self.observatory,
            self.perkeo,
            self.deck,
            self.erratic,
            self.no_faces,
            self.min_face_cards,
            self.suit_ratio,
        )
    }
}

#[derive(Debug)]
struct Args {
    case: String,
    budget: i64,
    threads: i32,
    repeat: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            case: "all".to_owned(),
            budget: 1_000_000,
            threads: 1,
            repeat: 3,
        }
    }
}

fn main() {
    let args = parse_args().unwrap_or_else(|message| {
        eprintln!("{message}");
        eprintln!(
            "usage: brainstorm_bench [--case all|shop-miss|pack-miss|tag-hit|erratic] [--budget N] [--threads N] [--repeat N]"
        );
        process::exit(2);
    });

    println!("case\tbudget\tscanned\tthreads\trepeat\telapsed_ms\tseeds_per_sec\tresult");
    for case in selected_cases(&args.case).unwrap_or_else(|| {
        eprintln!("unknown case: {}", args.case);
        process::exit(2);
    }) {
        let cfg = case.config();
        for repeat in 1..=args.repeat {
            let started = Instant::now();
            let result = brainstorm_search_core(case.seed_start, &cfg, args.budget, args.threads);
            let elapsed = started.elapsed();
            let elapsed_secs = elapsed.as_secs_f64();
            let scanned = scanned_count(case.seed_start, result.as_deref(), args.budget);
            let seeds_per_sec = if elapsed_secs > 0.0 {
                scanned as f64 / elapsed_secs
            } else {
                f64::INFINITY
            };
            let result = match result.as_deref() {
                Some("") | None => "<null>",
                Some(seed) => seed,
            };
            println!(
                "{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.0}\t{}",
                case.name,
                args.budget,
                scanned,
                args.threads,
                repeat,
                elapsed.as_secs_f64() * 1000.0,
                seeds_per_sec,
                result,
            );
        }
    }
}

fn scanned_count(seed_start: &str, result: Option<&str>, budget: i64) -> i64 {
    let Some(result) = result else {
        return budget;
    };
    if result.is_empty() {
        return 1;
    }
    (Seed::from_str(result).id() - Seed::from_str(seed_start).id() + 1).clamp(1, budget)
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut iter = env::args().skip(1);
    while let Some(flag) = iter.next() {
        let value = iter
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--case" => args.case = value,
            "--budget" => {
                args.budget = value
                    .parse::<i64>()
                    .map_err(|_| format!("invalid --budget: {value}"))?;
                if args.budget <= 0 {
                    return Err("--budget must be positive".to_owned());
                }
            },
            "--threads" => {
                args.threads = value
                    .parse::<i32>()
                    .map_err(|_| format!("invalid --threads: {value}"))?;
            },
            "--repeat" => {
                args.repeat = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --repeat: {value}"))?;
                if args.repeat == 0 {
                    return Err("--repeat must be positive".to_owned());
                }
            },
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }
    Ok(args)
}

fn selected_cases(name: &str) -> Option<Vec<BenchCase>> {
    let cases = bench_cases();
    if name == "all" {
        return Some(cases);
    }
    let selected: Vec<_> = cases.into_iter().filter(|case| case.name == name).collect();
    if selected.is_empty() {
        None
    } else {
        Some(selected)
    }
}

fn bench_cases() -> Vec<BenchCase> {
    vec![
        BenchCase {
            name: "shop-miss",
            seed_start: "",
            voucher: "",
            pack: "",
            tag1: "",
            tag2: "",
            joker: "Perkeo",
            joker_location: "shop",
            souls: 0.0,
            observatory: false,
            perkeo: false,
            deck: "b_red",
            erratic: false,
            no_faces: false,
            min_face_cards: 0,
            suit_ratio: 0.0,
        },
        BenchCase {
            name: "pack-miss",
            seed_start: "",
            voucher: "",
            pack: "p_spectral_normal_1",
            tag1: "",
            tag2: "",
            joker: "",
            joker_location: "any",
            souls: 3.0,
            observatory: false,
            perkeo: false,
            deck: "b_red",
            erratic: false,
            no_faces: false,
            min_face_cards: 0,
            suit_ratio: 0.0,
        },
        BenchCase {
            name: "tag-hit",
            seed_start: "",
            voucher: "",
            pack: "",
            tag1: "tag_charm",
            tag2: "",
            joker: "",
            joker_location: "any",
            souls: 0.0,
            observatory: false,
            perkeo: false,
            deck: "b_red",
            erratic: false,
            no_faces: false,
            min_face_cards: 0,
            suit_ratio: 0.0,
        },
        BenchCase {
            name: "erratic",
            seed_start: "",
            voucher: "",
            pack: "",
            tag1: "",
            tag2: "",
            joker: "",
            joker_location: "any",
            souls: 0.0,
            observatory: false,
            perkeo: false,
            deck: "b_erratic",
            erratic: true,
            no_faces: false,
            min_face_cards: 12,
            suit_ratio: 0.0,
        },
    ]
}
