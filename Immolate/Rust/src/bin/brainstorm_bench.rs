use std::env;
use std::process;
use std::time::Instant;

use immolate::filters::FilterConfig;
use immolate::search::brainstorm_search_core;
use immolate::seed::Seed;
use immolate::v2::brainstorm_search_core_v2;
use immolate::v3::brainstorm_search_core_v3;

#[path = "../bench_cases.rs"]
mod bench_cases;

#[derive(Debug)]
struct Args {
    case: String,
    budget: i64,
    threads: i32,
    repeat: usize,
    engine: Engine,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            case: "all".to_owned(),
            budget: 1_000_000,
            threads: 1,
            repeat: 3,
            engine: Engine::V3,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Engine {
    Base,
    V2,
    V3,
    Both,
}

fn main() {
    let args = parse_args().unwrap_or_else(|message| {
        eprintln!("{message}");
        eprintln!(
            "usage: brainstorm_bench [--case all|GROUP|NAME] [--budget N] [--threads N] [--repeat N] [--engine v3|v2|base|both]"
        );
        process::exit(2);
    });

    println!(
        "engine\tcase\tgroup\tshape\tbudget\tscanned\tscan_pct\tthreads\trepeat\telapsed_ms\tseeds_per_sec\tns_per_seed\tresult\tnote"
    );
    for case in bench_cases::selected_bench_cases(&args.case).unwrap_or_else(|err| {
        eprintln!("{err}");
        process::exit(2);
    }) {
        let cfg = case_config(case);
        for engine in selected_engines(args.engine) {
            for repeat in 1..=args.repeat {
                let started = Instant::now();
                let result = match engine {
                    Engine::Base => {
                        brainstorm_search_core(case.seed_start, &cfg, args.budget, args.threads)
                    },
                    Engine::V2 => {
                        brainstorm_search_core_v2(case.seed_start, &cfg, args.budget, args.threads)
                    },
                    Engine::V3 => {
                        brainstorm_search_core_v3(case.seed_start, &cfg, args.budget, args.threads)
                    },
                    Engine::Both => unreachable!("expanded by selected_engines"),
                };
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
                    "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{}\t{:.3}\t{:.0}\t{:.3}\t{}\t{}",
                    engine.label(),
                    case.name,
                    case.group.label(),
                    case.shape.label(),
                    args.budget,
                    scanned,
                    scanned as f64 / args.budget as f64,
                    args.threads,
                    repeat,
                    elapsed.as_secs_f64() * 1000.0,
                    seeds_per_sec,
                    elapsed_secs * 1_000_000_000.0 / scanned as f64,
                    result,
                    case.note,
                );
            }
        }
    }
}

impl Engine {
    const fn label(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::V2 => "v2",
            Self::V3 => "v3",
            Self::Both => "both",
        }
    }
}

fn selected_engines(engine: Engine) -> &'static [Engine] {
    match engine {
        Engine::Base => &[Engine::Base],
        Engine::V2 => &[Engine::V2],
        Engine::V3 => &[Engine::V3],
        Engine::Both => &[Engine::Base, Engine::V2, Engine::V3],
    }
}

fn case_config(case: bench_cases::BenchCase) -> FilterConfig {
    FilterConfig::from_raw(
        case.voucher,
        case.pack,
        case.tag1,
        case.tag2,
        case.joker,
        case.joker_location,
        case.souls,
        case.observatory,
        case.perkeo,
        case.deck,
        case.erratic,
        case.no_faces,
        case.min_face_cards,
        case.suit_ratio,
    )
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
            "--engine" => {
                args.engine = match value.as_str() {
                    "base" => Engine::Base,
                    "v2" => Engine::V2,
                    "v3" => Engine::V3,
                    "both" => Engine::Both,
                    _ => return Err(format!("invalid --engine: {value}")),
                };
            },
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }
    Ok(args)
}
