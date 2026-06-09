use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::filters::{FilterConfig, apply_filters};
use crate::instance::Instance;
use crate::seed::{SEED_SPACE, Seed};

const DEFAULT_SEED_BUDGET: i64 = 100_000_000;
const BLOCK_SIZE: i64 = 1_000_000;

pub fn resolve_seed_budget(num_seeds: i64) -> i64 {
    let budget = if num_seeds <= 0 {
        DEFAULT_SEED_BUDGET
    } else {
        num_seeds
    };
    budget.min(SEED_SPACE)
}

pub fn resolve_threads(threads: i32) -> usize {
    if threads > 0 {
        return threads.clamp(1, 4) as usize;
    }
    thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .clamp(1, 4)
}

pub fn brainstorm_search_core(
    seed_start: &str,
    cfg: &FilterConfig,
    num_seeds: i64,
    threads: i32,
) -> Option<String> {
    let budget = resolve_seed_budget(num_seeds);
    let thread_count = resolve_threads(threads);
    search_filters(seed_start, cfg, budget, thread_count)
}

fn search_filters(
    seed_start: &str,
    cfg: &FilterConfig,
    num_seeds: i64,
    threads: usize,
) -> Option<String> {
    let start_seed = Seed::from_str(seed_start).id();
    if threads <= 1 {
        return search_block(start_seed, num_seeds, cfg);
    }

    let total_blocks = (num_seeds + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let next_block = Arc::new(AtomicI64::new(0));
    let found = Arc::new(AtomicBool::new(false));
    let result = Arc::new(Mutex::new(None::<String>));

    thread::scope(|scope| {
        for _ in 0..threads {
            let next_block = Arc::clone(&next_block);
            let found = Arc::clone(&found);
            let result = Arc::clone(&result);
            scope.spawn(move || {
                loop {
                    if found.load(Ordering::Relaxed) {
                        break;
                    }
                    let block = next_block.fetch_add(1, Ordering::Relaxed);
                    if block >= total_blocks {
                        break;
                    }
                    let offset = block * BLOCK_SIZE;
                    let start = (start_seed + offset).rem_euclid(SEED_SPACE);
                    let count = BLOCK_SIZE.min(num_seeds - offset);
                    if let Some(seed) = search_block_with_flag(start, count, cfg, &found) {
                        if let Ok(mut guard) = result.lock() {
                            if guard.is_none() {
                                *guard = Some(seed);
                                found.store(true, Ordering::Relaxed);
                            }
                        }
                        break;
                    }
                }
            });
        }
    });

    result.lock().map_or(None, |guard| guard.clone())
}

fn search_block(start: i64, count: i64, cfg: &FilterConfig) -> Option<String> {
    let mut inst = Instance::new(Seed::from_id(start));
    for _ in 0..count {
        if apply_filters(&mut inst, cfg) {
            let out = inst.seed.to_string();
            return Some(out);
        }
        inst.next();
    }
    None
}

fn search_block_with_flag(
    start: i64,
    count: i64,
    cfg: &FilterConfig,
    found: &AtomicBool,
) -> Option<String> {
    let mut inst = Instance::new(Seed::from_id(start));
    for _ in 0..count {
        if found.load(Ordering::Relaxed) {
            return None;
        }
        if apply_filters(&mut inst, cfg) {
            let out = inst.seed.to_string();
            return Some(out);
        }
        inst.next();
    }
    None
}

#[allow(dead_code)]
pub fn seed_space() -> i64 {
    SEED_SPACE
}
