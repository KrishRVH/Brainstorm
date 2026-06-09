use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::filters::FilterConfig;
use crate::search::{resolve_seed_budget, resolve_threads};
use crate::seed::{SEED_SPACE, Seed};
use crate::v2::config::{CompiledFilter, KernelShape};
use crate::v2::kernels::apply_compiled_filter;
use crate::v2::seed::V2State;

pub fn brainstorm_search_core_v2(
    seed_start: &str,
    cfg: &FilterConfig,
    num_seeds: i64,
    threads: i32,
) -> Option<String> {
    let budget = resolve_seed_budget(num_seeds);
    let thread_count = resolve_threads(threads);
    let compiled = CompiledFilter::compile(cfg);
    search_filters(seed_start, compiled, budget, thread_count)
}

fn search_filters(
    seed_start: &str,
    cfg: CompiledFilter,
    num_seeds: i64,
    threads: usize,
) -> Option<String> {
    let start_seed = Seed::from_str(seed_start).id();
    if cfg.shape == KernelShape::NoMatch {
        return None;
    }
    if threads <= 1 {
        return search_block(start_seed, num_seeds, &cfg);
    }

    let block_size = cfg.chunk_size();
    let total_blocks = (num_seeds + block_size - 1) / block_size;
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
                    let offset = block * block_size;
                    let start = (start_seed + offset).rem_euclid(SEED_SPACE);
                    let count = block_size.min(num_seeds - offset);
                    if let Some(seed) = search_block_with_flag(start, count, &cfg, &found) {
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

fn search_block(start: i64, count: i64, cfg: &CompiledFilter) -> Option<String> {
    let mut state = V2State::from_id(start);
    for _ in 0..count {
        if apply_compiled_filter(&mut state, cfg) {
            return Some(state.seed.to_string());
        }
        state.next();
    }
    None
}

fn search_block_with_flag(
    start: i64,
    count: i64,
    cfg: &CompiledFilter,
    found: &AtomicBool,
) -> Option<String> {
    let mut state = V2State::from_id(start);
    for _ in 0..count {
        if found.load(Ordering::Relaxed) {
            return None;
        }
        if apply_compiled_filter(&mut state, cfg) {
            return Some(state.seed.to_string());
        }
        state.next();
    }
    None
}
