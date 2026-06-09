#![allow(unsafe_code)]

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_int, c_longlong};
use std::ptr;

use crate::filters::FilterConfig;
use crate::search::brainstorm_search_core;

#[unsafe(no_mangle)]
pub extern "C" fn brainstorm_search(
    seed_start: *const c_char,
    voucher_key: *const c_char,
    pack_key: *const c_char,
    tag1_key: *const c_char,
    tag2_key: *const c_char,
    joker_name: *const c_char,
    joker_location: *const c_char,
    souls: c_double,
    observatory: bool,
    perkeo: bool,
    deck_key: *const c_char,
    erratic: bool,
    no_faces: bool,
    min_face_cards: c_int,
    suit_ratio: c_double,
    num_seeds: c_longlong,
    threads: c_int,
) -> *const c_char {
    let seed_start = c_string_lossy(seed_start);
    let voucher_key = c_string_lossy(voucher_key);
    let pack_key = c_string_lossy(pack_key);
    let tag1_key = c_string_lossy(tag1_key);
    let tag2_key = c_string_lossy(tag2_key);
    let joker_name = c_string_lossy(joker_name);
    let joker_location = c_string_lossy(joker_location);
    let deck_key = c_string_lossy(deck_key);

    let cfg = FilterConfig::from_raw(
        &voucher_key,
        &pack_key,
        &tag1_key,
        &tag2_key,
        &joker_name,
        &joker_location,
        souls,
        observatory,
        perkeo,
        &deck_key,
        erratic,
        no_faces,
        min_face_cards,
        suit_ratio,
    );

    let Some(result) = brainstorm_search_core(&seed_start, &cfg, num_seeds, threads) else {
        return ptr::null();
    };
    if result.is_empty() {
        return ptr::null();
    }
    match CString::new(result) {
        Ok(result) => result.into_raw().cast_const(),
        Err(_) => ptr::null(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn immolate_set_log_path(_path: *const c_char) {}

#[unsafe(no_mangle)]
pub extern "C" fn free_result(result: *const c_char) {
    if result.is_null() {
        return;
    }
    unsafe {
        drop(CString::from_raw(result.cast_mut()));
    }
}

fn c_string_lossy(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
}
