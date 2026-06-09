#![allow(clippy::too_many_arguments)]

pub mod bench_cases;
mod ffi;
pub mod filters;
pub mod instance;
pub mod item;
pub mod rng;
pub mod search;
pub mod seed;
pub mod v2;
pub mod v3;

pub use filters::{FilterConfig, JokerLocation};
pub use search::{brainstorm_search_core, resolve_seed_budget, resolve_threads};

#[cfg(test)]
mod tests {
    #![allow(unsafe_code)]

    use super::*;
    use std::ffi::{CStr, CString};

    use crate::ffi::{brainstorm_search, free_result};
    use crate::item::{
        COMMON_JOKERS, COMMON_JOKERS_100, Item, LEGENDARY_JOKERS, RARE_JOKERS, RARE_JOKERS_100,
        UNCOMMON_JOKERS, UNCOMMON_JOKERS_100,
    };
    use crate::rng::{LuaRandom, fract, pseudohash, pseudohash_from, pseudostep, round13};
    use crate::seed::Seed;
    use crate::v2::brainstorm_search_core_v2;
    use crate::v3::brainstorm_search_core_v3;

    #[test]
    fn seed_order_starts_like_cpp() {
        let mut seed = Seed::default();
        assert_eq!(seed.to_string(), "");
        seed.next();
        assert_eq!(seed.to_string(), "1");
        seed.next();
        assert_eq!(seed.to_string(), "11");
    }

    #[test]
    fn seed_id_roundtrip_for_basic_seed() {
        for id in [0, 1, 2, 35, 36, 37, 1_000, 1_000_000] {
            let seed = Seed::from_id(id);
            assert_eq!(Seed::from_str(&seed.to_string()).id(), id);
        }
    }

    #[test]
    fn seed_id_strings_match_cpp_oracle() {
        let cases = [
            (0, ""),
            (1, "1"),
            (2, "11"),
            (35, "S1111111"),
            (36, "T1111111"),
            (37, "U1111111"),
            (1_000, "LS111111"),
            (1_000_000, "ZZNN1111"),
        ];
        for (id, expected) in cases {
            assert_eq!(Seed::from_id(id).to_string(), expected);
        }
    }

    #[test]
    fn seed_id_normalizes_at_seed_space_boundary() {
        assert_eq!(Seed::from_id(crate::seed::SEED_SPACE).to_string(), "");
        assert_eq!(Seed::from_id(crate::seed::SEED_SPACE + 1).to_string(), "1");
        assert_eq!(Seed::from_id(-1).to_string(), "ZZZZZZZZ");
    }

    #[test]
    fn no_filter_returns_start_seed_after_first_candidate() {
        let cfg = FilterConfig::default();
        assert_eq!(brainstorm_search_core("", &cfg, 1, 1).as_deref(), Some(""),);
        assert_eq!(
            brainstorm_search_core("1", &cfg, 1, 1).as_deref(),
            Some("1"),
        );
    }

    #[test]
    fn parsers_match_current_defaults_for_unknowns() {
        let cfg = FilterConfig::from_raw(
            "unknown", "unknown", "unknown", "", "unknown", "weird", 0.0, false, false, "unknown",
            false, false, -1, 2.0,
        );
        assert_eq!(cfg.voucher, Item::RETRY);
        assert_eq!(cfg.pack, Item::RETRY);
        assert_eq!(cfg.tag1, Item::RETRY);
        assert_eq!(cfg.joker, Item::RETRY);
        assert_eq!(cfg.deck, Item::Red_Deck);
        assert_eq!(cfg.min_face_cards, 0);
        assert_eq!(cfg.suit_ratio, 1.0);

        assert_eq!(
            FilterConfig::from_raw(
                "", "", "", "", "j_caino", "any", 0.0, false, false, "b_red", false, false, 0, 0.0
            )
            .joker,
            Item::Canio
        );
        assert_eq!(
            FilterConfig::from_raw(
                "", "", "", "", "j_seance", "any", 0.0, false, false, "b_red", false, false, 0, 0.0
            )
            .joker,
            Item::Seance
        );
        assert_eq!(
            FilterConfig::from_raw(
                "", "", "", "", "Seance", "any", 0.0, false, false, "b_red", false, false, 0, 0.0
            )
            .joker,
            Item::Seance
        );
    }

    #[test]
    fn rng_smoke_is_stable() {
        assert_eq!(pseudohash(""), 1.0);
        let mut rng = LuaRandom::new(0.5);
        let first = rng.random();
        assert!((0.0..1.0).contains(&first));
    }

    #[test]
    fn rng_vectors_match_cpp_oracle() {
        let hash_cases = [
            ("", 1.0),
            ("1", 0.15694342689690188),
            ("11", 0.68745689631282403),
            ("ABCDE", 0.55659692676272243),
            ("Tag1", 0.47049862973562995),
            ("shop_pack1", 0.39373360824367865),
            ("soul_Spectral1", 0.24677008613650742),
        ];
        for (input, expected) in hash_cases {
            assert_close(pseudohash(input), expected);
        }

        let mut rng = LuaRandom::new(0.5);
        assert_close(rng.random(), 0.09657393438653461);
        assert_close(rng.random(), 0.96226945770684003);
    }

    #[test]
    fn rng_primitive_goldens_are_stable() {
        assert_close(fract(12.3456789012345), 0.34567890123449985);
        assert_close(round13(0.12345678901234567), 0.1234567890123);
        assert_close(
            pseudohash_from("Tag1", 0.15694342689690188),
            0.61303326083861975,
        );
        assert_close(pseudostep(b'1', 1, 1.0), 0.15694342689690188);
    }

    #[test]
    fn search_vectors_match_cpp_oracle() {
        let empty = FilterConfig::default();
        assert_eq!(
            brainstorm_search_core("", &empty, 1, 1).as_deref(),
            Some("")
        );
        assert_eq!(
            brainstorm_search_core("1", &empty, 1, 1).as_deref(),
            Some("1")
        );

        let tag = FilterConfig::from_raw(
            "",
            "",
            "tag_charm",
            "",
            "",
            "any",
            0.0,
            false,
            false,
            "b_red",
            false,
            false,
            0,
            0.0,
        );
        assert_eq!(
            brainstorm_search_core("", &tag, 10_000, 1).as_deref(),
            Some("21111111"),
        );

        let voucher = FilterConfig::from_raw(
            "v_telescope",
            "",
            "",
            "",
            "",
            "any",
            0.0,
            false,
            false,
            "b_red",
            false,
            false,
            0,
            0.0,
        );
        assert_eq!(
            brainstorm_search_core("", &voucher, 10_000, 1).as_deref(),
            Some("P1111111"),
        );

        let pack = FilterConfig::from_raw(
            "",
            "p_spectral_mega_1",
            "",
            "",
            "",
            "any",
            0.0,
            false,
            false,
            "b_red",
            false,
            false,
            0,
            0.0,
        );
        assert_eq!(
            brainstorm_search_core("", &pack, 10_000, 1).as_deref(),
            Some("Z2111111"),
        );

        let observatory = FilterConfig::from_raw(
            "", "", "", "", "", "any", 0.0, true, false, "b_red", false, false, 0, 0.0,
        );
        assert_eq!(
            brainstorm_search_core("", &observatory, 100_000, 1).as_deref(),
            Some("S111111"),
        );

        let erratic = FilterConfig::from_raw(
            "",
            "",
            "",
            "",
            "",
            "any",
            0.0,
            false,
            false,
            "b_erratic",
            true,
            false,
            12,
            0.0,
        );
        assert_eq!(
            brainstorm_search_core("", &erratic, 10_000, 1).as_deref(),
            Some("11"),
        );
    }

    #[test]
    fn v2_matches_current_core_for_search_vectors() {
        let cases = [
            ("", FilterConfig::default(), 1, Some("")),
            ("1", FilterConfig::default(), 1, Some("1")),
            (
                "",
                FilterConfig::from_raw(
                    "",
                    "",
                    "tag_charm",
                    "",
                    "",
                    "any",
                    0.0,
                    false,
                    false,
                    "b_red",
                    false,
                    false,
                    0,
                    0.0,
                ),
                10_000,
                Some("21111111"),
            ),
            (
                "",
                FilterConfig::from_raw(
                    "v_telescope",
                    "",
                    "",
                    "",
                    "",
                    "any",
                    0.0,
                    false,
                    false,
                    "b_red",
                    false,
                    false,
                    0,
                    0.0,
                ),
                10_000,
                Some("P1111111"),
            ),
            (
                "",
                FilterConfig::from_raw(
                    "",
                    "p_spectral_mega_1",
                    "",
                    "",
                    "",
                    "any",
                    0.0,
                    false,
                    false,
                    "b_red",
                    false,
                    false,
                    0,
                    0.0,
                ),
                10_000,
                Some("Z2111111"),
            ),
            (
                "",
                FilterConfig::from_raw(
                    "", "", "", "", "", "any", 0.0, true, false, "b_red", false, false, 0, 0.0,
                ),
                100_000,
                Some("S111111"),
            ),
            (
                "",
                FilterConfig::from_raw(
                    "",
                    "",
                    "",
                    "",
                    "",
                    "any",
                    0.0,
                    false,
                    false,
                    "b_erratic",
                    true,
                    false,
                    12,
                    0.0,
                ),
                10_000,
                Some("11"),
            ),
        ];

        for (seed_start, cfg, budget, expected) in cases {
            assert_eq!(
                brainstorm_search_core_v2(seed_start, &cfg, budget, 1).as_deref(),
                expected
            );
            assert_eq!(
                brainstorm_search_core_v2(seed_start, &cfg, budget, 1),
                brainstorm_search_core(seed_start, &cfg, budget, 1)
            );
        }
    }

    #[test]
    fn v2_matches_current_core_for_benchmark_cases() {
        for case in crate::bench_cases::bench_cases() {
            let cfg = FilterConfig::from_raw(
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
            );
            let budget = match case.name {
                "observatory" => 100_000,
                "shop-hit" | "any-joker" | "souls-arcana" => 50_000,
                "perkeo" => 20_000,
                _ => 10_000,
            };
            assert_eq!(
                brainstorm_search_core_v2(case.seed_start, &cfg, budget, 1),
                brainstorm_search_core(case.seed_start, &cfg, budget, 1),
                "V2 mismatch for benchmark case {}",
                case.name,
            );
        }
    }

    #[test]
    fn v2_falls_back_for_unsupported_filter_combinations() {
        use crate::v2::config::{CompiledFilter, KernelShape};

        let cases = [
            (
                "observatory+perkeo",
                FilterConfig::from_raw(
                    "", "", "", "", "", "any", 0.0, true, true, "b_red", false, false, 0, 0.0,
                ),
                100_000,
            ),
            (
                "souls+perkeo",
                FilterConfig::from_raw(
                    "", "", "", "", "", "any", 1.0, false, true, "b_red", false, false, 0, 0.0,
                ),
                100_000,
            ),
            (
                "shop-joker+perkeo",
                FilterConfig::from_raw(
                    "",
                    "",
                    "",
                    "",
                    "Burnt Joker",
                    "shop",
                    0.0,
                    false,
                    true,
                    "b_red",
                    false,
                    false,
                    0,
                    0.0,
                ),
                50_000,
            ),
            (
                "joker+observatory",
                FilterConfig::from_raw(
                    "",
                    "",
                    "",
                    "",
                    "Riff-raff",
                    "any",
                    0.0,
                    true,
                    false,
                    "b_red",
                    false,
                    false,
                    0,
                    0.0,
                ),
                100_000,
            ),
            (
                "pack+shop-joker",
                FilterConfig::from_raw(
                    "",
                    "p_spectral_mega_1",
                    "",
                    "",
                    "Burnt Joker",
                    "shop",
                    0.0,
                    false,
                    false,
                    "b_red",
                    false,
                    false,
                    0,
                    0.0,
                ),
                50_000,
            ),
        ];

        for (name, cfg, budget) in cases {
            assert_eq!(
                CompiledFilter::compile(&cfg).shape,
                KernelShape::Generic,
                "{name} should use generic fallback",
            );
            assert_eq!(
                brainstorm_search_core_v2("", &cfg, budget, 1),
                brainstorm_search_core("", &cfg, budget, 1),
                "V2 mismatch for {name}",
            );
        }
    }

    #[test]
    fn v2_rejects_static_no_match_filters() {
        use crate::v2::config::{CompiledFilter, KernelShape};

        let cases = [
            (
                "legendary shop joker",
                FilterConfig::from_raw(
                    "", "", "", "", "Perkeo", "shop", 0.0, false, false, "b_red", false, false, 0,
                    0.0,
                ),
            ),
            (
                "legendary pack joker",
                FilterConfig::from_raw(
                    "", "", "", "", "Perkeo", "pack", 0.0, false, false, "b_red", false, false, 0,
                    0.0,
                ),
            ),
            (
                "legendary any joker",
                FilterConfig::from_raw(
                    "", "", "", "", "Perkeo", "any", 0.0, false, false, "b_red", false, false, 0,
                    0.0,
                ),
            ),
            (
                "too many souls in selected pack",
                FilterConfig::from_raw(
                    "",
                    "p_spectral_normal_1",
                    "",
                    "",
                    "",
                    "any",
                    3.0,
                    false,
                    false,
                    "b_red",
                    false,
                    false,
                    0,
                    0.0,
                ),
            ),
            (
                "too many souls without selected pack",
                FilterConfig::from_raw(
                    "", "", "", "", "", "any", 6.0, false, false, "b_red", false, false, 0, 0.0,
                ),
            ),
            (
                "souls in non-soulable pack",
                FilterConfig::from_raw(
                    "",
                    "p_buffoon_mega_1",
                    "",
                    "",
                    "",
                    "any",
                    1.0,
                    false,
                    false,
                    "b_red",
                    false,
                    false,
                    0,
                    0.0,
                ),
            ),
            (
                "perkeo in non-soulable pack",
                FilterConfig::from_raw(
                    "",
                    "p_buffoon_mega_1",
                    "",
                    "",
                    "",
                    "any",
                    0.0,
                    false,
                    true,
                    "b_red",
                    false,
                    false,
                    0,
                    0.0,
                ),
            ),
            (
                "no faces with required face count",
                FilterConfig::from_raw(
                    "",
                    "",
                    "",
                    "",
                    "",
                    "any",
                    0.0,
                    false,
                    false,
                    "b_erratic",
                    true,
                    true,
                    1,
                    0.0,
                ),
            ),
        ];

        for (name, cfg) in cases {
            assert_eq!(
                CompiledFilter::compile(&cfg).shape,
                KernelShape::NoMatch,
                "{name} should be rejected before seed scanning",
            );
            assert_eq!(
                brainstorm_search_core_v2("", &cfg, 10_000, 1),
                brainstorm_search_core("", &cfg, 10_000, 1),
                "V2 mismatch for {name}",
            );
        }
    }

    #[test]
    fn v3_rejects_static_no_match_filters() {
        use crate::v3::config::{CompiledFilter, KernelShape};

        let cases = [
            (
                "ante-1 locked tag",
                FilterConfig::from_raw(
                    "",
                    "",
                    "tag_buffoon",
                    "",
                    "",
                    "any",
                    0.0,
                    false,
                    false,
                    "b_red",
                    false,
                    false,
                    0,
                    0.0,
                ),
            ),
            (
                "legendary shop joker",
                FilterConfig::from_raw(
                    "", "", "", "", "Perkeo", "shop", 0.0, false, false, "b_red", false, false, 0,
                    0.0,
                ),
            ),
            (
                "too many souls in selected pack",
                FilterConfig::from_raw(
                    "",
                    "p_spectral_normal_1",
                    "",
                    "",
                    "",
                    "any",
                    3.0,
                    false,
                    false,
                    "b_red",
                    false,
                    false,
                    0,
                    0.0,
                ),
            ),
        ];

        for (name, cfg) in cases {
            assert_eq!(
                CompiledFilter::compile(&cfg).shape,
                KernelShape::NoMatch,
                "{name} should be rejected before seed scanning",
            );
            assert_eq!(
                brainstorm_search_core_v3("", &cfg, 10_000, 1),
                brainstorm_search_core_v2("", &cfg, 10_000, 1),
                "V3 mismatch for {name}",
            );
        }
    }

    #[test]
    fn search_wraps_without_panicking_near_seed_space_end() {
        let cfg = FilterConfig::default();
        assert_eq!(
            brainstorm_search_core("ZZZZZZZZ", &cfg, 2, 1).as_deref(),
            Some("ZZZZZZZZ"),
        );
        assert_eq!(
            resolve_seed_budget(crate::seed::SEED_SPACE + 1),
            crate::seed::SEED_SPACE
        );
    }

    #[test]
    fn current_joker_pools_match_cpp_current_version_boundaries() {
        assert_eq!(COMMON_JOKERS.len(), 61);
        assert_eq!(COMMON_JOKERS_100.len(), 60);
        assert_eq!(COMMON_JOKERS[47], Item::Reserved_Parking);
        assert_eq!(COMMON_JOKERS[48], Item::Mail_In_Rebate);

        assert_eq!(UNCOMMON_JOKERS.len(), 64);
        assert_eq!(UNCOMMON_JOKERS_100.len(), 66);
        assert_eq!(UNCOMMON_JOKERS[14], Item::Sixth_Sense);
        assert_eq!(UNCOMMON_JOKERS[19], Item::Seance);
        assert!(!UNCOMMON_JOKERS.contains(&Item::Vagabond));
        assert!(!UNCOMMON_JOKERS.contains(&Item::Reserved_Parking));
        assert!(!UNCOMMON_JOKERS.contains(&Item::Stuntman));
        assert!(!UNCOMMON_JOKERS.contains(&Item::Burnt_Joker));

        assert_eq!(RARE_JOKERS.len(), 20);
        assert_eq!(RARE_JOKERS_100.len(), 19);
        assert_eq!(RARE_JOKERS[1], Item::Vagabond);
        assert_eq!(RARE_JOKERS[15], Item::Stuntman);
        assert_eq!(RARE_JOKERS[19], Item::Burnt_Joker);
        assert!(!RARE_JOKERS.contains(&Item::Sixth_Sense));
        assert!(!RARE_JOKERS.contains(&Item::Seance));
    }

    #[test]
    fn joker_rarity_pools_are_disjoint() {
        let pools: &[(&str, &[Item])] = &[
            ("common", &COMMON_JOKERS),
            ("uncommon", &UNCOMMON_JOKERS),
            ("rare", &RARE_JOKERS),
            ("legendary", &LEGENDARY_JOKERS),
        ];
        for i in 0..pools.len() {
            for j in (i + 1)..pools.len() {
                let (left_name, left) = pools[i];
                let (right_name, right) = pools[j];
                for item in left {
                    assert!(
                        !right.contains(item),
                        "{item:?} appears in both {left_name} and {right_name} joker pools",
                    );
                }
            }
        }
    }

    #[test]
    fn ffi_contract_matches_cpp_empty_and_allocated_results() {
        let empty = CString::new("").expect("literal has no interior nul");
        let one = CString::new("1").expect("literal has no interior nul");

        let empty_result = brainstorm_search(
            empty.as_ptr(),
            empty.as_ptr(),
            empty.as_ptr(),
            empty.as_ptr(),
            empty.as_ptr(),
            empty.as_ptr(),
            empty.as_ptr(),
            0.0,
            false,
            false,
            empty.as_ptr(),
            false,
            false,
            0,
            0.0,
            1,
            1,
        );
        assert!(empty_result.is_null());

        let one_result = brainstorm_search(
            one.as_ptr(),
            empty.as_ptr(),
            empty.as_ptr(),
            empty.as_ptr(),
            empty.as_ptr(),
            empty.as_ptr(),
            empty.as_ptr(),
            0.0,
            false,
            false,
            empty.as_ptr(),
            false,
            false,
            0,
            0.0,
            1,
            1,
        );
        assert!(!one_result.is_null());
        let result = unsafe { CStr::from_ptr(one_result) }
            .to_string_lossy()
            .into_owned();
        assert_eq!(result, "1");
        free_result(one_result);
        free_result(std::ptr::null());
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-15,
            "actual={actual:.17} expected={expected:.17}",
        );
    }
}
