#include "functions.hpp"
#include "immolate.hpp"
#include "instance.hpp"
#include "items.hpp"
#include "search.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

namespace {

struct FilterMetrics {
    long long seeds_evaluated = 0;
    long long tag_checks = 0;
    long long voucher_checks = 0;
    long long pack_checks = 0;
    long long perkeo_checks = 0;
    long long souls_checks = 0;
    long long passes = 0;
};

struct FilterConfig {
    Item voucher = Item::RETRY;
    Item pack = Item::RETRY;
    Item tag1 = Item::RETRY;
    Item tag2 = Item::RETRY;
    long souls = 0;
    bool observatory = false;
    bool perkeo = false;
};

Item parse_item_or_retry(const std::string& name) {
    if (name.empty()) {
        return Item::RETRY;
    }
    return stringToItem(name);
}

FilterConfig make_config(const std::string& voucher,
                         const std::string& pack,
                         const std::string& tag1,
                         const std::string& tag2,
                         double souls,
                         bool observatory,
                         bool perkeo) {
    FilterConfig cfg;
    cfg.voucher = parse_item_or_retry(voucher);
    cfg.pack = parse_item_or_retry(pack);
    cfg.tag1 = parse_item_or_retry(tag1);
    cfg.tag2 = parse_item_or_retry(tag2);
    cfg.souls = (souls > 0) ? static_cast<long>(souls) : 0;
    cfg.observatory = observatory;
    cfg.perkeo = perkeo;
    return cfg;
}

int apply_filters(Instance& inst, const FilterConfig& cfg, FilterMetrics* metrics) {
    // Pull blind tags once to avoid multiple RNG advances and reuse for all checks.
    const bool needs_tags =
        (cfg.tag1 != Item::RETRY || cfg.tag2 != Item::RETRY || cfg.perkeo);
    Item small_blind = Item::RETRY;
    Item big_blind = Item::RETRY;
    if (needs_tags) {
        small_blind = inst.nextTag(1);
        big_blind = inst.nextTag(1);
    }

    // Tag checks (order agnostic, supports duplicate tag requirement)
    if (cfg.tag1 != Item::RETRY || cfg.tag2 != Item::RETRY) {
        if (metrics) {
            metrics->tag_checks++;
        }
        if (cfg.tag2 == Item::RETRY) {
            if (small_blind != cfg.tag1 && big_blind != cfg.tag1) {
                return 0;
            }
        } else if (cfg.tag1 != cfg.tag2) {
            const bool has_tag1 = small_blind == cfg.tag1 || big_blind == cfg.tag1;
            const bool has_tag2 = small_blind == cfg.tag2 || big_blind == cfg.tag2;
            if (!has_tag1 || !has_tag2) {
                return 0;
            }
        } else {
            // same tag required twice
            if (small_blind != cfg.tag1 || big_blind != cfg.tag1) {
                return 0;
            }
        }
    }

    // Voucher check (first voucher in ante 1)
    if (cfg.voucher != Item::RETRY) {
        if (metrics) {
            metrics->voucher_checks++;
        }
        inst.initLocks(1, false, false);
        const Item first_voucher = inst.nextVoucher(1);
        if (first_voucher != cfg.voucher) {
            return 0;
        }
    }

    // Pack check: simulate two pack slots in shop (ante 1) and require at least one match.
    if (cfg.pack != Item::RETRY) {
        if (metrics) {
            metrics->pack_checks++;
        }
        const Item pack_slot_1 = inst.nextPack(1);
        const Item pack_slot_2 = inst.nextPack(1);
        const bool pack_match = (pack_slot_1 == cfg.pack) || (pack_slot_2 == cfg.pack);
        if (!pack_match) {
            return 0;
        }
    }

    // Observatory setup (Telescope + Mega Celestial Pack)
    if (cfg.observatory) {
        inst.initLocks(1, false, false);
        if (inst.nextVoucher(1) != Item::Telescope) {
            return 0;
        }
        const Item pack_slot_1 = inst.nextPack(1);
        const Item pack_slot_2 = inst.nextPack(1);
        const bool has_celestial =
            (pack_slot_1 == Item::Mega_Celestial_Pack) || (pack_slot_2 == Item::Mega_Celestial_Pack);
        if (!has_celestial) {
            return 0;
        }
    }

    // Perkeo setup (Investment tag + soul in Arcana)
    if (cfg.perkeo) {
        if (metrics) {
            metrics->perkeo_checks++;
        }
        if (small_blind != Item::Investment_Tag && big_blind != Item::Investment_Tag) {
            return 0;
        }

        const auto tarots = inst.nextArcanaPack(5, 1);
        const bool found_soul =
            std::any_of(tarots.begin(), tarots.end(), [](Item item) { return item == Item::The_Soul; });
        if (!found_soul) {
            return 0;
        }
    }

    // Soul count requirements (Arcana packs)
    if (cfg.souls > 0) {
        if (metrics) {
            metrics->souls_checks += cfg.souls;
        }
        for (long i = 0; i < cfg.souls; ++i) {
            const auto tarots = inst.nextArcanaPack(5, 1);
            const bool found_soul =
                std::any_of(tarots.begin(), tarots.end(), [](Item item) { return item == Item::The_Soul; });
            if (!found_soul) {
                return 0;
            }
        }
    }

    if (metrics) {
        metrics->passes++;
    }
    return 1;
}

std::string search_cpu(const std::string& seed, const FilterConfig& cfg, FilterMetrics& metrics) {
    auto filter_fn = [&cfg, &metrics](Instance& inst) -> int {
        metrics.seeds_evaluated++;
        return apply_filters(inst, cfg, &metrics);
    };
    Search search(filter_fn, seed, 1, 100000000);
    search.exitOnFind = true;
    return search.search();
}

bool debug_enabled() {
    return std::getenv("BRAINSTORM_CPU_DEBUG") != nullptr;
}

void log_debug(const std::string& line) {
    if (!debug_enabled()) {
        return;
    }
    std::ofstream out("brainstorm_cpu.log", std::ios::app);
    if (out.is_open()) {
        out << line << "\n";
    }
}

}  // namespace

extern "C" {

IMMOLATE_API const char* brainstorm(const char* seed,
                                    const char* voucher,
                                    const char* pack,
                                    const char* tag1,
                                    const char* tag2,
                                    double souls,
                                    bool observatory,
                                    bool perkeo) {
    const std::string cpp_seed(seed ? seed : "");
    const std::string cpp_voucher(voucher ? voucher : "");
    const std::string cpp_pack(pack ? pack : "");
    const std::string cpp_tag1(tag1 ? tag1 : "");
    const std::string cpp_tag2(tag2 ? tag2 : "");

    FilterConfig cfg =
        make_config(cpp_voucher, cpp_pack, cpp_tag1, cpp_tag2, souls, observatory, perkeo);
    FilterMetrics metrics;
    const std::string result = search_cpu(cpp_seed, cfg, metrics);

    if (result.empty()) {
        return nullptr;
    }

    char* output = static_cast<char*>(std::malloc(result.size() + 1));
    if (!output) {
        return nullptr;
    }
    std::memcpy(output, result.c_str(), result.size() + 1);

    if (debug_enabled()) {
        log_debug("==== brainstorm_cpu ====");
        log_debug("seed_in=" + cpp_seed + " seed_out=" + result);
        log_debug("voucher=" + itemToString(cfg.voucher) + " pack=" + itemToString(cfg.pack) +
                  " tag1=" + itemToString(cfg.tag1) + " tag2=" + itemToString(cfg.tag2));
        log_debug("souls=" + std::to_string(cfg.souls) +
                  " observatory=" + std::to_string(cfg.observatory) +
                  " perkeo=" + std::to_string(cfg.perkeo));
        log_debug("metrics: seeds=" + std::to_string(metrics.seeds_evaluated) +
                  " passes=" + std::to_string(metrics.passes) +
                  " tag_checks=" + std::to_string(metrics.tag_checks) +
                  " voucher_checks=" + std::to_string(metrics.voucher_checks) +
                  " pack_checks=" + std::to_string(metrics.pack_checks) +
                  " perkeo_checks=" + std::to_string(metrics.perkeo_checks) +
                  " souls_checks=" + std::to_string(metrics.souls_checks));
    }
    return output;
}

IMMOLATE_API const char* get_tags(const char* seed) {
    const std::string cpp_seed(seed ? seed : "");
    Seed s(cpp_seed);
    Instance inst(s);
    const Item small = inst.nextTag(1);
    const Item big = inst.nextTag(1);
    const std::string formatted = itemToString(small) + "|" + itemToString(big);
    char* output = static_cast<char*>(std::malloc(formatted.size() + 1));
    if (!output) {
        return nullptr;
    }
    std::memcpy(output, formatted.c_str(), formatted.size() + 1);
    return output;
}

IMMOLATE_API void free_result(const char* result) {
    if (result) {
        std::free(const_cast<char*>(result));
    }
}

IMMOLATE_API int get_acceleration_type() {
    return 0;  // CPU-only build
}

IMMOLATE_API const char* get_hardware_info() {
    static const char kInfo[] = "CPU-only build (CUDA disabled)";
    return kInfo;
}

IMMOLATE_API void set_use_cuda(bool /*enable*/) {
    // No-op for CPU-only build; exists for API compatibility.
}

}  // extern "C"
