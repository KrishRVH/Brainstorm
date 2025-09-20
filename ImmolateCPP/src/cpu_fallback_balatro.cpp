/*
 * CPU Fallback with Correct Balatro RNG
 * Uses the actual pseudohash/pseudoseed algorithm from the game
 */

#include "gpu/gpu_types.h"
#include "gpu/seed_conversion.hpp"
#include "balatro_rng.hpp"
#include <string>
#include <cstdint>
#include <cstdio>
#include <chrono>

// CPU-based seed search using Balatro's actual RNG
extern "C" std::string cpu_search_balatro(
    const std::string& start_seed_str,
    const FilterParams& params
) {
    FILE* log = fopen("C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm\\cpu_balatro.log", "a");
    if (log) {
        fprintf(log, "\n[CPU-BALATRO] Search started for seed: %s\n", start_seed_str.c_str());
        fprintf(log, "[CPU-BALATRO] Filters - tag1(s=%u,b=%u) tag2(s=%u,b=%u) voucher:%u pack1:%u pack2:%u\n",
                params.tag1_small, params.tag1_big, params.tag2_small, params.tag2_big, 
                params.voucher, params.pack1, params.pack2);
        fflush(log);
    }
    
    // Convert start seed to numeric
    uint64_t current_seed_num = seed_to_int(start_seed_str.c_str());
    
    const uint64_t MAX_SEEDS = 10000000;  // 10M seeds for CPU
    const uint64_t MAX_SEED_VALUE = 2821109907455ULL;  // 36^8 - 1
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (uint64_t i = 0; i < MAX_SEEDS; i++) {
        // Wrap around if we exceed max
        if (current_seed_num > MAX_SEED_VALUE) {
            current_seed_num = 0;
        }
        
        // Convert to string
        char seed_str[9];
        int_to_seed(current_seed_num, seed_str);
        
        // Create RNG with this seed
        BalatroRNG rng(seed_str);
        
        // Generate shop items using Balatro's actual method
        // Note: We need to get the actual pool selection working
        
        // For voucher
        uint32_t voucher_id = get_next_voucher(rng);
        
        // For packs - get both shop packs as keys
        std::string pack1_key = get_next_pack_key(rng, 1);
        std::string pack2_key = get_next_pack_key(rng, 1);  // Second pack in shop
        
        // For tags - we need both small and big
        uint32_t small_tag = get_next_tag(rng, "small");
        uint32_t big_tag = get_next_tag(rng, "big");
        
        // Check filters
        bool match = true;
        
        // Tag filter (per-context indices)
        bool want_t1 = (params.tag1_small != 0xFFFFFFFF) || (params.tag1_big != 0xFFFFFFFF);
        bool want_t2 = (params.tag2_small != 0xFFFFFFFF) || (params.tag2_big != 0xFFFFFFFF);
        
        if (want_t1) {
            bool has_t1 = ((params.tag1_small != 0xFFFFFFFF) && (small_tag == params.tag1_small)) ||
                          ((params.tag1_big   != 0xFFFFFFFF) && (big_tag   == params.tag1_big));
            if (!has_t1) match = false;
        }
        
        if (match && want_t2) {
            bool has_t2 = ((params.tag2_small != 0xFFFFFFFF) && (small_tag == params.tag2_small)) ||
                          ((params.tag2_big   != 0xFFFFFFFF) && (big_tag   == params.tag2_big));
            if (!has_t2) match = false;
        }
        
        // Voucher filter
        if (match && params.voucher != 0xFFFFFFFF) {
            match = (voucher_id == params.voucher);
        }
        
        // Pack filter - check against generated keys
        // Note: params.pack is still an index, need to convert to check
        // For now, skip pack filter until we fix the index->key mapping
        // TODO: Implement proper pack key matching
        
        if (match) {
            if (log) {
                fprintf(log, "[CPU-BALATRO] FOUND MATCH: %s (numeric: %llu)\n", seed_str, current_seed_num);
                fprintf(log, "[CPU-BALATRO] Values - small_tag:%u big_tag:%u voucher:%u\n",
                        small_tag, big_tag, voucher_id);
                fprintf(log, "[CPU-BALATRO] Pack keys: %s, %s\n", pack1_key.c_str(), pack2_key.c_str());
                fclose(log);
            }
            return std::string(seed_str);
        }
        
        // Increment seed
        current_seed_num++;
        
        // Check timeout (10 seconds for CPU)
        if (i % 100000 == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - start_time
            ).count();
            
            if (elapsed > 10) {
                if (log) {
                    fprintf(log, "[CPU-BALATRO] Timeout after %lld seconds, searched %llu seeds\n", 
                            elapsed, i);
                    fclose(log);
                }
                break;
            }
        }
    }
    
    if (log) {
        fprintf(log, "[CPU-BALATRO] No match found in search\n");
        fclose(log);
    }
    
    return "";  // Not found
}