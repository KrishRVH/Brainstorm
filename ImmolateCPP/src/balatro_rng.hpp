/*
 * Balatro RNG Implementation
 * Exact reproduction of Balatro's pseudorandom system
 */

#ifndef BALATRO_RNG_HPP
#define BALATRO_RNG_HPP

#include <cstdint>
#include <cmath>
#include <string>
#include <unordered_map>
#include <cstring>
#include <random>
#include "pack_definitions.hpp"

// Constants from Balatro
const double PI = 3.14159265358979323846;
const double HASH_MULT = 1.1239285023;
const double LCG_A = 2.134453429141;
const double LCG_B = 1.72431234;

// Balatro's pseudohash function (functions/misc_functions.lua:279)
inline double pseudohash(const std::string& str) {
    double num = 1.0;
    for (int i = str.length() - 1; i >= 0; i--) {
        double byte_val = static_cast<double>(static_cast<unsigned char>(str[i]));
        num = fmod((HASH_MULT / num) * byte_val * PI + PI * (i + 1), 1.0);
    }
    return num;
}

// Balatro's pseudohash for C-string
inline double pseudohash(const char* str) {
    double num = 1.0;
    int len = strlen(str);
    for (int i = len - 1; i >= 0; i--) {
        double byte_val = static_cast<double>(static_cast<unsigned char>(str[i]));
        num = fmod((HASH_MULT / num) * byte_val * PI + PI * (i + 1), 1.0);
    }
    return num;
}

// RNG state for pseudoseed (mimics G.GAME.pseudorandom table)
class BalatroRNG {
private:
    std::unordered_map<std::string, double> pseudorandom;
    std::string seed;
    double hashed_seed;
    
public:
    BalatroRNG(const std::string& game_seed) : seed(game_seed) {
        hashed_seed = pseudohash(seed);
    }
    
    // Balatro's pseudoseed function (functions/misc_functions.lua:298)
    double pseudoseed(const std::string& key) {
        // Initialize if not exists
        if (pseudorandom.find(key) == pseudorandom.end()) {
            pseudorandom[key] = pseudohash(key + seed);
        }
        
        // Update state with LCG
        double current = pseudorandom[key];
        current = fmod(LCG_A + current * LCG_B, 1.0);
        current = fabs(current);
        pseudorandom[key] = current;
        
        // Return average of state and hashed_seed
        return (current + hashed_seed) / 2.0;
    }
    
    // Get element from pool using Balatro's method
    uint32_t get_pool_index(const std::string& context, uint32_t pool_size) {
        double seed_val = pseudoseed(context);
        
        // Convert to integer seed for C++ RNG (mimics math.randomseed)
        uint32_t int_seed = static_cast<uint32_t>(seed_val * 4294967295.0);
        std::mt19937 gen(int_seed);
        std::uniform_int_distribution<uint32_t> dist(0, pool_size - 1);
        
        return dist(gen);
    }
};

// Context strings used by Balatro
namespace BalatroContext {
    const char* VOUCHER = "Voucher";
    const char* VOUCHER_FROM_TAG = "Voucher_fromtag";
    const char* TAG = "Tag";
    const char* PACK_SHOP = "shop_pack";
    const char* PACK_GENERIC = "pack_generic";
    
    // For tags, the actual keys depend on blind type
    const char* TAG_SMALL = "Tag_small";
    const char* TAG_BIG = "Tag_big";
    
    // These are wrong - Balatro doesn't use these!
    // const char* STANDARD_PACK = "StandardPack";
    // const char* JUMBO_STANDARD = "JumboStandardPack";
}

// Get voucher using Balatro's method
inline uint32_t get_next_voucher(BalatroRNG& rng) {
    return rng.get_pool_index(BalatroContext::VOUCHER, 32);
}

// Get pack using Balatro's weighted selection method
inline std::string get_next_pack_key(BalatroRNG& rng, uint32_t ante = 1, PackType* filter_type = nullptr) {
    // Calculate cumulative weight for valid packs
    double cume = 0.0;
    for (const auto& variant : ALL_PACK_VARIANTS) {
        if (!filter_type || variant.type == *filter_type) {
            cume += variant.weight;
        }
    }
    
    // Get random value using Balatro's RNG
    std::string context = std::string(BalatroContext::PACK_SHOP) + std::to_string(ante);
    double seed_val = rng.pseudoseed(context);
    double poll = seed_val * cume;
    
    // Find the selected pack
    double it = 0.0;
    for (const auto& variant : ALL_PACK_VARIANTS) {
        if (!filter_type || variant.type == *filter_type) {
            it += variant.weight;
            if (it >= poll && (it - variant.weight) <= poll) {
                return variant.key;
            }
        }
    }
    
    // Fallback (shouldn't happen)
    return "p_standard_normal_1";
}

// Check if a generated pack key matches what we're looking for
inline bool pack_matches_filter(const std::string& generated_key, const std::string& target_pack_name) {
    if (target_pack_name.empty()) return true;
    
    // Get all valid variant keys for the target pack
    std::vector<std::string> valid_keys = get_pack_variant_keys(target_pack_name);
    
    // Check if generated key is in the valid list
    for (const auto& key : valid_keys) {
        if (generated_key == key) {
            return true;
        }
    }
    
    return false;
}

// Get tag using Balatro's method
inline uint32_t get_next_tag(BalatroRNG& rng, const std::string& blind_type) {
    std::string context = std::string(BalatroContext::TAG) + "_" + blind_type;
    return rng.get_pool_index(context, 30);
}

#endif // BALATRO_RNG_HPP