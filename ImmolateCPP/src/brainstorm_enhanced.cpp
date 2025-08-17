// Brainstorm Enhanced DLL - High-performance seed filtering for Balatro
// Author: Community Edition v3.0
// Description: Provides native code acceleration for seed searching,
//              including dual tag support for 10-100x performance improvement

#include "functions.hpp"
#include "search.hpp"
#include <cstring>
#include <cstdlib>
#include <vector>

// Define IMMOLATE_API if not already defined
#ifndef IMMOLATE_API
#ifdef _WIN32
#ifdef BUILDING_DLL
#define IMMOLATE_API __declspec(dllexport)
#else
#define IMMOLATE_API __declspec(dllimport)
#endif
#else
#define IMMOLATE_API
#endif
#endif

// Global filter settings for the search
// These are set by the brainstorm function and used by the filter callback
// Item::RETRY is used as a sentinel value meaning "no filter"
Item BRAINSTORM_VOUCHER = Item::RETRY;      // First shop voucher requirement
Item BRAINSTORM_PACK = Item::RETRY;         // First pack type requirement
Item BRAINSTORM_TAG1 = Item::RETRY;         // First blind tag requirement
Item BRAINSTORM_TAG2 = Item::RETRY;         // Second blind tag requirement
long BRAINSTORM_SOULS = 0;                  // Number of shops with The Soul
bool BRAINSTORM_OBSERVATORY = false;        // Telescope + Mega Celestial combo
bool BRAINSTORM_PERKEO = false;             // Investment Tag + The Soul combo

// Structure to hold dual tag information
struct DualTagResult {
    std::string seed;
    std::string small_blind_tag;
    std::string big_blind_tag;
    bool matches;
};

// Check if seed matches all filters
long filter(Instance inst) {
    // Early exit optimization: Check tags first (cheapest operation)
    if (BRAINSTORM_TAG1 != Item::RETRY || BRAINSTORM_TAG2 != Item::RETRY) {
        // Get both tags for ante 1
        Item smallBlindTag = inst.nextTag(1);
        Item bigBlindTag = inst.nextTag(1);
        
        // If only one tag specified, check if it exists in either position
        if (BRAINSTORM_TAG2 == Item::RETRY) {
            if (smallBlindTag != BRAINSTORM_TAG1 && bigBlindTag != BRAINSTORM_TAG1) {
                return 0;
            }
        }
        // If two different tags specified, check both are present (order doesn't matter)
        else if (BRAINSTORM_TAG1 != BRAINSTORM_TAG2) {
            bool hasTag1 = (smallBlindTag == BRAINSTORM_TAG1 || bigBlindTag == BRAINSTORM_TAG1);
            bool hasTag2 = (smallBlindTag == BRAINSTORM_TAG2 || bigBlindTag == BRAINSTORM_TAG2);
            if (!hasTag1 || !hasTag2) {
                return 0;
            }
        }
        // If same tag specified twice, both positions must have it
        else {
            if (smallBlindTag != BRAINSTORM_TAG1 || bigBlindTag != BRAINSTORM_TAG1) {
                return 0;
            }
        }
    }
    
    // Check voucher if specified (second cheapest)
    if (BRAINSTORM_VOUCHER != Item::RETRY) {
        inst.initLocks(1, false, false);
        Item firstVoucher = inst.nextVoucher(1);
        if (firstVoucher != BRAINSTORM_VOUCHER) {
            return 0;
        }
    }
    
    // Check pack if specified
    if (BRAINSTORM_PACK != Item::RETRY) {
        inst.cache.generatedFirstPack = true;
        if (inst.nextPack(1) != BRAINSTORM_PACK) {
            return 0;
        }
    }
    
    // Check Observatory condition (Telescope voucher and Planet pack)
    if (BRAINSTORM_OBSERVATORY) {
        inst.initLocks(1, false, false);
        Item firstVoucher = inst.nextVoucher(1);
        if (firstVoucher != Item::Telescope) {
            return 0;
        }
        inst.cache.generatedFirstPack = true;
        Item pack = inst.nextPack(1);
        if (pack != Item::Mega_Celestial_Pack) {
            return 0;
        }
    }
    
    // Check Perkeo condition (special combo for Joker duplication)
    // Requires: Investment Tag + The Soul in Mega Arcana Pack
    if (BRAINSTORM_PERKEO) {
        Item smallBlindTag = inst.nextTag(1);
        Item bigBlindTag = inst.nextTag(1);
        
        // Need at least one Investment Tag for money generation
        if (smallBlindTag != Item::Investment_Tag && bigBlindTag != Item::Investment_Tag) {
            return 0;
        }
        
        // Check for The Soul in Mega Arcana pack (for Legendary Joker)
        auto tarots = inst.nextArcanaPack(5, 1);
        bool found_soul = false;
        
        // Search through the 5 cards in the pack
        for (int t = 0; t < 5; t++) {
            if (tarots[t] == Item::The_Soul) {
                found_soul = true;
                break;
            }
        }
        
        if (!found_soul) {
            return 0;  // Perkeo combo requires The Soul
        }
    }
    
    // Check Souls condition
    if (BRAINSTORM_SOULS > 0) {
        for (int i = 1; i <= BRAINSTORM_SOULS; i++) {
            auto tarots = inst.nextArcanaPack(5, 1);
            bool found_soul = false;
            for (int t = 0; t < 5; t++) {
                if (tarots[t] == Item::The_Soul) {
                    found_soul = true;
                    break;
                }
            }
            if (!found_soul) {
                return 0;
            }
        }
    }
    
    return 1;
}

// Main brainstorm function - searches for matching seeds
IMMOLATE_API std::string brainstorm_cpp(
    std::string seed, 
    std::string voucher,
    std::string pack, 
    std::string tag1,
    std::string tag2,
    double souls,
    bool observatory,
    bool perkeo
) {
    BRAINSTORM_VOUCHER = stringToItem(voucher);
    BRAINSTORM_PACK = stringToItem(pack);
    BRAINSTORM_TAG1 = stringToItem(tag1);
    BRAINSTORM_TAG2 = stringToItem(tag2);
    BRAINSTORM_SOULS = souls;
    BRAINSTORM_OBSERVATORY = observatory;
    BRAINSTORM_PERKEO = perkeo;
    
    // Configure and run the search
    Search search(filter, seed, 1, 100000000);  // Search up to 100M seeds
    search.exitOnFind = true;  // Stop on first match for performance
    return search.search();
}

// Get tags for a specific seed (for validation)
IMMOLATE_API DualTagResult get_tags_cpp(std::string seed) {
    Seed s(seed);
    Instance inst(s);
    Item smallBlindTag = inst.nextTag(1);
    Item bigBlindTag = inst.nextTag(1);
    
    DualTagResult result;
    result.seed = seed;
    result.small_blind_tag = itemToString(smallBlindTag);
    result.big_blind_tag = itemToString(bigBlindTag);
    result.matches = false; // Will be set by caller based on requirements
    
    return result;
}

// C interface for DLL export
// C interface for DLL export (used by LuaJIT FFI)
// All functions use C linkage to avoid name mangling
extern "C" {
    // Main search function with enhanced dual tag support
    IMMOLATE_API const char* brainstorm(
        const char* seed,
        const char* voucher,
        const char* pack,
        const char* tag1,
        const char* tag2,
        double souls,
        bool observatory,
        bool perkeo
    ) {
        std::string cpp_seed(seed);
        std::string cpp_voucher(voucher);
        std::string cpp_pack(pack);
        std::string cpp_tag1(tag1);
        std::string cpp_tag2(tag2);
        
        std::string result = brainstorm_cpp(
            cpp_seed, cpp_voucher, cpp_pack, 
            cpp_tag1, cpp_tag2, souls, 
            observatory, perkeo
        );
        
        // Use strdup for cleaner memory allocation
        return strdup(result.c_str());
    }
    
    // New function to get both tags for a seed
    IMMOLATE_API const char* get_tags(const char* seed) {
        std::string cpp_seed(seed);
        DualTagResult result = get_tags_cpp(cpp_seed);
        
        // Format as "small_tag|big_tag"
        std::string formatted = result.small_blind_tag + "|" + result.big_blind_tag;
        
        // Use strdup for cleaner memory allocation
        return strdup(formatted.c_str());
    }
    
    IMMOLATE_API void free_result(const char* result) {
        free((void*)result);
    }
}