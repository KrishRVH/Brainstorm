#include "functions.hpp"
#include "search.hpp"
#include <cstring>
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
Item BRAINSTORM_VOUCHER = Item::RETRY;
Item BRAINSTORM_PACK = Item::RETRY;
Item BRAINSTORM_TAG1 = Item::RETRY;
Item BRAINSTORM_TAG2 = Item::RETRY;
long BRAINSTORM_SOULS = 0;
bool BRAINSTORM_OBSERVATORY = false;
bool BRAINSTORM_PERKEO = false;

// Structure to hold dual tag information
struct DualTagResult {
    std::string seed;
    std::string small_blind_tag;
    std::string big_blind_tag;
    bool matches;
};

// Check if seed matches all filters
long filter(Instance inst) {
    // Check voucher if specified
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
    
    // Check tags if specified
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
    
    // Check Perkeo condition (Investment Tag and The Soul in Mega Arcana)
    if (BRAINSTORM_PERKEO) {
        Item smallBlindTag = inst.nextTag(1);
        Item bigBlindTag = inst.nextTag(1);
        
        // Need at least one Investment Tag
        if (smallBlindTag != Item::Investment_Tag && bigBlindTag != Item::Investment_Tag) {
            return 0;
        }
        
        // Check for The Soul in Mega Arcana pack
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
    
    Search search(filter, seed, 1, 100000000);
    search.exitOnFind = true;
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
extern "C" {
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
        
        char* c_result = (char*)malloc(result.length() + 1);
        strcpy(c_result, result.c_str());
        
        return c_result;
    }
    
    // New function to get both tags for a seed
    IMMOLATE_API const char* get_tags(const char* seed) {
        std::string cpp_seed(seed);
        DualTagResult result = get_tags_cpp(cpp_seed);
        
        // Format as "small_tag|big_tag"
        std::string formatted = result.small_blind_tag + "|" + result.big_blind_tag;
        
        char* c_result = (char*)malloc(formatted.length() + 1);
        strcpy(c_result, formatted.c_str());
        
        return c_result;
    }
    
    IMMOLATE_API void free_result(const char* result) {
        free((void*)result);
    }
}