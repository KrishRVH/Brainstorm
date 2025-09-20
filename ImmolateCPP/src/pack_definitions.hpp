/*
 * Pack Definitions and Weighted Selection
 * Matches Balatro's actual pack system
 */

#ifndef PACK_DEFINITIONS_HPP
#define PACK_DEFINITIONS_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

// Pack type enum matching Balatro's indices
enum class PackType : uint32_t {
    Arcana = 0,
    Celestial = 1,
    Spectral = 2,
    Standard = 3,
    Buffoon = 4
};

// Pack variant with weight
struct PackVariant {
    std::string key;
    double weight;
    PackType type;
    bool is_jumbo;
    bool is_mega;
};

// All pack variants with their weights from game.lua
const std::vector<PackVariant> ALL_PACK_VARIANTS = {
    // Arcana Packs
    {"p_arcana_normal_1", 1.0, PackType::Arcana, false, false},
    {"p_arcana_normal_2", 1.0, PackType::Arcana, false, false},
    {"p_arcana_normal_3", 1.0, PackType::Arcana, false, false},
    {"p_arcana_normal_4", 1.0, PackType::Arcana, false, false},
    {"p_arcana_jumbo_1", 1.0, PackType::Arcana, true, false},
    {"p_arcana_jumbo_2", 1.0, PackType::Arcana, true, false},
    {"p_arcana_mega_1", 0.25, PackType::Arcana, false, true},
    {"p_arcana_mega_2", 0.25, PackType::Arcana, false, true},
    
    // Celestial Packs
    {"p_celestial_normal_1", 1.0, PackType::Celestial, false, false},
    {"p_celestial_normal_2", 1.0, PackType::Celestial, false, false},
    {"p_celestial_normal_3", 1.0, PackType::Celestial, false, false},
    {"p_celestial_normal_4", 1.0, PackType::Celestial, false, false},
    {"p_celestial_jumbo_1", 1.0, PackType::Celestial, true, false},
    {"p_celestial_jumbo_2", 1.0, PackType::Celestial, true, false},
    {"p_celestial_mega_1", 0.25, PackType::Celestial, false, true},
    {"p_celestial_mega_2", 0.25, PackType::Celestial, false, true},
    
    // Spectral Packs - CORRECTED WEIGHTS
    {"p_spectral_normal_1", 0.3, PackType::Spectral, false, false},
    {"p_spectral_normal_2", 0.3, PackType::Spectral, false, false},
    {"p_spectral_jumbo_1", 0.3, PackType::Spectral, true, false},  // Actually 0.3, not 0.15
    {"p_spectral_mega_1", 0.07, PackType::Spectral, false, true},   // Actually 0.07, not 0.0375
    
    // Standard Packs
    {"p_standard_normal_1", 1.0, PackType::Standard, false, false},
    {"p_standard_normal_2", 1.0, PackType::Standard, false, false},
    {"p_standard_normal_3", 1.0, PackType::Standard, false, false},
    {"p_standard_normal_4", 1.0, PackType::Standard, false, false},
    {"p_standard_jumbo_1", 1.0, PackType::Standard, true, false},  // Weight 1.0, not 0.5
    {"p_standard_jumbo_2", 1.0, PackType::Standard, true, false},  // Weight 1.0, not 0.5
    {"p_standard_mega_1", 0.25, PackType::Standard, false, true},  // Weight 0.25, not 0.125
    {"p_standard_mega_2", 0.25, PackType::Standard, false, true},  // Weight 0.25, not 0.125
    
    // Buffoon Packs
    {"p_buffoon_normal_1", 0.6, PackType::Buffoon, false, false},
    {"p_buffoon_normal_2", 0.6, PackType::Buffoon, false, false},
    {"p_buffoon_jumbo_1", 0.6, PackType::Buffoon, true, false},   // Weight 0.6, not 0.3
    {"p_buffoon_mega_1", 0.15, PackType::Buffoon, false, true}    // Weight 0.15, not 0.075
};

// Map from user-friendly pack names to PackType
const std::unordered_map<std::string, PackType> PACK_NAME_TO_TYPE = {
    {"Arcana Pack", PackType::Arcana},
    {"Celestial Pack", PackType::Celestial},
    {"Spectral Pack", PackType::Spectral},
    {"Standard Pack", PackType::Standard},
    {"Buffoon Pack", PackType::Buffoon},
    
    // Jumbo variants
    {"Jumbo Arcana Pack", PackType::Arcana},
    {"Jumbo Celestial Pack", PackType::Celestial},
    {"Jumbo Spectral Pack", PackType::Spectral},
    {"Jumbo Standard Pack", PackType::Standard},
    {"Jumbo Buffoon Pack", PackType::Buffoon},
    
    // Mega variants
    {"Mega Arcana Pack", PackType::Arcana},
    {"Mega Celestial Pack", PackType::Celestial},
    {"Mega Spectral Pack", PackType::Spectral},
    {"Mega Standard Pack", PackType::Standard},
    {"Mega Buffoon Pack", PackType::Buffoon}
};

// Check if a pack name is jumbo
inline bool is_jumbo_pack(const std::string& name) {
    return name.find("Jumbo") != std::string::npos;
}

// Check if a pack name is mega
inline bool is_mega_pack(const std::string& name) {
    return name.find("Mega") != std::string::npos;
}

// Get all valid variant keys for a pack name
inline std::vector<std::string> get_pack_variant_keys(const std::string& pack_name) {
    std::vector<std::string> result;
    
    auto it = PACK_NAME_TO_TYPE.find(pack_name);
    if (it == PACK_NAME_TO_TYPE.end()) {
        return result; // Empty if not found
    }
    
    PackType type = it->second;
    bool want_jumbo = is_jumbo_pack(pack_name);
    bool want_mega = is_mega_pack(pack_name);
    
    for (const auto& variant : ALL_PACK_VARIANTS) {
        if (variant.type == type) {
            // Match the size variant
            if (want_mega && variant.is_mega) {
                result.push_back(variant.key);
            } else if (want_jumbo && variant.is_jumbo && !variant.is_mega) {
                result.push_back(variant.key);
            } else if (!want_jumbo && !want_mega && !variant.is_jumbo && !variant.is_mega) {
                result.push_back(variant.key);
            }
        }
    }
    
    return result;
}

#endif // PACK_DEFINITIONS_HPP