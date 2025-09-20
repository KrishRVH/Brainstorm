/*
 * Test Balatro RNG Implementation with Weighted Pack Selection
 */

#include <iostream>
#include <iomanip>
#include "ImmolateCPP/src/balatro_rng.hpp"
#include "ImmolateCPP/src/pack_definitions.hpp"

void test_specific_seed() {
    std::cout << "Testing seed '1RP6DY6Y' (from game logs):" << std::endl;
    
    BalatroRNG rng("1RP6DY6Y");
    
    // Test voucher generation
    uint32_t voucher = rng.get_pool_index("Voucher", 32);
    std::cout << "  Voucher index: " << voucher << std::endl;
    
    // Test pack generation with weighted selection
    std::string pack1 = get_next_pack_key(rng, 1);
    std::string pack2 = get_next_pack_key(rng, 1);
    std::cout << "  Pack 1 key: " << pack1 << std::endl;
    std::cout << "  Pack 2 key: " << pack2 << std::endl;
    
    // Test tag generation
    uint32_t tag_small = rng.get_pool_index("Tag_small", 30);
    uint32_t tag_big = rng.get_pool_index("Tag_big", 30);
    std::cout << "  Small tag index: " << tag_small << std::endl;
    std::cout << "  Big tag index: " << tag_big << std::endl;
}

void test_pack_matching() {
    std::cout << "\nTesting pack variant matching:" << std::endl;
    
    // Test spectral pack matching
    std::vector<std::string> spectral_keys = get_pack_variant_keys("Spectral Pack");
    std::cout << "  Spectral Pack variants: ";
    for (const auto& key : spectral_keys) {
        std::cout << key << " ";
    }
    std::cout << std::endl;
    
    // Test jumbo spectral pack
    std::vector<std::string> jumbo_spectral = get_pack_variant_keys("Jumbo Spectral Pack");
    std::cout << "  Jumbo Spectral Pack variants: ";
    for (const auto& key : jumbo_spectral) {
        std::cout << key << " ";
    }
    std::cout << std::endl;
    
    // Test matching
    bool match1 = pack_matches_filter("p_spectral_normal_1", "Spectral Pack");
    bool match2 = pack_matches_filter("p_spectral_normal_2", "Spectral Pack");
    bool match3 = pack_matches_filter("p_spectral_jumbo_1", "Spectral Pack");
    bool match4 = pack_matches_filter("p_spectral_jumbo_1", "Jumbo Spectral Pack");
    
    std::cout << "  p_spectral_normal_1 matches 'Spectral Pack': " << (match1 ? "YES" : "NO") << std::endl;
    std::cout << "  p_spectral_normal_2 matches 'Spectral Pack': " << (match2 ? "YES" : "NO") << std::endl;
    std::cout << "  p_spectral_jumbo_1 matches 'Spectral Pack': " << (match3 ? "NO (correct)" : "YES (wrong)") << std::endl;
    std::cout << "  p_spectral_jumbo_1 matches 'Jumbo Spectral Pack': " << (match4 ? "YES" : "NO") << std::endl;
}

void test_weighted_distribution() {
    std::cout << "\nTesting weighted pack selection (100 samples):" << std::endl;
    
    BalatroRNG rng("TESTTEST");
    
    int normal_count = 0;
    int jumbo_count = 0;
    int mega_count = 0;
    
    PackType spectral = PackType::Spectral;
    
    for (int i = 0; i < 100; i++) {
        std::string pack = get_next_pack_key(rng, 1, &spectral);
        
        if (pack.find("normal") != std::string::npos) normal_count++;
        else if (pack.find("jumbo") != std::string::npos) jumbo_count++;
        else if (pack.find("mega") != std::string::npos) mega_count++;
    }
    
    std::cout << "  Normal packs: " << normal_count << " (expected ~60 with weight 0.6)" << std::endl;
    std::cout << "  Jumbo packs: " << jumbo_count << " (expected ~30 with weight 0.3)" << std::endl;
    std::cout << "  Mega packs: " << mega_count << " (expected ~7 with weight 0.07)" << std::endl;
}

int main() {
    std::cout << "=== Balatro RNG Implementation Test ===\n" << std::endl;
    
    test_specific_seed();
    test_pack_matching();
    test_weighted_distribution();
    
    std::cout << "\n✓ Tests complete" << std::endl;
    
    return 0;
}