/*
 * Test Balatro RNG Implementation
 */

#include <iostream>
#include <iomanip>
#include "ImmolateCPP/src/balatro_rng.hpp"

void test_pseudohash() {
    std::cout << "Testing pseudohash function:" << std::endl;
    
    const char* test_seeds[] = {
        "AAAAAAAA",
        "00000000", 
        "ZZZZZZZZ",
        "1RP6DY6Y",
        "TUTORIAL"
    };
    
    for (const char* seed : test_seeds) {
        double hash = pseudohash(seed);
        std::cout << "  " << seed << " -> " << std::fixed << std::setprecision(13) << hash << std::endl;
    }
}

void test_rng_generation() {
    std::cout << "\nTesting RNG generation for seed '1RP6DY6Y':" << std::endl;
    
    BalatroRNG rng("1RP6DY6Y");
    
    // Test voucher generation
    uint32_t voucher = rng.get_pool_index("Voucher", 32);
    std::cout << "  Voucher index: " << voucher << std::endl;
    
    // Test pack generation
    uint32_t pack = rng.get_pool_index("shop_pack1", 15);
    std::cout << "  Pack index: " << pack << std::endl;
    
    // Test tag generation
    uint32_t tag_small = rng.get_pool_index("Tag_small", 30);
    uint32_t tag_big = rng.get_pool_index("Tag_big", 30);
    std::cout << "  Small tag index: " << tag_small << std::endl;
    std::cout << "  Big tag index: " << tag_big << std::endl;
}

void test_context_strings() {
    std::cout << "\nContext strings being used:" << std::endl;
    std::cout << "  Voucher: " << BalatroContext::VOUCHER << std::endl;
    std::cout << "  Pack (shop): " << BalatroContext::PACK_SHOP << std::endl;
    std::cout << "  Tag: " << BalatroContext::TAG << std::endl;
}

int main() {
    std::cout << "=== Balatro RNG Test ===" << std::endl << std::endl;
    
    test_pseudohash();
    test_rng_generation();
    test_context_strings();
    
    std::cout << "\n✓ Tests complete" << std::endl;
    
    return 0;
}