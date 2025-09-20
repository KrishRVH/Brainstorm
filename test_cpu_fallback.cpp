/*
 * Test CPU Fallback with numeric seeds
 */

#include <iostream>
#include <cstdint>
#include <cstring>
#include "ImmolateCPP/src/gpu/seed_conversion.hpp"

// Test the conversion functions
void test_base36_conversion() {
    std::cout << "Testing base-36 conversion:" << std::endl;
    
    // Test some seeds with numbers
    const char* test_seeds[] = {
        "1RP6DY6Y",
        "8GD52JIF", 
        "9N6MPGM8",
        "00000000",
        "99999999",
        "ZZZZZZZZ",
        "1234ABCD"
    };
    
    for (const char* seed : test_seeds) {
        uint64_t num = seed_to_int(seed);
        char converted[9];
        int_to_seed(num, converted);
        
        std::cout << "  " << seed << " -> " << num << " -> " << converted;
        
        if (strcmp(seed, converted) == 0) {
            std::cout << " ✓" << std::endl;
        } else {
            std::cout << " ✗ MISMATCH!" << std::endl;
        }
    }
    
    // Test edge cases
    std::cout << "\nEdge cases:" << std::endl;
    std::cout << "  Max value (ZZZZZZZZ): " << seed_to_int("ZZZZZZZZ") 
              << " (expected: " << max_seed_value() << ")" << std::endl;
    std::cout << "  Min value (00000000): " << seed_to_int("00000000") 
              << " (expected: 0)" << std::endl;
}

// Test FNV-1a hash
uint32_t fnv1a8(const char seed[8]) {
    uint32_t hash = 2166136261u;
    for (int i = 0; i < 8; i++) {
        hash ^= (uint32_t)(uint8_t)seed[i];
        hash *= 16777619u;
    }
    return hash;
}

void test_hash() {
    std::cout << "\nTesting FNV-1a hash:" << std::endl;
    
    const char* seeds[] = {
        "1RP6DY6Y",
        "8GD52JIF",
        "AAAAAAAA",
        "00000000"
    };
    
    for (const char* seed : seeds) {
        uint32_t hash = fnv1a8(seed);
        std::cout << "  " << seed << " -> " << hash << std::endl;
    }
}

int main() {
    std::cout << "=== CPU Fallback Test ===" << std::endl << std::endl;
    
    test_base36_conversion();
    test_hash();
    
    std::cout << "\n✓ All tests complete" << std::endl;
    
    return 0;
}