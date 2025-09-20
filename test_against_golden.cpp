#include <iostream>
#include <cmath>
#include <string>
#include <iomanip>
#include "ImmolateCPP/src/balatro_exact.hpp"

// Golden values from actual game capture
struct GoldenTest {
    std::string seed;
    std::string key;
    double expected;
    int call_number;
};

int main() {
    std::cout << std::fixed << std::setprecision(17);
    
    // Test pseudohash first
    std::cout << "=== Testing pseudohash ===\n";
    
    struct HashTest {
        std::string seed;
        double expected;
    } hash_tests[] = {
        {"AAAAAAAA", 0.43257138351543745},
        {"7NTPKW6P", 0.17960061211027778},
        {"00000000", 0.1253387937352386}
    };
    
    bool all_hash_pass = true;
    for (const auto& test : hash_tests) {
        double result = pseudohash_exact(test.seed);
        bool match = std::abs(result - test.expected) < 1e-15;
        
        std::cout << "pseudohash('" << test.seed << "'):\n";
        std::cout << "  Expected: " << test.expected << "\n";
        std::cout << "  Got:      " << result << "\n";
        std::cout << "  Match:    " << (match ? "✓" : "✗") << "\n\n";
        
        if (!match) all_hash_pass = false;
    }
    
    std::cout << "=== Testing pseudoseed ===\n";
    std::cout << "With seed 'AAAAAAAA':\n\n";
    
    // Initialize RNG state
    BalatroExactRNG rng("AAAAAAAA");
    
    // Test pseudoseed - these are the ACTUAL values from the game
    GoldenTest seed_tests[] = {
        {"AAAAAAAA", "Voucher", 0.24901819449170001, 1},
        {"AAAAAAAA", "shop_pack1", 0.38681086553959998, 1},
        {"AAAAAAAA", "shop_pack1", 0.2342094632665, 2}
    };
    
    // Test Voucher
    double voucher_result = rng.pseudoseed("Voucher");
    
    std::cout << "pseudoseed('Voucher'):\n";
    std::cout << "  Expected: " << seed_tests[0].expected << "\n";
    std::cout << "  Got:      " << voucher_result << "\n";
    std::cout << "  Match:    " << (std::abs(voucher_result - seed_tests[0].expected) < 1e-15 ? "✓" : "✗") << "\n\n";
    
    // Test shop_pack1 - first call
    BalatroExactRNG rng2("AAAAAAAA");  // Fresh RNG for pack tests
    double pack1_result = rng2.pseudoseed("shop_pack1");
    double pack2_result = rng2.pseudoseed("shop_pack1");  // Second call
    
    std::cout << "pseudoseed('shop_pack1') first call:\n";
    std::cout << "  Expected: " << seed_tests[1].expected << "\n";
    std::cout << "  Got:      " << pack1_result << "\n";
    std::cout << "  Match:    " << (std::abs(pack1_result - seed_tests[1].expected) < 1e-15 ? "✓" : "✗") << "\n\n";
    
    std::cout << "pseudoseed('shop_pack1') second call:\n";
    std::cout << "  Expected: " << seed_tests[2].expected << "\n";
    std::cout << "  Got:      " << pack2_result << "\n";
    std::cout << "  Match:    " << (std::abs(pack2_result - seed_tests[2].expected) < 1e-15 ? "✓" : "✗") << "\n\n";
    
    // Check if hashed value changed (it shouldn't)
    std::cout << "State check:\n";
    std::cout << "  Hashed after calls: " << rng2.get_hashed() << "\n";
    std::cout << "  Should be unchanged: " << pseudohash_exact("AAAAAAAA") << "\n";
    std::cout << "  Unchanged: " << (rng2.get_hashed() == pseudohash_exact("AAAAAAAA") ? "✓" : "✗") << "\n";
    
    return 0;
}