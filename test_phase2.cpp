// Phase 2 Acceptance Test: Verify chooser functions
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "ImmolateCPP/src/balatro_correct.hpp"
#include "ImmolateCPP/src/choosers.hpp"

void print17(double val) {
    std::cout << std::setprecision(17) << std::fixed << val;
}

int main() {
    std::cout << "=== Phase 2 Acceptance Test ===\n\n";
    
    // Test 1: Uniform chooser
    std::cout << "Test 1: Uniform chooser\n";
    
    struct UniformTest {
        double r;
        uint32_t n;
        uint32_t expected;
    } uniform_tests[] = {
        {0.0, 32, 0},
        {0.5, 32, 16},
        {0.999999, 32, 31},
        {0.24901819449170001, 32, 7},  // Voucher value from golden
        {0.38681086553959998, 15, 5},   // Pack value from golden
    };
    
    bool uniform_pass = true;
    for (const auto& test : uniform_tests) {
        uint32_t result = choose_uniform(test.r, test.n);
        bool match = (result == test.expected);
        
        std::cout << "  r=";
        print17(test.r);
        std::cout << " n=" << test.n << " -> " << result;
        std::cout << " (expected " << test.expected << ") ";
        std::cout << (match ? "✓" : "✗") << "\n";
        
        if (!match) uniform_pass = false;
    }
    
    std::cout << "\n";
    
    // Test 2: Weighted chooser
    std::cout << "Test 2: Weighted chooser\n";
    
    // Example weights: [10, 20, 30, 40] -> prefix: [10, 30, 60, 100]
    uint64_t prefix[] = {10, 30, 60, 100};
    
    struct WeightedTest {
        double r;
        uint32_t expected;
    } weighted_tests[] = {
        {0.0, 0},      // First bucket
        {0.09, 0},     // Still first (9 < 10)
        {0.10, 1},     // Second bucket (10 <= t < 30)
        {0.29, 1},     // Still second (29 < 30)
        {0.30, 1},     // Still second! (30.0 == 30.0, not <)
        {0.301, 2},    // Third bucket (30.1 > 30)
        {0.59, 2},     // Still third (59 < 60)
        {0.60, 2},     // Still third! (60.0 == 60.0, not <)
        {0.601, 3},    // Fourth bucket (60.1 > 60)
        {0.99, 3},     // Still fourth
    };
    
    bool weighted_pass = true;
    for (const auto& test : weighted_tests) {
        uint32_t result = choose_weighted(test.r, prefix, 4);
        bool match = (result == test.expected);
        
        std::cout << "  r=";
        print17(test.r);
        std::cout << " -> " << result;
        std::cout << " (expected " << test.expected << ") ";
        std::cout << (match ? "✓" : "✗") << "\n";
        
        if (!match) weighted_pass = false;
    }
    
    std::cout << "\n";
    
    // Test 3: Actual shop generation with choosers
    std::cout << "Test 3: Shop generation with choosers\n";
    
    BalatroRNG rng("AAAAAAAA");
    
    // Generate voucher
    double voucher_val = rng.pseudoseed("Voucher");
    uint32_t voucher_idx = choose_uniform(voucher_val, 32);
    
    std::cout << "  Voucher value: ";
    print17(voucher_val);
    std::cout << " -> index " << voucher_idx << "\n";
    
    // Generate packs
    double pack1_val = rng.pseudoseed("shop_pack1");
    double pack2_val = rng.pseudoseed("shop_pack1");
    uint32_t pack1_idx = choose_uniform(pack1_val, 15);
    uint32_t pack2_idx = choose_uniform(pack2_val, 15);
    
    std::cout << "  Pack1 value: ";
    print17(pack1_val);
    std::cout << " -> index " << pack1_idx << "\n";
    
    std::cout << "  Pack2 value: ";
    print17(pack2_val);
    std::cout << " -> index " << pack2_idx << "\n";
    
    // Generate tags
    double tag_small_val = rng.pseudoseed("Tag_small");
    double tag_big_val = rng.pseudoseed("Tag_big");
    uint32_t tag_small_idx = choose_uniform(tag_small_val, 24);
    uint32_t tag_big_idx = choose_uniform(tag_big_val, 24);
    
    std::cout << "  Small tag value: ";
    print17(tag_small_val);
    std::cout << " -> index " << tag_small_idx << "\n";
    
    std::cout << "  Big tag value: ";
    print17(tag_big_val);
    std::cout << " -> index " << tag_big_idx << "\n";
    
    std::cout << "\n=== Phase 2 Result: " 
              << ((uniform_pass && weighted_pass) ? "PASS ✓" : "FAIL ✗") 
              << " ===\n";
    
    return (uniform_pass && weighted_pass) ? 0 : 1;
}