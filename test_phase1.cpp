// Phase 1 Acceptance Test: Verify cur/2.0 semantics
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include "ImmolateCPP/src/balatro_correct.hpp"

void print17(double val) {
    std::cout << std::setprecision(17) << std::fixed << val;
}

int main() {
    std::cout << "=== Phase 1 Acceptance Test ===\n\n";
    
    // Test 1: Verify both implementations use cur/2.0
    std::cout << "Test 1: CPU pseudoseed values with cur/2.0 semantics\n";
    std::cout << "Seed: AAAAAAAA\n\n";
    
    // Test with balatro_correct.hpp
    {
        BalatroRNG rng("AAAAAAAA");
        
        std::cout << "balatro_correct.hpp:\n";
        double v1 = rng.pseudoseed("Voucher");
        std::cout << "  pseudoseed('Voucher') #1 = ";
        print17(v1);
        std::cout << "\n";
        
        double v2 = rng.pseudoseed("Voucher");
        std::cout << "  pseudoseed('Voucher') #2 = ";
        print17(v2);
        std::cout << "\n";
        
        // Verify stored state is double the returned value
        double stored = rng.get_state("Voucher");
        std::cout << "  Stored state = ";
        print17(stored);
        std::cout << "\n";
        std::cout << "  Returned * 2 = ";
        print17(v2 * 2.0);
        std::cout << "\n";
        std::cout << "  Match: " << (std::abs(stored - v2 * 2.0) < 1e-15 ? "✓" : "✗") << "\n\n";
    }
    
    // Note: balatro_exact.hpp also updated with same fix
    
    // Test 2: Golden value comparison
    std::cout << "Test 2: Golden value comparison\n";
    
    struct GoldenTest {
        const char* key;
        int call_num;
        double expected;
    } golden[] = {
        {"Voucher", 1, 0.24901819449170001},
        {"Voucher", 2, 0.49661186021705001},
        {"shop_pack1", 1, 0.38681086553959998},
        {"shop_pack1", 2, 0.23420946326650000}
    };
    
    BalatroRNG rng("AAAAAAAA");
    BalatroRNG rng2("AAAAAAAA");
    
    bool all_pass = true;
    for (const auto& test : golden) {
        BalatroRNG* r = (strcmp(test.key, "Voucher") == 0) ? &rng : &rng2;
        double val = r->pseudoseed(test.key);
        
        std::cout << "  " << test.key << " call #" << test.call_num << ":\n";
        std::cout << "    Expected: ";
        print17(test.expected);
        std::cout << "\n";
        std::cout << "    Got:      ";
        print17(val);
        std::cout << "\n";
        
        bool match = std::abs(val - test.expected) < 1e-10;
        std::cout << "    Match: " << (match ? "✓" : "✗") << "\n";
        if (!match) all_pass = false;
    }
    
    std::cout << "\n=== Phase 1 Result: " << (all_pass ? "PASS ✓" : "FAIL ✗") << " ===\n";
    
    return all_pass ? 0 : 1;
}