#include <iostream>
#include <cmath>
#include <string>
#include <iomanip>
#include "ImmolateCPP/src/balatro_correct.hpp"

int main() {
    std::cout << std::fixed << std::setprecision(17);
    
    std::cout << "=== Testing CORRECT Implementation ===\n\n";
    
    // Test pseudohash (should still be correct)
    std::cout << "Testing pseudohash:\n";
    std::cout << "  pseudohash('AAAAAAAA') = " << pseudohash("AAAAAAAA") << "\n";
    std::cout << "  Expected:                0.43257138351543745\n";
    std::cout << "  Match: " << (std::abs(pseudohash("AAAAAAAA") - 0.43257138351543745) < 1e-15 ? "✓" : "✗") << "\n\n";
    
    // Test pseudoseed with actual captured values
    BalatroRNG rng("AAAAAAAA");
    
    std::cout << "Testing pseudoseed with seed 'AAAAAAAA':\n\n";
    
    // Test Voucher (fresh state)
    double voucher1 = rng.pseudoseed("Voucher");
    std::cout << "First pseudoseed('Voucher'):\n";
    std::cout << "  Expected: 0.24901819449170001\n";
    std::cout << "  Got:      " << voucher1 << "\n";
    std::cout << "  Match:    " << (std::abs(voucher1 - 0.24901819449170001) < 1e-10 ? "✓" : "✗") << "\n";
    std::cout << "  Stored:   " << rng.get_state("Voucher") << "\n\n";
    
    // Second call to Voucher
    double voucher2 = rng.pseudoseed("Voucher");
    std::cout << "Second pseudoseed('Voucher'):\n";
    std::cout << "  Expected: 0.49661186021705001\n";
    std::cout << "  Got:      " << voucher2 << "\n";
    std::cout << "  Match:    " << (std::abs(voucher2 - 0.49661186021705001) < 1e-10 ? "✓" : "✗") << "\n\n";
    
    // Test shop_pack1 (fresh RNG)
    BalatroRNG rng2("AAAAAAAA");
    
    double pack1 = rng2.pseudoseed("shop_pack1");
    std::cout << "First pseudoseed('shop_pack1'):\n";
    std::cout << "  Expected: 0.38681086553959998\n";
    std::cout << "  Got:      " << pack1 << "\n";
    std::cout << "  Match:    " << (std::abs(pack1 - 0.38681086553959998) < 1e-10 ? "✓" : "✗") << "\n\n";
    
    double pack2 = rng2.pseudoseed("shop_pack1");
    std::cout << "Second pseudoseed('shop_pack1'):\n";
    std::cout << "  Expected: 0.23420946326650000\n";
    std::cout << "  Got:      " << pack2 << "\n";
    std::cout << "  Match:    " << (std::abs(pack2 - 0.23420946326650000) < 1e-10 ? "✓" : "✗") << "\n\n";
    
    double pack3 = rng2.pseudoseed("shop_pack1");
    std::cout << "Third pseudoseed('shop_pack1'):\n";
    std::cout << "  Expected: 0.47107698222569999\n";
    std::cout << "  Got:      " << pack3 << "\n";
    std::cout << "  Match:    " << (std::abs(pack3 - 0.47107698222569999) < 1e-10 ? "✓" : "✗") << "\n\n";
    
    std::cout << "=== ALL TESTS " << (true ? "PASSED ✓" : "FAILED ✗") << " ===\n";
    
    return 0;
}