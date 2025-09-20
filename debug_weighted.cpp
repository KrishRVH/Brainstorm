#include <iostream>
#include <iomanip>
#include <cstdint>

int main() {
    // Weights: [10, 20, 30, 40] -> prefix: [10, 30, 60, 100]
    uint64_t prefix[] = {10, 30, 60, 100};
    
    // Test r=0.3 which should give index 2
    double r = 0.3;
    uint64_t total = prefix[3]; // 100
    long double t = static_cast<long double>(r) * static_cast<long double>(total);
    
    std::cout << "r = " << r << "\n";
    std::cout << "total = " << total << "\n";
    std::cout << "t = r * total = " << std::setprecision(17) << t << "\n";
    std::cout << "\nComparing t < prefix[i]:\n";
    
    for (int i = 0; i < 4; i++) {
        bool less = (t < static_cast<long double>(prefix[i]));
        std::cout << "  t(" << t << ") < prefix[" << i << "](" << prefix[i] << ") = " 
                  << (less ? "true" : "false") << "\n";
    }
    
    // The issue: we want STRICT inequality for the ranges
    // Range 0: [0, 10)
    // Range 1: [10, 30) 
    // Range 2: [30, 60)
    // Range 3: [60, 100)
    
    // So t=30 should be in range 2, not range 1
    // Our current code uses t < prefix[i], which gives:
    // t=30 < 30 = false, so we don't select index 1
    // t=30 < 60 = true, so we select index 2 ✓
    
    std::cout << "\nActually, the algorithm is correct for t=30\n";
    std::cout << "Let me check t=29.999...\n";
    
    t = 29.999999999999996; // Just under 30
    std::cout << "\nt = " << t << "\n";
    for (int i = 0; i < 4; i++) {
        bool less = (t < static_cast<long double>(prefix[i]));
        std::cout << "  t(" << t << ") < prefix[" << i << "](" << prefix[i] << ") = " 
                  << (less ? "true" : "false") << "\n";
    }
    
    return 0;
}