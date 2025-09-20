#include <iostream>
#include <cmath>
#include <iomanip>

// Analyze the relationship between our values and actual values
int main() {
    std::cout << std::fixed << std::setprecision(17);
    
    struct Comparison {
        const char* name;
        double actual;
        double our_value;
    } data[] = {
        {"Voucher", 0.24901819449170001, 0.46530388624939389},
        {"shop_pack1 #1", 0.38681086553959998, 0.60309655729733147},
        {"shop_pack1 #2", 0.23420946326650000, 0.45049515502425352}
    };
    
    std::cout << "=== Analyzing pseudoseed differences ===\n\n";
    
    for (const auto& d : data) {
        double ratio = d.actual / d.our_value;
        double diff = d.our_value - d.actual;
        
        std::cout << d.name << ":\n";
        std::cout << "  Actual:     " << d.actual << "\n";
        std::cout << "  Our value:  " << d.our_value << "\n";
        std::cout << "  Ratio:      " << ratio << "\n";
        std::cout << "  Difference: " << diff << "\n";
        std::cout << "\n";
    }
    
    // Check if there's a consistent transformation
    std::cout << "=== Checking for patterns ===\n\n";
    
    // The hashed value for AAAAAAAA
    double hashed = 0.43257138351543745;
    
    for (const auto& d : data) {
        // Check if actual = our_value - constant
        double const_diff = d.our_value - d.actual;
        std::cout << d.name << " difference: " << const_diff << "\n";
        
        // Check if it's related to hashed somehow
        double hashed_diff = const_diff - hashed;
        std::cout << "  Diff - hashed: " << hashed_diff << "\n";
        
        // Check if actual = (our_value - hashed) * 2
        double transformed = (d.our_value - hashed) * 2;
        std::cout << "  (our - hashed) * 2: " << transformed << "\n";
        
        // Check if actual = our_value * k for some k
        double k = d.actual / d.our_value;
        std::cout << "  Ratio: " << k << "\n";
        
        std::cout << "\n";
    }
    
    // Try reverse engineering the formula
    std::cout << "=== Reverse engineering ===\n\n";
    
    // If our formula is: return (cur + hashed) / 2.0
    // And we're getting different values, maybe the actual formula is different
    
    for (const auto& d : data) {
        // If actual = (cur + hashed) / 2.0, what would cur be?
        double cur_from_actual = d.actual * 2.0 - hashed;
        std::cout << d.name << ":\n";
        std::cout << "  If result = (cur + hashed) / 2\n";
        std::cout << "  Then cur would be: " << cur_from_actual << "\n";
        
        // If our value = (cur_old + hashed) / 2.0, what was cur_old?
        double cur_from_our = d.our_value * 2.0 - hashed;
        std::cout << "  Our cur was: " << cur_from_our << "\n";
        std::cout << "  Difference in cur: " << (cur_from_our - cur_from_actual) << "\n";
        std::cout << "\n";
    }
    
    return 0;
}