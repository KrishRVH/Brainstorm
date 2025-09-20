/*
 * Full simulation test - validates the entire parameter flow
 */

#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include "ImmolateCPP/src/items.hpp"
#include "ImmolateCPP/src/gpu/gpu_types.h"

// Simulate the exact conversion logic from brainstorm_driver.cpp
FilterParams simulate_dll_conversion(
    const char* voucher,
    const char* pack,
    const char* tag1,
    const char* tag2,
    double souls,
    bool observatory,
    bool perkeo
) {
    FilterParams params;
    
    // Convert strings to Items (exactly as DLL does)
    Item tag1_item = tag1 && strlen(tag1) > 0 ? stringToItem(tag1) : Item::RETRY;
    Item tag2_item = tag2 && strlen(tag2) > 0 ? stringToItem(tag2) : Item::RETRY;
    Item voucher_item = voucher && strlen(voucher) > 0 ? stringToItem(voucher) : Item::RETRY;
    Item pack_item = pack && strlen(pack) > 0 ? stringToItem(pack) : Item::RETRY;
    
    // Convert to indices (exactly as DLL does)
    params.tag1 = (tag1_item != Item::RETRY) ? (static_cast<uint32_t>(tag1_item) - 310) : 0xFFFFFFFF;
    params.tag2 = (tag2_item != Item::RETRY) ? (static_cast<uint32_t>(tag2_item) - 310) : 0xFFFFFFFF;
    params.voucher = (voucher_item != Item::RETRY) ? (static_cast<uint32_t>(voucher_item) - 162) : 0xFFFFFFFF;
    params.pack = (pack_item != Item::RETRY) ? (static_cast<uint32_t>(pack_item) - 293) : 0xFFFFFFFF;
    params.require_souls = (souls > 0) ? 1 : 0;
    params.require_observatory = observatory ? 1 : 0;
    params.require_perkeo = perkeo ? 1 : 0;
    
    return params;
}

// Test case structure
struct TestCase {
    std::string name;
    std::string voucher;
    std::string pack;
    std::string tag1;
    std::string tag2;
    double souls;
    bool observatory;
    bool perkeo;
};

bool validate_params(const FilterParams& params, const TestCase& tc) {
    bool valid = true;
    
    // Check voucher
    if (params.voucher != 0xFFFFFFFF) {
        if (params.voucher > 31) {
            std::cout << "    ❌ Voucher index " << params.voucher << " out of range (max=31)" << std::endl;
            valid = false;
        } else {
            std::cout << "    ✓ Voucher index: " << params.voucher << std::endl;
        }
    } else if (!tc.voucher.empty()) {
        std::cout << "    ⚠ Voucher '" << tc.voucher << "' not recognized" << std::endl;
    }
    
    // Check pack
    if (params.pack != 0xFFFFFFFF) {
        if (params.pack > 14) {
            std::cout << "    ❌ Pack index " << params.pack << " out of range (max=14)" << std::endl;
            valid = false;
        } else {
            std::cout << "    ✓ Pack index: " << params.pack << std::endl;
        }
    } else if (!tc.pack.empty()) {
        std::cout << "    ⚠ Pack '" << tc.pack << "' not recognized" << std::endl;
    }
    
    // Check tag1
    if (params.tag1 != 0xFFFFFFFF) {
        if (params.tag1 > 23) {
            std::cout << "    ❌ Tag1 index " << params.tag1 << " out of range (max=23)" << std::endl;
            valid = false;
        } else {
            std::cout << "    ✓ Tag1 index: " << params.tag1 << std::endl;
        }
    } else if (!tc.tag1.empty()) {
        std::cout << "    ⚠ Tag1 '" << tc.tag1 << "' not recognized" << std::endl;
    }
    
    // Check tag2
    if (params.tag2 != 0xFFFFFFFF) {
        if (params.tag2 > 23) {
            std::cout << "    ❌ Tag2 index " << params.tag2 << " out of range (max=23)" << std::endl;
            valid = false;
        } else {
            std::cout << "    ✓ Tag2 index: " << params.tag2 << std::endl;
        }
    } else if (!tc.tag2.empty()) {
        std::cout << "    ⚠ Tag2 '" << tc.tag2 << "' not recognized" << std::endl;
    }
    
    return valid;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Brainstorm Full Simulation Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Real-world test cases from game
    std::vector<TestCase> tests = {
        // Basic filters
        {"No filters", "", "", "", "", 0, false, false},
        {"Overstock only", "Overstock", "", "", "", 0, false, false},
        {"Clearance Sale only", "Clearance Sale", "", "", "", 0, false, false},
        {"Arcana Pack only", "", "Arcana Pack", "", "", 0, false, false},
        {"Buffoon Pack only", "", "Buffoon Pack", "", "", 0, false, false},
        {"Spectral Pack only", "", "Spectral Pack", "", "", 0, false, false},
        
        // Tags
        {"Charm Tag only", "", "", "Charm Tag", "", 0, false, false},
        {"Double Tag only", "", "", "Double Tag", "", 0, false, false},
        {"Investment Tag only", "", "", "Investment Tag", "", 0, false, false},
        {"Double + Investment", "", "", "Double Tag", "Investment Tag", 0, false, false},
        
        // Complex combinations (from screenshots)
        {"Clearance + Spectral + Double + Investment", 
         "Clearance Sale", "Spectral Pack", "Double Tag", "Investment Tag", 0, false, false},
        
        // Special requirements
        {"Souls requirement", "", "", "", "", 1.0, false, false},
        {"Observatory requirement", "", "", "", "", 0, true, false},
        {"Perkeo requirement", "", "", "", "", 0, false, true},
        
        // Edge cases
        {"All vouchers (last)", "Curator", "", "", "", 0, false, false},
        {"Invalid items", "Invalid Voucher", "Invalid Pack", "Invalid Tag", "", 0, false, false},
    };
    
    int passed = 0;
    int failed = 0;
    
    for (const auto& tc : tests) {
        std::cout << "Test: " << tc.name << std::endl;
        
        // Simulate DLL conversion
        FilterParams params = simulate_dll_conversion(
            tc.voucher.c_str(),
            tc.pack.c_str(),
            tc.tag1.c_str(),
            tc.tag2.c_str(),
            tc.souls,
            tc.observatory,
            tc.perkeo
        );
        
        // Validate
        if (validate_params(params, tc)) {
            std::cout << "  ✅ PASS" << std::endl;
            passed++;
        } else {
            std::cout << "  ❌ FAIL" << std::endl;
            failed++;
        }
        std::cout << std::endl;
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "  Results: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return failed > 0 ? 1 : 0;
}