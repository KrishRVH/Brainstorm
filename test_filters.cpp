#include <iostream>
#include <cstring>
#include "ImmolateCPP/src/gpu/gpu_types.h"
#include "ImmolateCPP/src/items.hpp"

// Simulate the DLL parameter conversion
void test_parameter_conversion() {
    std::cout << "=== Testing Parameter Conversion ===" << std::endl;
    
    // Test cases
    const char* voucher_str = "Clearance Sale";
    const char* pack_str = "Spectral Pack";
    const char* tag1_str = "Double Tag";
    const char* tag2_str = "Investment Tag";
    
    // Convert strings to Items
    Item voucher_item = stringToItem(voucher_str);
    Item pack_item = stringToItem(pack_str);
    Item tag1_item = stringToItem(tag1_str);
    Item tag2_item = stringToItem(tag2_str);
    
    std::cout << "\nString to Item conversion:" << std::endl;
    std::cout << "  '" << voucher_str << "' -> Item::" << static_cast<int>(voucher_item) << std::endl;
    std::cout << "  '" << pack_str << "' -> Item::" << static_cast<int>(pack_item) << std::endl;
    std::cout << "  '" << tag1_str << "' -> Item::" << static_cast<int>(tag1_item) << std::endl;
    std::cout << "  '" << tag2_str << "' -> Item::" << static_cast<int>(tag2_item) << std::endl;
    
    // Convert to indices (what GPU expects)
    // We need to find the actual enum values first
    std::cout << "\nChecking enum positions:" << std::endl;
    
    // Count voucher position
    int voucher_pos = 0;
    for (int i = static_cast<int>(Item::V_BEGIN) + 1; i <= static_cast<int>(voucher_item); i++) {
        if (i == static_cast<int>(voucher_item)) break;
        voucher_pos++;
    }
    
    // For packs and tags, we need to know their begin positions
    // Let's find them
    std::cout << "  V_BEGIN = " << static_cast<int>(Item::V_BEGIN) << std::endl;
    std::cout << "  Overstock = " << static_cast<int>(Item::Overstock) << std::endl;
    std::cout << "  Clearance_Sale = " << static_cast<int>(Item::Clearance_Sale) << std::endl;
    
    // Calculate indices
    uint32_t voucher_idx = static_cast<uint32_t>(voucher_item) - static_cast<uint32_t>(Item::Overstock);
    uint32_t pack_idx = static_cast<uint32_t>(pack_item) - 329;  // Approximate P_BEGIN
    uint32_t tag1_idx = static_cast<uint32_t>(tag1_item) - 313;  // Approximate TAG_BEGIN
    uint32_t tag2_idx = static_cast<uint32_t>(tag2_item) - 313;
    
    std::cout << "\nCalculated indices for GPU:" << std::endl;
    std::cout << "  Voucher: " << voucher_idx << " (expected 0-31)" << std::endl;
    std::cout << "  Pack: " << pack_idx << " (expected 0-19)" << std::endl;
    std::cout << "  Tag1: " << tag1_idx << " (expected 0-29)" << std::endl;
    std::cout << "  Tag2: " << tag2_idx << " (expected 0-29)" << std::endl;
    
    // Validate ranges
    if (voucher_idx > 31) std::cout << "  WARNING: Voucher index out of range!" << std::endl;
    if (pack_idx > 19) std::cout << "  WARNING: Pack index out of range!" << std::endl;
    if (tag1_idx > 29) std::cout << "  WARNING: Tag1 index out of range!" << std::endl;
    if (tag2_idx > 29) std::cout << "  WARNING: Tag2 index out of range!" << std::endl;
}

// Test common filters
void test_common_filters() {
    std::cout << "\n=== Testing Common Filters ===" << std::endl;
    
    struct TestCase {
        const char* name;
        const char* voucher;
        const char* pack;
        const char* tag1;
        const char* tag2;
    };
    
    TestCase cases[] = {
        {"Overstock only", "Overstock", "", "", ""},
        {"Buffoon Pack only", "", "Buffoon Pack", "", ""},
        {"Charm Tag only", "", "", "Charm Tag", ""},
        {"Double + Investment Tags", "", "", "Double Tag", "Investment Tag"},
        {"All filters", "Clearance Sale", "Arcana Pack", "Economy Tag", "Coupon Tag"}
    };
    
    for (const auto& tc : cases) {
        std::cout << "\nTest: " << tc.name << std::endl;
        
        FilterParams params;
        
        // Convert and calculate indices
        if (tc.voucher && strlen(tc.voucher) > 0) {
            Item item = stringToItem(tc.voucher);
            params.voucher = (item != Item::RETRY) ? 
                (static_cast<uint32_t>(item) - static_cast<uint32_t>(Item::Overstock)) : 0xFFFFFFFF;
            std::cout << "  Voucher '" << tc.voucher << "' -> index " << params.voucher << std::endl;
        } else {
            params.voucher = 0xFFFFFFFF;
        }
        
        if (tc.pack && strlen(tc.pack) > 0) {
            Item item = stringToItem(tc.pack);
            // Find the actual pack begin position
            params.pack = (item != Item::RETRY) ? 
                (static_cast<uint32_t>(item) - static_cast<uint32_t>(Item::Arcana_Pack)) : 0xFFFFFFFF;
            std::cout << "  Pack '" << tc.pack << "' -> index " << params.pack << std::endl;
        } else {
            params.pack = 0xFFFFFFFF;
        }
        
        if (tc.tag1 && strlen(tc.tag1) > 0) {
            Item item = stringToItem(tc.tag1);
            // Find actual tag begin position
            params.tag1 = (item != Item::RETRY) ? 
                (static_cast<uint32_t>(item) - static_cast<uint32_t>(Item::Charm_Tag)) : 0xFFFFFFFF;
            std::cout << "  Tag1 '" << tc.tag1 << "' -> index " << params.tag1 << std::endl;
        } else {
            params.tag1 = 0xFFFFFFFF;
        }
        
        if (tc.tag2 && strlen(tc.tag2) > 0) {
            Item item = stringToItem(tc.tag2);
            params.tag2 = (item != Item::RETRY) ? 
                (static_cast<uint32_t>(item) - static_cast<uint32_t>(Item::Charm_Tag)) : 0xFFFFFFFF;
            std::cout << "  Tag2 '" << tc.tag2 << "' -> index " << params.tag2 << std::endl;
        } else {
            params.tag2 = 0xFFFFFFFF;
        }
    }
}

int main() {
    std::cout << "Brainstorm Filter Testing\n";
    std::cout << "==========================\n\n";
    
    test_parameter_conversion();
    test_common_filters();
    
    return 0;
}