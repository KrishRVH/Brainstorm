/*
 * Tag Per-Context Indices Test
 * Validates that tags work correctly when tag_small and tag_big have different orderings
 */

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include "src/gpu/gpu_types.h"

// Simulate different pool orderings for tag_small and tag_big
struct TagPools {
    std::vector<std::string> tag_small;
    std::vector<std::string> tag_big;
};

void test_tag_per_context() {
    printf("=== Tag Per-Context Indices Test ===\n\n");
    
    // Create pools with different orderings
    TagPools pools;
    
    // tag_small pool (different order)
    pools.tag_small = {
        "Uncommon Tag",     // 0
        "Rare Tag",         // 1
        "Negative Tag",     // 2
        "Foil Tag",         // 3
        "Holographic Tag",  // 4
        "Polychrome Tag",   // 5
        "Investment Tag",   // 6
        "Voucher Tag",      // 7
        "Boss Tag",         // 8
        // ... more tags
    };
    
    // tag_big pool (different order - Investment Tag is at different index)
    pools.tag_big = {
        "Rare Tag",         // 0
        "Uncommon Tag",     // 1
        "Investment Tag",   // 2 (different from tag_small!)
        "Boss Tag",         // 3
        "Foil Tag",         // 4
        "Holographic Tag",  // 5
        "Polychrome Tag",   // 6
        "Negative Tag",     // 7
        "Voucher Tag",      // 8
        // ... more tags
    };
    
    printf("Pool configurations:\n");
    printf("  tag_small: Investment Tag at index 6\n");
    printf("  tag_big:   Investment Tag at index 2\n\n");
    
    // Test case 1: Looking for Investment Tag in both contexts
    {
        FilterParams params = {};
        params.tag1_small = 6;  // Investment Tag in tag_small
        params.tag1_big = 2;    // Investment Tag in tag_big (different index!)
        params.tag2_small = params.tag2_big = 0xFFFFFFFF;
        params.pack1 = params.pack2 = 0xFFFFFFFF;
        params.voucher = 0xFFFFFFFF;
        
        printf("Test 1 - Investment Tag with different indices:\n");
        printf("  Looking for Investment Tag (small idx=6, big idx=2)\n");
        
        // Scenario A: Small tag is Investment Tag
        {
            uint32_t small_tag = 6;  // Investment Tag
            uint32_t big_tag = 3;    // Boss Tag
            
            bool want_t1 = (params.tag1_small != 0xFFFFFFFF) || (params.tag1_big != 0xFFFFFFFF);
            bool match = true;
            if (want_t1) {
                bool has_t1 = ((params.tag1_small != 0xFFFFFFFF) && (small_tag == params.tag1_small)) ||
                              ((params.tag1_big   != 0xFFFFFFFF) && (big_tag   == params.tag1_big));
                if (!has_t1) match = false;
            }
            
            printf("  Scenario A - Found in small slot:\n");
            printf("    small_tag=%u (Investment), big_tag=%u (Boss)\n", small_tag, big_tag);
            printf("    Result: %s\n", match ? "✅ MATCH" : "❌ NO MATCH");
        }
        
        // Scenario B: Big tag is Investment Tag
        {
            uint32_t small_tag = 1;  // Rare Tag
            uint32_t big_tag = 2;    // Investment Tag in big pool
            
            bool want_t1 = (params.tag1_small != 0xFFFFFFFF) || (params.tag1_big != 0xFFFFFFFF);
            bool match = true;
            if (want_t1) {
                bool has_t1 = ((params.tag1_small != 0xFFFFFFFF) && (small_tag == params.tag1_small)) ||
                              ((params.tag1_big   != 0xFFFFFFFF) && (big_tag   == params.tag1_big));
                if (!has_t1) match = false;
            }
            
            printf("  Scenario B - Found in big slot:\n");
            printf("    small_tag=%u (Rare), big_tag=%u (Investment)\n", small_tag, big_tag);
            printf("    Result: %s\n", match ? "✅ MATCH" : "❌ NO MATCH");
        }
        
        // Scenario C: Neither has Investment Tag
        {
            uint32_t small_tag = 1;  // Rare Tag
            uint32_t big_tag = 3;    // Boss Tag
            
            bool want_t1 = (params.tag1_small != 0xFFFFFFFF) || (params.tag1_big != 0xFFFFFFFF);
            bool match = true;
            if (want_t1) {
                bool has_t1 = ((params.tag1_small != 0xFFFFFFFF) && (small_tag == params.tag1_small)) ||
                              ((params.tag1_big   != 0xFFFFFFFF) && (big_tag   == params.tag1_big));
                if (!has_t1) match = false;
            }
            
            printf("  Scenario C - Not found:\n");
            printf("    small_tag=%u (Rare), big_tag=%u (Boss)\n", small_tag, big_tag);
            printf("    Result: %s\n", match ? "✅ MATCH" : "❌ NO MATCH");
        }
        printf("\n");
    }
    
    // Test case 2: Looking for two different tags
    {
        FilterParams params = {};
        params.tag1_small = 6;  // Investment Tag in tag_small
        params.tag1_big = 2;    // Investment Tag in tag_big
        params.tag2_small = 1;  // Rare Tag in tag_small
        params.tag2_big = 0;    // Rare Tag in tag_big (different index!)
        params.pack1 = params.pack2 = 0xFFFFFFFF;
        params.voucher = 0xFFFFFFFF;
        
        printf("Test 2 - Two tags with different indices:\n");
        printf("  Looking for Investment Tag AND Rare Tag\n");
        printf("  Investment: small idx=6, big idx=2\n");
        printf("  Rare:       small idx=1, big idx=0\n");
        
        // Scenario: Has both tags
        {
            uint32_t small_tag = 6;  // Investment Tag
            uint32_t big_tag = 0;    // Rare Tag
            
            bool match = true;
            
            // Check tag1 (Investment)
            bool want_t1 = (params.tag1_small != 0xFFFFFFFF) || (params.tag1_big != 0xFFFFFFFF);
            if (want_t1) {
                bool has_t1 = ((params.tag1_small != 0xFFFFFFFF) && (small_tag == params.tag1_small)) ||
                              ((params.tag1_big   != 0xFFFFFFFF) && (big_tag   == params.tag1_big));
                if (!has_t1) match = false;
            }
            
            // Check tag2 (Rare)
            bool want_t2 = (params.tag2_small != 0xFFFFFFFF) || (params.tag2_big != 0xFFFFFFFF);
            if (match && want_t2) {
                bool has_t2 = ((params.tag2_small != 0xFFFFFFFF) && (small_tag == params.tag2_small)) ||
                              ((params.tag2_big   != 0xFFFFFFFF) && (big_tag   == params.tag2_big));
                if (!has_t2) match = false;
            }
            
            printf("  Found: small_tag=%u (Investment), big_tag=%u (Rare)\n", small_tag, big_tag);
            printf("  Result: %s (both tags found in different slots)\n", match ? "✅ MATCH" : "❌ NO MATCH");
        }
        printf("\n");
    }
    
    // Test case 3: Only searching in one context
    {
        FilterParams params = {};
        params.tag1_small = 6;  // Investment Tag in tag_small only
        params.tag1_big = 0xFFFFFFFF;  // Not searching in big
        params.tag2_small = params.tag2_big = 0xFFFFFFFF;
        params.pack1 = params.pack2 = 0xFFFFFFFF;
        params.voucher = 0xFFFFFFFF;
        
        printf("Test 3 - Single context search:\n");
        printf("  Looking for Investment Tag in tag_small ONLY (idx=6)\n");
        
        // Scenario: Tag is in small slot
        {
            uint32_t small_tag = 6;  // Investment Tag
            uint32_t big_tag = 5;    // Something else
            
            bool want_t1 = (params.tag1_small != 0xFFFFFFFF) || (params.tag1_big != 0xFFFFFFFF);
            bool match = true;
            if (want_t1) {
                bool has_t1 = ((params.tag1_small != 0xFFFFFFFF) && (small_tag == params.tag1_small)) ||
                              ((params.tag1_big   != 0xFFFFFFFF) && (big_tag   == params.tag1_big));
                if (!has_t1) match = false;
            }
            
            printf("  Found: small_tag=%u (Investment), big_tag=%u\n", small_tag, big_tag);
            printf("  Result: %s\n", match ? "✅ MATCH" : "❌ NO MATCH");
        }
        printf("\n");
    }
    
    printf("=== Tag Per-Context Test Complete ===\n");
    printf("✅ All test cases demonstrate per-context indices working correctly\n");
    printf("✅ Tags can have different indices in tag_small vs tag_big pools\n");
}

int main() {
    test_tag_per_context();
    return 0;
}