/*
 * Seed Conversion Utilities
 * Handles Balatro's base-36 seed format (0-9, A-Z)
 */

#ifndef SEED_CONVERSION_HPP
#define SEED_CONVERSION_HPP

#include <cstdint>
#include <string>

// Convert Balatro seed string (base-36: 0-9, A-Z) to numeric value
inline uint64_t seed_to_int(const char* seed_str) {
    uint64_t result = 0;
    
    for (int i = 0; i < 8 && seed_str[i]; i++) {
        char c = seed_str[i];
        uint32_t digit;
        
        if (c >= '0' && c <= '9') {
            digit = c - '0';  // 0-9 maps to 0-9
        } else if (c >= 'A' && c <= 'Z') {
            digit = c - 'A' + 10;  // A-Z maps to 10-35
        } else {
            // Invalid character, treat as 0
            digit = 0;
        }
        
        result = result * 36 + digit;  // Base-36
    }
    
    return result;
}

// Convert numeric seed back to base-36 string
inline void int_to_seed(uint64_t seed_num, char* seed_str) {
    static const char BASE36_CHARS[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    
    for (int i = 7; i >= 0; i--) {
        seed_str[i] = BASE36_CHARS[seed_num % 36];
        seed_num /= 36;
    }
    seed_str[8] = '\0';
}

// Get maximum valid seed value (ZZZZZZZZ in base-36)
inline uint64_t max_seed_value() {
    // 36^8 - 1 = 2,821,109,907,455
    return 2821109907455ULL;
}

#endif // SEED_CONVERSION_HPP