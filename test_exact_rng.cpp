/*
 * Test program for exact Balatro RNG matching
 * Parses JSON traces from the game and verifies our implementation
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdio>
#include "ImmolateCPP/src/balatro_exact.hpp"

// Simple JSON parsing (for test traces)
class SimpleJSONParser {
public:
    static double parse_double(const std::string& line, const std::string& key) {
        size_t pos = line.find("\"" + key + "\":");
        if (pos == std::string::npos) return 0.0;
        
        pos = line.find(":", pos) + 1;
        while (pos < line.size() && (line[pos] == ' ' || line[pos] == '"')) pos++;
        
        if (line[pos] == 'n' && line.substr(pos, 4) == "null") {
            return -999.0; // Special marker for null
        }
        
        size_t end = pos;
        while (end < line.size() && line[end] != ',' && line[end] != '}' && line[end] != '"') end++;
        
        std::string val = line.substr(pos, end - pos);
        return std::stod(val);
    }
    
    static std::string parse_string(const std::string& line, const std::string& key) {
        size_t pos = line.find("\"" + key + "\":\"");
        if (pos == std::string::npos) return "";
        
        pos = line.find("\":\"", pos) + 3;
        size_t end = line.find("\"", pos);
        
        return line.substr(pos, end - pos);
    }
    
    static bool parse_bool(const std::string& line, const std::string& key) {
        size_t pos = line.find("\"" + key + "\":");
        if (pos == std::string::npos) return false;
        
        return line.find("true", pos) != std::string::npos;
    }
};

// Test a single seed against its trace
bool test_seed_trace(const std::string& trace_file, const std::string& test_seed) {
    std::ifstream file(trace_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open trace file: " << trace_file << std::endl;
        return false;
    }
    
    std::cout << "\n=== Testing seed: " << test_seed << " ===" << std::endl;
    
    // Create our RNG instance
    BalatroExactRNG rng(test_seed);
    
    std::string line;
    bool in_test_seed = false;
    bool all_match = true;
    int test_count = 0;
    int match_count = 0;
    
    while (std::getline(file, line)) {
        // Look for test seed start
        if (line.find("\"type\":\"test_seed_start\"") != std::string::npos) {
            std::string seed = SimpleJSONParser::parse_string(line, "seed");
            in_test_seed = (seed == test_seed);
            if (in_test_seed) {
                std::cout << "Found trace for seed: " << seed << std::endl;
            }
        }
        
        // Process events for this seed
        if (in_test_seed) {
            // Check for test seed end
            if (line.find("\"type\":\"test_seed_end\"") != std::string::npos) {
                break;
            }
            
            // Process pseudohash events
            if (line.find("\"type\":\"pseudohash\"") != std::string::npos) {
                std::string input = SimpleJSONParser::parse_string(line, "input");
                double expected = SimpleJSONParser::parse_double(line, "output");
                double actual = pseudohash_exact(input);
                
                test_count++;
                bool matches = (format_double(expected) == format_double(actual));
                if (matches) match_count++;
                else all_match = false;
                
                printf("  pseudohash(\"%s\"): expected=%.17g, actual=%.17g, %s\n",
                       input.c_str(), expected, actual, matches ? "MATCH" : "MISMATCH");
            }
            
            // Process pseudoseed events
            if (line.find("\"type\":\"pseudoseed\"") != std::string::npos) {
                std::string key = SimpleJSONParser::parse_string(line, "key");
                double expected_ret = SimpleJSONParser::parse_double(line, "ret");
                double expected_after = SimpleJSONParser::parse_double(line, "after");
                double expected_hashed = SimpleJSONParser::parse_double(line, "hashed");
                
                // Call our implementation
                double actual_ret = rng.pseudoseed(key);
                double actual_state = rng.get_state(key);
                double actual_hashed = rng.get_hashed();
                
                test_count++;
                bool ret_matches = (format_double(expected_ret) == format_double(actual_ret));
                bool state_matches = (format_double(expected_after) == format_double(actual_state));
                bool hashed_matches = (format_double(expected_hashed) == format_double(actual_hashed));
                
                if (ret_matches && state_matches && hashed_matches) {
                    match_count++;
                } else {
                    all_match = false;
                }
                
                printf("  pseudoseed(\"%s\"):\n", key.c_str());
                printf("    ret: expected=%.17g, actual=%.17g, %s\n",
                       expected_ret, actual_ret, ret_matches ? "MATCH" : "MISMATCH");
                printf("    state: expected=%.17g, actual=%.17g, %s\n",
                       expected_after, actual_state, state_matches ? "MATCH" : "MISMATCH");
                printf("    hashed: expected=%.17g, actual=%.17g, %s\n",
                       expected_hashed, actual_hashed, hashed_matches ? "MATCH" : "MISMATCH");
            }
            
            // Process selection events
            if (line.find("\"type\":\"choose\"") != std::string::npos) {
                double seed_val = SimpleJSONParser::parse_double(line, "seed");
                uint32_t expected_index = static_cast<uint32_t>(SimpleJSONParser::parse_double(line, "index"));
                std::string selected = SimpleJSONParser::parse_string(line, "selected");
                std::string pool_key = SimpleJSONParser::parse_string(line, "pool_key");
                uint32_t pool_size = static_cast<uint32_t>(SimpleJSONParser::parse_double(line, "pool_size"));
                
                // For now, test uniform selection
                uint32_t actual_index = BalatroSelection::select_uniform(seed_val, pool_size);
                
                test_count++;
                bool matches = (expected_index == actual_index);
                if (matches) match_count++;
                else all_match = false;
                
                printf("  select(%s, r=%.17g, size=%u): expected=%u, actual=%u, %s\n",
                       pool_key.c_str(), seed_val, pool_size, expected_index, actual_index,
                       matches ? "MATCH" : "MISMATCH");
            }
        }
    }
    
    // Summary
    std::cout << "\nSummary for " << test_seed << ":" << std::endl;
    std::cout << "  Tests: " << test_count << std::endl;
    std::cout << "  Matches: " << match_count << std::endl;
    std::cout << "  Result: " << (all_match ? "ALL MATCH ✓" : "MISMATCHES FOUND ✗") << std::endl;
    
    return all_match;
}

// Test against sample traces
void test_sample_traces() {
    std::cout << "=== Testing Sample Traces ===" << std::endl;
    
    // Test some known values first
    std::cout << "\nBasic pseudohash tests:" << std::endl;
    printf("  pseudohash(\"AAAAAAAA\") = %.17g\n", pseudohash_exact("AAAAAAAA"));
    printf("  pseudohash(\"00000000\") = %.17g\n", pseudohash_exact("00000000"));
    printf("  pseudohash(\"7NTPKW6P\") = %.17g\n", pseudohash_exact("7NTPKW6P"));
    
    // Test pseudoseed sequence
    std::cout << "\nBasic pseudoseed tests for AAAAAAAA:" << std::endl;
    BalatroExactRNG rng("AAAAAAAA");
    printf("  hashed = %.17g\n", rng.get_hashed());
    printf("  pseudoseed(\"Voucher\") = %.17g\n", rng.pseudoseed("Voucher"));
    printf("  pseudoseed(\"shop_pack1\") = %.17g\n", rng.pseudoseed("shop_pack1"));
    printf("  pseudoseed(\"shop_pack1\") again = %.17g\n", rng.pseudoseed("shop_pack1"));
    printf("  pseudoseed(\"Tag_small\") = %.17g\n", rng.pseudoseed("Tag_small"));
    printf("  pseudoseed(\"Tag_big\") = %.17g\n", rng.pseudoseed("Tag_big"));
}

int main(int argc, char* argv[]) {
    std::cout << "=== Balatro Exact RNG Test ===" << std::endl;
    
    // Run basic tests
    test_sample_traces();
    
    // If trace file provided, test against it
    if (argc > 1) {
        std::string trace_file = argv[1];
        std::cout << "\nTesting against trace file: " << trace_file << std::endl;
        
        // Test each seed
        std::vector<std::string> test_seeds = {"AAAAAAAA", "00000000", "7NTPKW6P", "ZZZZZZZZ"};
        
        int passed = 0;
        for (const auto& seed : test_seeds) {
            if (test_seed_trace(trace_file, seed)) {
                passed++;
            }
        }
        
        std::cout << "\n=== Final Results ===" << std::endl;
        std::cout << "Passed: " << passed << "/" << test_seeds.size() << std::endl;
        
        if (passed == test_seeds.size()) {
            std::cout << "SUCCESS: All seeds match! ✓" << std::endl;
            return 0;
        } else {
            std::cout << "FAILURE: Some seeds don't match ✗" << std::endl;
            return 1;
        }
    }
    
    return 0;
}