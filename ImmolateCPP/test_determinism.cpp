/*
 * Determinism Self-Test Runner
 * Validates FP math consistency
 */

#include <cstdio>
#include "src/determinism_test.hpp"

int main() {
    printf("=== Determinism Self-Test ===\n\n");
    
    // Open log for detailed output
    FILE* log = stdout;
    
    // Run self-test
    bool passed = determinism_selftest(log);
    
    printf("\n=== Test Result ===\n");
    if (passed) {
        printf("✅ PASS: All determinism checks passed\n");
        printf("✅ FP math is consistent\n");
        printf("✅ Safe to use GPU acceleration\n");
        return 0;
    } else {
        printf("❌ FAIL: Determinism checks failed\n");
        printf("⚠️  System would fall back to CPU\n");
        printf("⚠️  Check compiler flags and environment\n");
        return 1;
    }
}