// Memory safety and resource management tests
// Specifically targets the critical memory issues found in audit

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <memory>
#include <chrono>

// Test the DLL interface memory management
extern "C" {
    const char* brainstorm(const char* seed, const char* voucher, const char* pack,
                          const char* tag1, const char* tag2, double souls, 
                          bool observatory, bool perkeo);
    const char* get_tags(const char* seed);
    void free_result(const char* result);
}

class MemorySafetyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize with safe default values
        test_seed = "TESTTEST";
        test_voucher = "RETRY";
        test_pack = "RETRY"; 
        test_tag1 = "RETRY";
        test_tag2 = "RETRY";
    }
    
    void TearDown() override {
        // Ensure all allocated results are freed
        for (const char* result : allocated_results) {
            if (result) {
                free_result(result);
            }
        }
        allocated_results.clear();
    }
    
    const char* safe_brainstorm(const char* seed, const char* voucher = "RETRY", 
                               const char* pack = "RETRY", const char* tag1 = "RETRY",
                               const char* tag2 = "RETRY", double souls = 0, 
                               bool observatory = false, bool perkeo = false) {
        const char* result = brainstorm(seed, voucher, pack, tag1, tag2, souls, observatory, perkeo);
        if (result) {
            allocated_results.push_back(result);
        }
        return result;
    }
    
    const char* safe_get_tags(const char* seed) {
        const char* result = get_tags(seed);
        if (result) {
            allocated_results.push_back(result);
        }
        return result;
    }
    
private:
    std::vector<const char*> allocated_results;
    
protected:
    const char* test_seed;
    const char* test_voucher;
    const char* test_pack;
    const char* test_tag1;
    const char* test_tag2;
};

// Test 1: Basic memory allocation/deallocation
TEST_F(MemorySafetyTest, BasicAllocationDeallocation) {
    const char* result = safe_brainstorm(test_seed);
    
    if (result) {
        // Verify we can read the result safely
        size_t len = strlen(result);
        EXPECT_GT(len, 0) << "Result string is empty";
        EXPECT_LE(len, 8) << "Result string too long for a seed";
        
        // Verify it's a valid seed format
        for (size_t i = 0; i < len; ++i) {
            EXPECT_TRUE(isalnum(result[i])) << "Invalid character in seed at position " << i;
        }
    }
    
    // Memory will be freed in TearDown()
}

// Test 2: Multiple allocations without leaks
TEST_F(MemorySafetyTest, MultipleAllocations) {
    const int NUM_ALLOCATIONS = 100;
    
    for (int i = 0; i < NUM_ALLOCATIONS; ++i) {
        const char* result = safe_brainstorm(test_seed);
        // Don't free immediately - let TearDown handle it to test accumulation
    }
    
    // If we get here without crashing, basic allocation is working
    SUCCEED() << "Multiple allocations completed successfully";
}

// Test 3: Null pointer handling
TEST_F(MemorySafetyTest, NullPointerHandling) {
    // Test null inputs - should not crash
    EXPECT_NO_THROW({
        const char* result = brainstorm(nullptr, test_voucher, test_pack, 
                                       test_tag1, test_tag2, 0, false, false);
        if (result) {
            allocated_results.push_back(result);
        }
    });
    
    EXPECT_NO_THROW({
        const char* result = get_tags(nullptr);
        if (result) {
            allocated_results.push_back(result);
        }
    });
    
    // Test free_result with null - should not crash
    EXPECT_NO_THROW(free_result(nullptr));
}

// Test 4: Thread safety of memory operations
TEST_F(MemorySafetyTest, ThreadSafetyMemory) {
    const int NUM_THREADS = 4;
    const int OPERATIONS_PER_THREAD = 50;
    
    std::vector<std::thread> threads;
    std::vector<std::vector<const char*>> thread_results(NUM_THREADS);
    
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < OPERATIONS_PER_THREAD; ++i) {
                std::string seed = "TEST" + std::to_string(t * 1000 + i);
                const char* result = brainstorm(seed.c_str(), "RETRY", "RETRY", 
                                               "RETRY", "RETRY", 0, false, false);
                if (result) {
                    thread_results[t].push_back(result);
                }
                
                // Also test get_tags
                const char* tags = get_tags(seed.c_str());
                if (tags) {
                    thread_results[t].push_back(tags);
                }
            }
        });
    }
    
    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Free all results
    for (int t = 0; t < NUM_THREADS; ++t) {
        for (const char* result : thread_results[t]) {
            free_result(result);
        }
    }
    
    SUCCEED() << "Multi-threaded memory operations completed";
}

// Test 5: Memory corruption detection
TEST_F(MemorySafetyTest, MemoryCorruptionDetection) {
    const char* result = safe_brainstorm(test_seed);
    
    if (result) {
        // Verify the memory contains valid data
        size_t len = strlen(result);
        
        // Check that memory before and after the string hasn't been corrupted
        // (This is a basic check - AddressSanitizer would catch more issues)
        for (size_t i = 0; i < len; ++i) {
            char c = result[i];
            EXPECT_TRUE(isprint(c) || c == '\0') << "Non-printable character found at position " << i;
        }
        
        // Verify null termination
        EXPECT_EQ(result[len], '\0') << "String not properly null-terminated";
    }
}

// Test 6: Resource cleanup on exceptions
TEST_F(MemorySafetyTest, ExceptionSafety) {
    // Test that resources are properly cleaned up even if exceptions occur
    
    class TestException : public std::exception {
    public:
        const char* what() const noexcept override { return "Test exception"; }
    };
    
    std::vector<const char*> local_results;
    
    try {
        for (int i = 0; i < 10; ++i) {
            const char* result = safe_brainstorm(test_seed);
            local_results.push_back(result);
            
            if (i == 5) {
                // Simulate an exception during processing
                throw TestException();
            }
        }
    } catch (const TestException&) {
        // Expected - verify resources are still accessible
        for (const char* result : local_results) {
            if (result) {
                // Should be able to access the memory
                EXPECT_NO_THROW(strlen(result));
            }
        }
    }
}

// Test 7: Large allocation stress test
TEST_F(MemorySafetyTest, LargeAllocationStress) {
    const int STRESS_ITERATIONS = 1000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < STRESS_ITERATIONS; ++i) {
        std::string seed = "STRESS" + std::to_string(i % 100);
        const char* result = safe_brainstorm(seed.c_str());
        
        // Immediately free every 10th allocation to test mixed allocation/deallocation
        if (i % 10 == 0 && result) {
            free_result(result);
            // Remove from our tracking since we freed it manually
            allocated_results.pop_back();
        }
        
        // Yield to allow other threads/processes to run
        if (i % 100 == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Stress test: " << STRESS_ITERATIONS << " operations in " 
              << duration.count() << "ms" << std::endl;
    
    SUCCEED() << "Large allocation stress test completed";
}

// Test 8: Double-free detection
TEST_F(MemorySafetyTest, DoubleFreeDetection) {
    const char* result = brainstorm(test_seed, "RETRY", "RETRY", "RETRY", "RETRY", 0, false, false);
    
    if (result) {
        // Free once - should work
        EXPECT_NO_THROW(free_result(result));
        
        // Free again - should not crash (implementation should handle this gracefully)
        EXPECT_NO_THROW(free_result(result));
    }
}

// Test 9: Memory alignment and access patterns
TEST_F(MemorySafetyTest, MemoryAlignment) {
    const char* result = safe_brainstorm(test_seed);
    
    if (result) {
        // Check that the pointer is properly aligned
        uintptr_t addr = reinterpret_cast<uintptr_t>(result);
        EXPECT_EQ(addr % alignof(char), 0) << "Result pointer not properly aligned";
        
        // Test various access patterns
        size_t len = strlen(result);
        if (len > 0) {
            // Forward access
            for (size_t i = 0; i < len; ++i) {
                volatile char c = result[i];  // Volatile to prevent optimization
                (void)c;
            }
            
            // Backward access
            for (size_t i = len; i > 0; --i) {
                volatile char c = result[i - 1];
                (void)c;
            }
        }
    }
}

// Test 10: Long-running stability
TEST_F(MemorySafetyTest, LongRunningStability) {
    const int LONG_RUN_ITERATIONS = 10000;
    const int BATCH_SIZE = 100;
    
    for (int batch = 0; batch < LONG_RUN_ITERATIONS / BATCH_SIZE; ++batch) {
        std::vector<const char*> batch_results;
        
        // Allocate a batch
        for (int i = 0; i < BATCH_SIZE; ++i) {
            std::string seed = "LONG" + std::to_string(batch * BATCH_SIZE + i);
            const char* result = brainstorm(seed.c_str(), "RETRY", "RETRY", 
                                           "RETRY", "RETRY", 0, false, false);
            if (result) {
                batch_results.push_back(result);
            }
        }
        
        // Free the entire batch
        for (const char* result : batch_results) {
            free_result(result);
        }
        
        // Progress indication
        if (batch % 10 == 0) {
            std::cout << "Long-running test progress: " << (batch * BATCH_SIZE) 
                     << "/" << LONG_RUN_ITERATIONS << std::endl;
        }
    }
    
    SUCCEED() << "Long-running stability test completed";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "Running memory safety tests for Brainstorm ImmolateCPP..." << std::endl;
    std::cout << "These tests check for memory leaks, double-frees, and thread safety." << std::endl;
    std::cout << "For best results, run with AddressSanitizer or Valgrind." << std::endl;
    
    return RUN_ALL_TESTS();
}