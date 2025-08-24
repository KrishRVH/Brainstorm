// Unit tests for critical functions in Brainstorm ImmolateCPP
// Tests the most important functions that lack coverage

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <thread>
#include <vector>
#include <chrono>
#include <memory>

// Include the headers we're testing
#include "../util.hpp"
#include "../seed.hpp"
#include "../instance.hpp"
#include "../search.hpp"
#include "../functions.hpp"

// Test fixture for RNG functions
class UtilTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Known test values from actual Balatro game
        test_seeds = {"TESTTEST", "AAAAAAAA", "ZZZZZZZZ", "12345678"};
    }
    
    std::vector<std::string> test_seeds;
};

// Test pseudohash function for consistency and edge cases
TEST_F(UtilTest, PseudohashConsistency) {
    // Test that same input always produces same output
    for (const auto& seed : test_seeds) {
        double result1 = pseudohash(seed);
        double result2 = pseudohash(seed);
        EXPECT_DOUBLE_EQ(result1, result2) << "pseudohash not deterministic for seed: " << seed;
    }
}

TEST_F(UtilTest, PseudohashRange) {
    // Test that pseudohash returns values in [0, 1)
    for (const auto& seed : test_seeds) {
        double result = pseudohash(seed);
        EXPECT_GE(result, 0.0) << "pseudohash returned negative value for: " << seed;
        EXPECT_LT(result, 1.0) << "pseudohash returned value >= 1.0 for: " << seed;
    }
}

TEST_F(UtilTest, PseudohashEdgeCases) {
    // Test edge cases
    EXPECT_NO_THROW(pseudohash(""));  // Empty string
    EXPECT_NO_THROW(pseudohash("A")); // Single character
    EXPECT_NO_THROW(pseudohash(std::string(1000, 'X'))); // Very long string
    
    // Test that different inputs produce different outputs
    EXPECT_NE(pseudohash("AAAAAAAA"), pseudohash("AAAAAAAB"));
    EXPECT_NE(pseudohash("TEST1234"), pseudohash("1234TEST"));
}

// Test LuaRandom class
class LuaRandomTest : public ::testing::Test {
protected:
    void SetUp() override {
        rng1 = std::make_unique<LuaRandom>(0.5);
        rng2 = std::make_unique<LuaRandom>(0.5);
        rng_different = std::make_unique<LuaRandom>(0.7);
    }
    
    std::unique_ptr<LuaRandom> rng1, rng2, rng_different;
};

TEST_F(LuaRandomTest, DeterministicSameSeed) {
    // Same seed should produce same sequence
    for (int i = 0; i < 100; ++i) {
        EXPECT_DOUBLE_EQ(rng1->random(), rng2->random()) 
            << "RNG not deterministic at step " << i;
    }
}

TEST_F(LuaRandomTest, DifferentSeedsDiffer) {
    // Different seeds should produce different sequences
    bool found_difference = false;
    for (int i = 0; i < 10; ++i) {
        if (rng1->random() != rng_different->random()) {
            found_difference = true;
            break;
        }
    }
    EXPECT_TRUE(found_difference) << "Different seeds produced identical sequences";
}

TEST_F(LuaRandomTest, RandomRange) {
    // Test that random() returns values in [0, 1)
    for (int i = 0; i < 1000; ++i) {
        double val = rng1->random();
        EXPECT_GE(val, 0.0) << "Random value negative at step " << i;
        EXPECT_LT(val, 1.0) << "Random value >= 1.0 at step " << i;
    }
}

TEST_F(LuaRandomTest, RandintRange) {
    // Test randint range compliance
    for (int i = 0; i < 100; ++i) {
        int val = rng1->randint(5, 15);
        EXPECT_GE(val, 5) << "randint below minimum at step " << i;
        EXPECT_LE(val, 15) << "randint above maximum at step " << i;
    }
    
    // Test edge case: min == max
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(rng1->randint(42, 42), 42);
    }
}

// Test fract function
TEST(MathUtilTest, FractBasicCases) {
    EXPECT_DOUBLE_EQ(fract(3.14159), 0.14159);
    EXPECT_DOUBLE_EQ(fract(-2.5), -0.5);
    EXPECT_DOUBLE_EQ(fract(0.0), 0.0);
    EXPECT_DOUBLE_EQ(fract(1.0), 0.0);
    EXPECT_DOUBLE_EQ(fract(-1.0), 0.0);
}

TEST(MathUtilTest, FractEdgeCases) {
    // Test very large numbers
    EXPECT_EQ(fract(1e20), 0.0); // Should be 0 for integers too large for fractional part
    
    // Test very small numbers
    double small = 1e-15;
    EXPECT_DOUBLE_EQ(fract(small), small);
    
    // Test NaN and infinity
    EXPECT_TRUE(std::isnan(fract(std::numeric_limits<double>::quiet_NaN())));
}

// Test Seed class
class SeedTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_seed = std::make_unique<Seed>("TESTTEST");
    }
    
    std::unique_ptr<Seed> test_seed;
};

TEST_F(SeedTest, SeedConstruction) {
    EXPECT_NO_THROW(Seed("AAAAAAAA"));
    EXPECT_NO_THROW(Seed("12345678"));
    EXPECT_NO_THROW(Seed("ZZZZZZZZ"));
}

TEST_F(SeedTest, SeedProgression) {
    std::string initial = test_seed->tostring();
    test_seed->next();
    std::string after_next = test_seed->tostring();
    
    EXPECT_NE(initial, after_next) << "Seed didn't change after next()";
    
    // Test that multiple next() calls produce different seeds
    std::set<std::string> unique_seeds;
    unique_seeds.insert(initial);
    
    Seed test("TESTTEST");
    for (int i = 0; i < 100; ++i) {
        test.next();
        unique_seeds.insert(test.tostring());
    }
    
    EXPECT_GT(unique_seeds.size(), 90) << "Seed progression not diverse enough";
}

// Test Instance class critical functions
class InstanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        seed = std::make_unique<Seed>("TESTTEST");
        instance = std::make_unique<Instance>(*seed);
    }
    
    std::unique_ptr<Seed> seed;
    std::unique_ptr<Instance> instance;
};

TEST_F(InstanceTest, LockUnlockSystem) {
    Item test_item = Item::The_Fool;
    
    // Initially unlocked
    EXPECT_FALSE(instance->isLocked(test_item));
    
    // Lock and verify
    instance->lock(test_item);
    EXPECT_TRUE(instance->isLocked(test_item));
    
    // Unlock and verify
    instance->unlock(test_item);
    EXPECT_FALSE(instance->isLocked(test_item));
}

TEST_F(InstanceTest, RandomConsistency) {
    // Same key should produce same result for same instance state
    std::string key = "test_key";
    double result1 = instance->random(key);
    
    // Reset to same state
    instance->reset(*seed);
    double result2 = instance->random(key);
    
    EXPECT_DOUBLE_EQ(result1, result2) << "Instance random not deterministic";
}

TEST_F(InstanceTest, NextProgression) {
    std::string key = "test_progression";
    double initial = instance->random(key);
    
    instance->next();
    double after_next = instance->random(key);
    
    EXPECT_NE(initial, after_next) << "Instance state didn't change after next()";
}

// Test Search class thread safety
class SearchTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Simple filter that accepts every 100th seed
        test_filter = [](Instance inst) -> int {
            return (inst.seed.getID() % 100 == 0) ? 1 : 0;
        };
    }
    
    std::function<int(Instance)> test_filter;
};

TEST_F(SearchTest, SingleThreadedSearch) {
    Search search(test_filter, 1, 10000);
    search.exitOnFind = true;
    
    std::string result = search.search();
    EXPECT_FALSE(result.empty()) << "Single-threaded search found no results";
}

TEST_F(SearchTest, MultiThreadedSearch) {
    Search search(test_filter, 4, 10000);
    search.exitOnFind = true;
    
    std::string result = search.search();
    EXPECT_FALSE(result.empty()) << "Multi-threaded search found no results";
}

TEST_F(SearchTest, ThreadSafety) {
    // Test that multi-threaded search produces consistent results
    std::vector<std::string> results;
    
    for (int i = 0; i < 5; ++i) {
        Search search(test_filter, 4, 1000);
        search.exitOnFind = true;
        results.push_back(search.search());
    }
    
    // All runs should find the same first matching seed
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_EQ(results[0], results[i]) << "Thread safety violation - different results";
    }
}

// Memory leak detection test
TEST(MemoryTest, NoLeaksInBasicOperations) {
    // This test would ideally use AddressSanitizer or Valgrind
    // For now, just test that basic operations complete without crashing
    
    for (int i = 0; i < 1000; ++i) {
        Seed s("TEST" + std::to_string(i % 100).substr(0, 4));
        Instance inst(s);
        
        // Exercise various functions
        inst.random("test_key" + std::to_string(i));
        inst.next();
        inst.lock(Item::The_Fool);
        inst.unlock(Item::The_Fool);
    }
    
    SUCCEED() << "Basic operations completed without crashes";
}

// Performance regression test
TEST(PerformanceTest, PseudohashPerformance) {
    const int NUM_ITERATIONS = 100000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        pseudohash("TEST" + std::to_string(i % 1000));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Should complete 100k hashes in reasonable time (adjust threshold as needed)
    EXPECT_LT(duration.count(), 1000000) << "Pseudohash performance regression detected";
    
    std::cout << "Pseudohash performance: " << NUM_ITERATIONS << " iterations in " 
              << duration.count() << " μs (" << (NUM_ITERATIONS * 1000000.0 / duration.count()) 
              << " ops/sec)" << std::endl;
}

// Integration test for the full pipeline
TEST(IntegrationTest, FullPipeline) {
    // Test the complete seed -> instance -> filter pipeline
    auto voucher_filter = [](Instance inst) -> int {
        // Try to find a specific voucher in first shop
        inst.initLocks(1, false, true);
        Item voucher = inst.nextVoucher(1);
        return (voucher == Item::Overstock) ? 1 : 0;
    };
    
    Search search(voucher_filter, "TESTTEST", 2, 10000);
    search.exitOnFind = true;
    
    std::string result = search.search();
    // Don't require a result (might be rare), just test that it doesn't crash
    SUCCEED() << "Full pipeline integration test completed";
}

// Main function for standalone test execution
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "Running critical function tests for Brainstorm ImmolateCPP..." << std::endl;
    std::cout << "Test categories: RNG, Seed progression, Instance state, Search threading" << std::endl;
    
    return RUN_ALL_TESTS();
}