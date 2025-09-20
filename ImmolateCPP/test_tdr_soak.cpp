/*
 * TDR Soak Test
 * Runs continuous GPU searches for 5 minutes with pool updates to verify TDR safety
 */

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <thread>
#include <atomic>
#include <windows.h>
#include "src/gpu/gpu_types.h"

// Import GPU functions
extern "C" std::string gpu_search_with_driver(
    const std::string& start_seed_str,
    const FilterParams& params
);

extern "C" void brainstorm_set_target_ms(uint32_t ms);
extern "C" void brainstorm_update_pools(const char* json_utf8);
extern "C" void brainstorm_calibrate();
extern "C" double brainstorm_get_throughput();

// Sample pool JSON for updates
const char* POOL_UPDATE_1 = R"({
    "ctx": [
        {"key": "Voucher", "weighted": false, "items": ["v1", "v2", "v3", "v4", "v5"]},
        {"key": "PackSlot1", "weighted": false, "items": ["p1", "p2", "p3", "p4"]},
        {"key": "PackSlot2", "weighted": false, "items": ["p1", "p2", "p3", "p4"]},
        {"key": "Tag_small", "weighted": false, "items": ["t1", "t2", "t3", "t4", "t5", "t6"]},
        {"key": "Tag_big", "weighted": false, "items": ["t1", "t2", "t3", "t4", "t5", "t6"]}
    ]
})";

const char* POOL_UPDATE_2 = R"({
    "ctx": [
        {"key": "Voucher", "weighted": true, "items": ["v1", "v2", "v3"], "weights": [10, 5, 3]},
        {"key": "PackSlot1", "weighted": true, "items": ["p1", "p2"], "weights": [7, 3]},
        {"key": "PackSlot2", "weighted": true, "items": ["p1", "p2"], "weights": [7, 3]},
        {"key": "Tag_small", "weighted": true, "items": ["t1", "t2", "t3"], "weights": [5, 3, 2]},
        {"key": "Tag_big", "weighted": true, "items": ["t1", "t2", "t3"], "weights": [5, 3, 2]}
    ]
})";

void run_tdr_soak_test() {
    printf("=== TDR Soak Test (5 minutes) ===\n");
    printf("Testing continuous GPU operation with pool updates...\n\n");
    
    // Set conservative target to avoid TDR
    brainstorm_set_target_ms(250);
    printf("Target kernel time set to 250ms (TDR limit is 2000ms)\n");
    
    // Initial calibration
    printf("Running initial calibration...\n");
    brainstorm_calibrate();
    double initial_throughput = brainstorm_get_throughput();
    printf("Initial throughput: %.2f M seeds/sec\n\n", initial_throughput / 1000000.0);
    
    // Test parameters
    FilterParams params = {};
    params.tag1_small = 0;
    params.tag1_big = 0;
    params.tag2_small = params.tag2_big = 0xFFFFFFFF;
    params.voucher = 1;
    params.pack1 = params.pack2 = 0xFFFFFFFF;
    params.require_souls = 0;
    params.require_observatory = 0;
    params.require_perkeo = 0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_update = start_time;
    
    int searches_completed = 0;
    int pool_updates = 0;
    int tdr_timeouts = 0;
    bool use_pool_1 = true;
    
    const int TEST_DURATION_SECONDS = 300;  // 5 minutes
    const int UPDATE_INTERVAL_SECONDS = 60; // Update pools every minute
    
    printf("Starting 5-minute soak test...\n");
    printf("Will update pools every %d seconds\n", UPDATE_INTERVAL_SECONDS);
    printf("Press Ctrl+C to abort\n\n");
    
    while (true) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
        
        if (elapsed >= TEST_DURATION_SECONDS) {
            printf("\n5-minute test completed successfully!\n");
            break;
        }
        
        // Check if we should update pools
        auto since_update = std::chrono::duration_cast<std::chrono::seconds>(now - last_update).count();
        if (since_update >= UPDATE_INTERVAL_SECONDS) {
            printf("\n[%d sec] Updating pools...\n", (int)elapsed);
            brainstorm_update_pools(use_pool_1 ? POOL_UPDATE_1 : POOL_UPDATE_2);
            use_pool_1 = !use_pool_1;
            pool_updates++;
            last_update = now;
        }
        
        // Run a search
        char seed[9];
        sprintf(seed, "%08X", searches_completed);
        
        auto search_start = std::chrono::high_resolution_clock::now();
        std::string result = gpu_search_with_driver(seed, params);
        auto search_end = std::chrono::high_resolution_clock::now();
        
        auto search_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            search_end - search_start).count();
        
        searches_completed++;
        
        // Check for TDR timeout (search took > 2 seconds)
        if (search_ms > 2000) {
            tdr_timeouts++;
            printf("\n⚠️ WARNING: Search took %lld ms (possible TDR event)\n", search_ms);
        }
        
        // Status update every 10 searches
        if (searches_completed % 10 == 0) {
            double throughput = brainstorm_get_throughput();
            printf("[%3d sec] Searches: %4d | Pool updates: %d | Throughput: %.2f M/s | Last: %lld ms\n",
                   (int)elapsed, searches_completed, pool_updates, 
                   throughput / 1000000.0, search_ms);
        }
        
        // Small delay between searches
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Final report
    printf("\n=== TDR Soak Test Results ===\n");
    printf("Duration:         %d seconds\n", TEST_DURATION_SECONDS);
    printf("Searches:         %d\n", searches_completed);
    printf("Pool updates:     %d\n", pool_updates);
    printf("TDR timeouts:     %d\n", tdr_timeouts);
    printf("Final throughput: %.2f M seeds/sec\n", brainstorm_get_throughput() / 1000000.0);
    
    if (tdr_timeouts == 0) {
        printf("\n✅ PASS: No TDR timeouts detected\n");
        printf("✅ System remained stable throughout 5-minute test\n");
    } else {
        printf("\n⚠️ WARNING: %d potential TDR events detected\n", tdr_timeouts);
        printf("Consider reducing target_ms further\n");
    }
}

int main() {
    // Load the DLL first
    HMODULE dll = LoadLibraryA("Immolate.dll");
    if (!dll) {
        printf("ERROR: Failed to load Immolate.dll\n");
        return 1;
    }
    
    run_tdr_soak_test();
    
    FreeLibrary(dll);
    return 0;
}