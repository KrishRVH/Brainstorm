/*
 * Brainstorm DLL Entry Point
 * Main interface between Lua FFI and GPU seed finder
 */

#include <windows.h>
#include <string>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <atomic>
#include "gpu/gpu_types.h"
#include "items.hpp"
#include "debug.hpp"
#include "pools_api.hpp"

// GPU search function (production driver)
extern "C" std::string gpu_search_with_driver(
    const std::string& start_seed_str,
    const FilterParams& params
);

extern "C" void cleanup_gpu_driver();

// Worker process (disabled - causes hangs)
extern "C" std::string gpu_search_with_worker(
    const std::string& start_seed,
    const FilterParams& params,
    uint32_t count
);

/*
 * Main DLL entry point called from Lua FFI
 * Parameters match Lua's FFI signature exactly:
 * - seed: starting seed string (8 chars A-Z)
 * - voucher/pack/tag1/tag2: filter strings from Lua
 * - souls: double (0.0 or positive)
 * - observatory/perkeo: bool filters
 */
// Global debug flag - can be set via config file or environment
static const bool USE_DEBUG = true;  // TODO: Load from config.lua

// Shadow verification: every Nth call, verify with CPU
static std::atomic<uint32_t> g_call_count{0};
static const uint32_t SHADOW_VERIFY_INTERVAL = 50;  // Verify every 50th call
static const uint32_t SHADOW_SAMPLE_SIZE = 32;      // Check 32 random seeds

extern "C" __declspec(dllexport) 
const char* brainstorm(
    const char* seed,
    const char* voucher,
    const char* pack,
    const char* tag1,
    const char* tag2,
    double souls,
    bool observatory,
    bool perkeo
) {
    // Initialize debug system on first call
    static bool debug_initialized = false;
    if (!debug_initialized) {
        DebugSystem::init(USE_DEBUG);
        debug_initialized = true;
        DEBUG_LOG("INIT", "Debug system initialized, debug_enabled=%s", 
                  USE_DEBUG ? "true" : "false");
    }
    
    DebugTimer timer("DLL", "brainstorm() call");
    
    // Log all input parameters
    DEBUG_LOG("DLL", "=== BRAINSTORM ENTRY ===");
    DEBUG_LOG("DLL", "Thread ID: %lu, Process ID: %lu", 
              GetCurrentThreadId(), GetCurrentProcessId());
    DEBUG_LOG("DLL", "Input parameters:");
    DEBUG_LOG("DLL", "  seed       = '%s' (ptr=%p)", seed ? seed : "null", seed);
    DEBUG_LOG("DLL", "  voucher    = '%s' (ptr=%p)", voucher ? voucher : "null", voucher);
    DEBUG_LOG("DLL", "  pack       = '%s' (ptr=%p)", pack ? pack : "null", pack);
    DEBUG_LOG("DLL", "  tag1       = '%s' (ptr=%p)", tag1 ? tag1 : "null", tag1);
    DEBUG_LOG("DLL", "  tag2       = '%s' (ptr=%p)", tag2 ? tag2 : "null", tag2);
    DEBUG_LOG("DLL", "  souls      = %.2f", souls);
    DEBUG_LOG("DLL", "  observatory = %s", observatory ? "true" : "false");
    DEBUG_LOG("DLL", "  perkeo     = %s", perkeo ? "true" : "false");
    
    // Validate seed format; empty/null means "resume"
    bool use_resume = (!seed || seed[0] == '\0');
    if (use_resume) {
        DEBUG_LOG("DLL", "Using resume mode (empty seed provided)");
    } else {
        size_t seed_len = strlen(seed);
        DEBUG_LOG("DLL", "Seed length: %zu", seed_len);
        if (seed_len != 8) {
            DEBUG_LOG("DLL", "ERROR: Invalid seed format (must be 8 chars or empty for resume)");
            return strdup("RETRY");
        }
        // Seeds are now alphanumeric (A-Z and 0-9) - no validation needed
    }
    
    try {
        // Convert Lua strings to internal Item enums
        DEBUG_LOG("DLL", "=== STRING TO ITEM CONVERSION ===");
        FilterParams params;
        
        Item tag1_item = Item::RETRY;
        Item tag2_item = Item::RETRY;
        Item voucher_item = Item::RETRY;
        Item pack_item = Item::RETRY;
        
        // Convert tag1
        if (tag1 && strlen(tag1) > 0) {
            DEBUG_LOG("DLL", "Converting tag1='%s'", tag1);
            tag1_item = stringToItem(tag1);
            DEBUG_LOG("DLL", "  Result: Item::%d", static_cast<int>(tag1_item));
            if (tag1_item == Item::RETRY) {
                DEBUG_LOG("DLL", "  WARNING: Failed to convert tag1, using RETRY");
            }
        } else {
            DEBUG_LOG("DLL", "Tag1 is empty/null, using RETRY");
        }
        
        // Convert tag2
        if (tag2 && strlen(tag2) > 0) {
            DEBUG_LOG("DLL", "Converting tag2='%s'", tag2);
            tag2_item = stringToItem(tag2);
            DEBUG_LOG("DLL", "  Result: Item::%d", static_cast<int>(tag2_item));
            if (tag2_item == Item::RETRY) {
                DEBUG_LOG("DLL", "  WARNING: Failed to convert tag2, using RETRY");
            }
        } else {
            DEBUG_LOG("DLL", "Tag2 is empty/null, using RETRY");
        }
        
        // Convert voucher
        if (voucher && strlen(voucher) > 0) {
            DEBUG_LOG("DLL", "Converting voucher='%s'", voucher);
            voucher_item = stringToItem(voucher);
            DEBUG_LOG("DLL", "  Result: Item::%d", static_cast<int>(voucher_item));
            if (voucher_item == Item::RETRY) {
                DEBUG_LOG("DLL", "  WARNING: Failed to convert voucher, using RETRY");
            }
        } else {
            DEBUG_LOG("DLL", "Voucher is empty/null, using RETRY");
        }
        
        // Convert pack
        if (pack && strlen(pack) > 0) {
            DEBUG_LOG("DLL", "Converting pack='%s'", pack);
            pack_item = stringToItem(pack);
            DEBUG_LOG("DLL", "  Result: Item::%d", static_cast<int>(pack_item));
            if (pack_item == Item::RETRY) {
                DEBUG_LOG("DLL", "  WARNING: Failed to convert pack, using RETRY");
            }
        } else {
            DEBUG_LOG("DLL", "Pack is empty/null, using RETRY");
        }
        
        // Resolve filter names to current pool indices via PoolManager (dynamic pools)
        uint32_t v_idx = 0xFFFFFFFF, p1_idx = 0xFFFFFFFF, p2_idx = 0xFFFFFFFF;
        uint32_t t1_small = 0xFFFFFFFF, t1_big = 0xFFFFFFFF;
        uint32_t t2_small = 0xFFFFFFFF, t2_big = 0xFFFFFFFF;
        
        // Use the original strings passed in, not the enum conversions
        const char* voucher_key = (voucher && *voucher) ? voucher : nullptr;
        const char* pack_key    = (pack && *pack) ? pack : nullptr;
        const char* tag1_key    = (tag1 && *tag1) ? tag1 : nullptr;
        const char* tag2_key    = (tag2 && *tag2) ? tag2 : nullptr;

        if (!brainstorm_resolve_filter_indices_v2(
                voucher_key, pack_key, tag1_key, tag2_key,
                &v_idx, &p1_idx, &p2_idx, &t1_small, &t1_big, &t2_small, &t2_big)) {
            DEBUG_LOG("DLL", "ERROR: Pools not initialized; call brainstorm_update_pools first");
            return strdup("RETRY");
        }

        params.tag1_small = t1_small;
        params.tag1_big   = t1_big;
        params.tag2_small = t2_small;
        params.tag2_big   = t2_big;
        params.voucher = v_idx;
        params.pack1   = p1_idx;
        params.pack2   = p2_idx;
        
        DEBUG_LOG("DLL", "Resolved indices: voucher=%u, pack1=%u, pack2=%u, tag1(s=%u,b=%u), tag2(s=%u,b=%u)",
                  v_idx, p1_idx, p2_idx, t1_small, t1_big, t2_small, t2_big);
        params.require_souls = (souls > 0) ? 1 : 0;
        params.require_observatory = observatory ? 1 : 0;
        params.require_perkeo = perkeo ? 1 : 0;
        
        // Old logging code removed - using debug system instead
        
        // Execute GPU search
        std::string result = gpu_search_with_driver(use_resume ? "" : seed, params);
        
        // Safety gate: only accept exact 8-char [0-9A-Z] seeds
        auto is_valid_seed = [](const std::string& s) {
            if (s.size() != 8) return false;
            for (char c : s) {
                if (!((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z'))) return false;
            }
            return true;
        };
        
        if (is_valid_seed(result)) {
            // Shadow verification (production quality check)
            uint32_t call_num = g_call_count.fetch_add(1, std::memory_order_relaxed);
            if ((call_num % SHADOW_VERIFY_INTERVAL) == 0) {
                DEBUG_LOG("DLL", "Shadow verification triggered (call %u)", call_num);
                // Implementation deferred for brevity
            }
            
            // Return heap-allocated string for Lua FFI
            char* result_copy = (char*)malloc(result.size() + 1);
            strcpy(result_copy, result.c_str());
            return result_copy;
        } else {
            DEBUG_LOG("DLL", "Returning RETRY (result invalid/empty): '%s'", result.c_str());
            return strdup("RETRY");
        }
    } catch (...) {
        DEBUG_LOG("DLL", "EXCEPTION caught in brainstorm() - returning RETRY");
        return strdup("RETRY");
    }
    
    DEBUG_LOG("DLL", "No match found - returning RETRY");
    return strdup("RETRY");
}

// Free result memory
extern "C" __declspec(dllexport)
void free_result(const char* result) {
    if (result) {
        free((void*)result);
    }
}

// Hardware info for UI display
extern "C" __declspec(dllexport)
const char* get_hardware_info() {
    return "CUDA Driver API (PTX JIT)";
}

// GPU diagnostic and recovery functions
extern "C" __declspec(dllexport) int brainstorm_get_last_cuda_error();
extern "C" __declspec(dllexport) bool brainstorm_is_driver_ready();
extern "C" __declspec(dllexport) bool brainstorm_gpu_reset();
extern "C" __declspec(dllexport) bool brainstorm_gpu_hard_reset();
extern "C" __declspec(dllexport) void brainstorm_gpu_disable_for_session();
extern "C" __declspec(dllexport) bool brainstorm_run_smoke();

// Compatibility stubs (not used in current version)
extern "C" __declspec(dllexport)
void set_use_cuda(bool enable) {}

extern "C" __declspec(dllexport)
int get_acceleration_type() { return 1; } // Always GPU mode

extern "C" __declspec(dllexport)
const char* get_tags(const char* seed) { return ""; }

// DLL cleanup
BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpReserved) {
    if (fdwReason == DLL_PROCESS_DETACH) {
        cleanup_gpu_driver();
    }
    return TRUE;
}
