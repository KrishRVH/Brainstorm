/*
 * Pool Manager - Handles dynamic pool updates from Lua
 */

#include <string>
#include <vector>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <atomic>
#include <mutex>
#include "gpu/pool_types.h"

// Use simple JSON parsing (minimal dependencies)
#include <sstream>
#include <map>

// External CUDA functions (from gpu_kernel_driver_prod.cpp)
extern "C" {
    void* cuda_alloc_device_mem(size_t size);
    void cuda_free_device_mem(void* ptr);
    bool cuda_copy_to_device(void* dst, const void* src, size_t size);
    void* get_cuda_context();
}

// Global pool state (double-buffered)
namespace PoolManager {
    // Double-buffered device allocations
    static void* device_pools[2] = {nullptr, nullptr};
    static std::atomic<int> active_pool_idx{0};
    static std::mutex update_mutex;
    
    // Host-side cache
    static std::vector<uint8_t> host_buffer;
    static std::atomic<size_t> buffer_size{0};
    
    // Track if pools have been initialized
    static std::atomic<bool> initialized{false};
    static std::atomic<bool> cuda_available{false};
    
    // Host-side item keys per context (0=voucher,1=pack1,2=pack2,3=tag_small,4=tag_big)
    static std::vector<std::string> host_items[5];
    static std::mutex items_mutex;  // Protect item cache access
}

// Helper: compute GCD of weights for scaling
static uint64_t gcd(uint64_t a, uint64_t b) {
    while (b) {
        uint64_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

static uint64_t gcd_array(const std::vector<uint64_t>& weights) {
    if (weights.empty()) return 1;
    uint64_t result = weights[0];
    for (size_t i = 1; i < weights.size(); i++) {
        result = gcd(result, weights[i]);
        if (result == 1) break;
    }
    return result;
}

// Helper: find index by exact key in given context; returns 0xFFFFFFFF on not found
static uint32_t find_index_in_context(int ctx_id, const char* key) {
    if (!key || !*key) return 0xFFFFFFFFu;
    if (ctx_id < 0 || ctx_id >= 5) return 0xFFFFFFFFu;
    
    std::lock_guard<std::mutex> lock(PoolManager::items_mutex);
    const auto& vec = PoolManager::host_items[ctx_id];
    for (uint32_t i = 0; i < vec.size(); ++i) {
        if (vec[i] == key) return i;
    }
    return 0xFFFFFFFFu;
}

// Resolve UI filter names to per-context indices into current pools.
// Out params are set to 0xFFFFFFFF if empty/not found.
extern "C" __declspec(dllexport)
bool brainstorm_resolve_filter_indices(
    const char* voucher_key,
    const char* pack_key,
    const char* tag1_key,
    const char* tag2_key,
    // outputs:
    uint32_t* out_voucher_idx,
    uint32_t* out_pack1_idx,
    uint32_t* out_pack2_idx,
    uint32_t* out_tag_idx1,
    uint32_t* out_tag_idx2
) {
    if (!PoolManager::initialized.load(std::memory_order_acquire)) return false;

    // Default "no filter"
    auto nf = [](uint32_t* p){ if (p) *p = 0xFFFFFFFFu; };
    nf(out_voucher_idx); 
    nf(out_pack1_idx); 
    nf(out_pack2_idx); 
    nf(out_tag_idx1); 
    nf(out_tag_idx2);

    if (out_voucher_idx && voucher_key && *voucher_key) {
        *out_voucher_idx = find_index_in_context(0, voucher_key);
    }
    if (pack_key && *pack_key) {
        if (out_pack1_idx) *out_pack1_idx = find_index_in_context(1, pack_key);
        if (out_pack2_idx) *out_pack2_idx = find_index_in_context(2, pack_key);
    }
    if (tag1_key && *tag1_key) {
        if (out_tag_idx1) *out_tag_idx1 = find_index_in_context(3, tag1_key);
        // Note: Assuming tag_small and tag_big share order for RC1
    }
    if (tag2_key && *tag2_key) {
        if (out_tag_idx2) *out_tag_idx2 = find_index_in_context(3, tag2_key);
    }

    return true;
}

// V2: Resolve UI filter names with separate tag contexts
extern "C" __declspec(dllexport)
bool brainstorm_resolve_filter_indices_v2(
    const char* voucher_key,
    const char* pack_key,
    const char* tag1_key,
    const char* tag2_key,
    uint32_t* out_voucher_idx,
    uint32_t* out_pack1_idx,
    uint32_t* out_pack2_idx,
    uint32_t* out_tag1_small_idx,
    uint32_t* out_tag1_big_idx,
    uint32_t* out_tag2_small_idx,
    uint32_t* out_tag2_big_idx
) {
    if (!PoolManager::initialized.load(std::memory_order_acquire)) return false;
    
    auto nf = [](uint32_t* p){ if (p) *p = 0xFFFFFFFFu; };
    nf(out_voucher_idx); 
    nf(out_pack1_idx); 
    nf(out_pack2_idx);
    nf(out_tag1_small_idx); 
    nf(out_tag1_big_idx);
    nf(out_tag2_small_idx); 
    nf(out_tag2_big_idx);

    // Voucher
    if (out_voucher_idx && voucher_key && *voucher_key) {
        *out_voucher_idx = find_index_in_context(0, voucher_key);
    }

    // Packs
    if (pack_key && *pack_key) {
        if (out_pack1_idx) *out_pack1_idx = find_index_in_context(1, pack_key);
        if (out_pack2_idx) *out_pack2_idx = find_index_in_context(2, pack_key);
    }

    // Tags: resolve separately for small (ctx=3) and big (ctx=4)
    if (tag1_key && *tag1_key) {
        if (out_tag1_small_idx) *out_tag1_small_idx = find_index_in_context(3, tag1_key);
        if (out_tag1_big_idx)   *out_tag1_big_idx   = find_index_in_context(4, tag1_key);
    }
    if (tag2_key && *tag2_key) {
        if (out_tag2_small_idx) *out_tag2_small_idx = find_index_in_context(3, tag2_key);
        if (out_tag2_big_idx)   *out_tag2_big_idx   = find_index_in_context(4, tag2_key);
    }
    
    return true;
}

// Simple JSON parser for pool data
struct PoolContext {
    std::string ctx_key;
    std::vector<std::string> items;
    std::vector<uint64_t> weights;
    uint64_t total;
    bool weighted;
};

struct PoolSnapshot {
    PoolContext voucher;
    PoolContext pack1;
    PoolContext pack2;
    PoolContext tag_small;
    PoolContext tag_big;
};

// Simple but robust JSON parser for pool data
static PoolSnapshot parse_pool_json(const char* json_str) {
    PoolSnapshot snap;
    std::string json(json_str ? json_str : "");
    
    // Helper lambda to find and extract string value
    auto extract_string = [&](const std::string& key, size_t start_pos = 0) -> std::string {
        size_t pos = json.find("\"" + key + "\"", start_pos);
        if (pos == std::string::npos) return "";
        
        pos = json.find(":", pos);
        if (pos == std::string::npos) return "";
        
        pos = json.find("\"", pos);
        if (pos == std::string::npos) return "";
        
        size_t end = json.find("\"", pos + 1);
        if (end == std::string::npos) return "";
        
        return json.substr(pos + 1, end - pos - 1);
    };
    
    // Helper lambda to extract array of strings
    auto extract_array = [&](const std::string& key, size_t start_pos = 0) -> std::vector<std::string> {
        std::vector<std::string> result;
        size_t pos = json.find("\"" + key + "\"", start_pos);
        if (pos == std::string::npos) return result;
        
        pos = json.find("[", pos);
        if (pos == std::string::npos) return result;
        
        size_t end = json.find("]", pos);
        if (end == std::string::npos) return result;
        
        size_t item_start = pos + 1;
        while (item_start < end) {
            size_t quote_start = json.find("\"", item_start);
            if (quote_start == std::string::npos || quote_start >= end) break;
            
            size_t quote_end = json.find("\"", quote_start + 1);
            if (quote_end == std::string::npos || quote_end >= end) break;
            
            result.push_back(json.substr(quote_start + 1, quote_end - quote_start - 1));
            item_start = quote_end + 1;
        }
        
        return result;
    };
    
    // Helper lambda to extract array of numbers
    auto extract_weights = [&](const std::string& key, size_t start_pos = 0) -> std::vector<uint64_t> {
        std::vector<uint64_t> result;
        size_t pos = json.find("\"" + key + "\"", start_pos);
        if (pos == std::string::npos) return result;
        
        pos = json.find("[", pos);
        if (pos == std::string::npos) return result;
        
        size_t end = json.find("]", pos);
        if (end == std::string::npos) return result;
        
        std::string array_str = json.substr(pos + 1, end - pos - 1);
        std::istringstream iss(array_str);
        std::string token;
        
        while (std::getline(iss, token, ',')) {
            try {
                // Trim whitespace
                token.erase(0, token.find_first_not_of(" \t\n\r"));
                token.erase(token.find_last_not_of(" \t\n\r") + 1);
                if (!token.empty()) {
                    result.push_back(std::stoull(token));
                }
            } catch (...) {
                // Skip invalid numbers
            }
        }
        
        return result;
    };
    
    // Helper lambda to parse a context
    auto parse_context = [&](PoolContext& ctx, const std::string& context_name) {
        size_t ctx_pos = json.find("\"" + context_name + "\"");
        if (ctx_pos == std::string::npos) return;
        
        // Find the context object boundaries
        size_t obj_start = json.find("{", ctx_pos);
        if (obj_start == std::string::npos) return;
        
        size_t obj_end = json.find("}", obj_start);
        if (obj_end == std::string::npos) return;
        
        // Extract fields within this context
        ctx.ctx_key = extract_string("ctx_key", obj_start);
        ctx.items = extract_array("items", obj_start);
        ctx.weights = extract_weights("weights", obj_start);
        
        // Weighted if weights provided and matches items count
        ctx.weighted = (!ctx.weights.empty() && ctx.weights.size() == ctx.items.size());
        
        // If uniform, create uniform weights sized to items
        if (!ctx.weighted && !ctx.items.empty()) {
            ctx.weights.resize(ctx.items.size(), 1);
        }
        
        // Validate matching lengths
        if (ctx.weights.size() != ctx.items.size()) {
            // Fallback: uniform
            ctx.weights.assign(ctx.items.size(), 1);
            ctx.weighted = false;
        }
        
        // Count of items
        ctx.total = ctx.items.size();
    };
    
    // Parse each context
    parse_context(snap.voucher, "voucher");
    parse_context(snap.pack1, "pack1");
    parse_context(snap.pack2, "pack2");
    parse_context(snap.tag_small, "tag_small");
    parse_context(snap.tag_big, "tag_big");
    
    // Handle special cases
    // Pack2 uses same key as pack1 if not specified
    if (snap.pack2.ctx_key.empty() && !snap.pack1.ctx_key.empty()) {
        snap.pack2.ctx_key = snap.pack1.ctx_key;
        snap.pack2.items = snap.pack1.items;
        snap.pack2.weights = snap.pack1.weights;
        snap.pack2.total = snap.pack1.total;
        snap.pack2.weighted = snap.pack1.weighted;
    }
    
    // Provide defaults if missing
    if (snap.voucher.ctx_key.empty()) snap.voucher.ctx_key = "Voucher";
    if (snap.pack1.ctx_key.empty()) snap.pack1.ctx_key = "shop_pack1";
    if (snap.pack2.ctx_key.empty()) snap.pack2.ctx_key = "shop_pack1";
    if (snap.tag_small.ctx_key.empty()) snap.tag_small.ctx_key = "Tag_small";
    if (snap.tag_big.ctx_key.empty()) snap.tag_big.ctx_key = "Tag_big";
    
    return snap;
}

// Build device buffer from snapshot
static std::vector<uint8_t> build_device_buffer(const PoolSnapshot& snap) {
    std::vector<uint8_t> buffer;
    
    // Reserve space for header
    DevicePools header;
    memset(&header, 0, sizeof(header));
    
    // Collect all contexts in order
    const PoolContext* contexts[5] = {
        &snap.voucher, &snap.pack1, &snap.pack2, 
        &snap.tag_small, &snap.tag_big
    };
    
    // Validate pools are non-empty
    for (int i = 0; i < 5; ++i) {
        const auto& ctx = *contexts[i];
        if (ctx.items.empty()) {
            fprintf(stderr, "[Pool] ERROR: context %d has empty pool\n", i);
            // Return empty buffer to signal error
            return buffer;
        }
    }
    
    // Calculate offsets
    size_t prefix_offset = 0;
    size_t weights_offset = 0;
    
    // First pass: calculate sizes
    for (int i = 0; i < 5; i++) {
        const auto& ctx = *contexts[i];
        header.ctx[i].prefix_len = static_cast<uint32_t>(ctx.ctx_key.size());
        header.ctx[i].pool_len = static_cast<uint32_t>(ctx.items.size());
        header.ctx[i].weighted = ctx.weighted ? 1 : 0;
    }
    
    // Build prefixes blob
    std::vector<uint8_t> prefixes_blob;
    for (int i = 0; i < 5; i++) {
        const auto& ctx = *contexts[i];
        header.ctx[i].prefix_off = static_cast<uint32_t>(prefixes_blob.size());
        prefixes_blob.insert(prefixes_blob.end(), 
                           ctx.ctx_key.begin(), 
                           ctx.ctx_key.end());
    }
    
    // Pad prefixes to 8-byte alignment
    while (prefixes_blob.size() % 8 != 0) {
        prefixes_blob.push_back(0);
    }
    header.prefixes_size = static_cast<uint32_t>(prefixes_blob.size());
    
    // Build weights blob (prefix sums)
    std::vector<uint64_t> weights_blob;
    for (int i = 0; i < 5; i++) {
        const auto& ctx = *contexts[i];
        if (ctx.weighted && !ctx.weights.empty()) {
            header.ctx[i].pool_off = static_cast<uint32_t>(weights_blob.size());
            
            // Compute prefix sums
            uint64_t sum = 0;
            for (auto w : ctx.weights) {
                sum += w;
                weights_blob.push_back(sum);
            }
            
            // Validate total <= 2^53
            if (sum > MAX_WEIGHT_TOTAL) {
                // Scale down by GCD
                uint64_t g = gcd_array(ctx.weights);
                if (g > 1) {
                    weights_blob.resize(weights_blob.size() - ctx.weights.size());
                    sum = 0;
                    for (auto w : ctx.weights) {
                        sum += w / g;
                        weights_blob.push_back(sum);
                    }
                }
                
                // Re-check
                if (sum > MAX_WEIGHT_TOTAL) {
                    fprintf(stderr, "[Pool] ERROR: weight total for context %d exceeds 2^53 even after GCD scaling\n", i);
                }
            }
        } else {
            header.ctx[i].pool_off = 0;  // Uniform pool
        }
    }
    header.weights_count = static_cast<uint32_t>(weights_blob.size());
    
    // Assemble final buffer
    buffer.resize(sizeof(DevicePools) + prefixes_blob.size() + 
                  weights_blob.size() * sizeof(uint64_t));
    
    // Copy header
    memcpy(buffer.data(), &header, sizeof(header));
    
    // Copy prefixes
    memcpy(buffer.data() + sizeof(header), 
           prefixes_blob.data(), 
           prefixes_blob.size());
    
    // Copy weights
    memcpy(buffer.data() + sizeof(header) + prefixes_blob.size(),
           weights_blob.data(),
           weights_blob.size() * sizeof(uint64_t));
    
    return buffer;
}

// External FFI function
extern "C" __declspec(dllexport)
void brainstorm_update_pools(const char* json_utf8) {
    if (!json_utf8) return;
    
    std::lock_guard<std::mutex> lock(PoolManager::update_mutex);
    
    try {
        // Parse JSON
        PoolSnapshot snap = parse_pool_json(json_utf8);
        
        // Build device buffer
        std::vector<uint8_t> buffer = build_device_buffer(snap);
        
        // Update host cache
        PoolManager::host_buffer = std::move(buffer);
        PoolManager::buffer_size.store(PoolManager::host_buffer.size(), std::memory_order_release);
        
        // Update host item cache for filter resolution
        {
            std::lock_guard<std::mutex> items_lock(PoolManager::items_mutex);
            PoolManager::host_items[0] = snap.voucher.items;
            PoolManager::host_items[1] = snap.pack1.items;
            PoolManager::host_items[2] = snap.pack2.items;
            PoolManager::host_items[3] = snap.tag_small.items;
            PoolManager::host_items[4] = snap.tag_big.items;
            
            // Sanity check: tag_small and tag_big order should be identical
            bool same_tags = (PoolManager::host_items[3].size() == PoolManager::host_items[4].size());
            if (same_tags) {
                for (size_t i = 0; i < PoolManager::host_items[3].size(); ++i) {
                    if (PoolManager::host_items[3][i] != PoolManager::host_items[4][i]) { 
                        same_tags = false; 
                        break;
                    }
                }
            }
            if (!same_tags) {
                // Log a warning; for RC1, we proceed but document that tag filters may be imprecise
                fprintf(stderr, "[Pool] WARNING: Tag_small and Tag_big pools differ in order; tag filters may be inaccurate.\n");
            }
        }
        
        // Check if CUDA is available
        void* cuda_ctx = get_cuda_context();
        PoolManager::cuda_available.store((cuda_ctx != nullptr), std::memory_order_relaxed);
        
        if (PoolManager::cuda_available && PoolManager::buffer_size > 0) {
            // Determine inactive buffer index
            int inactive_idx = 1 - PoolManager::active_pool_idx.load();
            
            // Free old inactive buffer if exists
            if (PoolManager::device_pools[inactive_idx]) {
                cuda_free_device_mem(PoolManager::device_pools[inactive_idx]);
                PoolManager::device_pools[inactive_idx] = nullptr;
            }
            
            // Allocate new device buffer
            PoolManager::device_pools[inactive_idx] = cuda_alloc_device_mem(PoolManager::buffer_size);
            
            if (PoolManager::device_pools[inactive_idx]) {
                // Copy data to device
                bool success = cuda_copy_to_device(
                    PoolManager::device_pools[inactive_idx],
                    PoolManager::host_buffer.data(),
                    PoolManager::buffer_size
                );
                
                if (success) {
                    // Atomically swap active buffer
                    PoolManager::active_pool_idx.store(inactive_idx, std::memory_order_release);
                    
                    // Log success
                    FILE* log = fopen("C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm\\pool_update.log", "a");
                    if (log) {
                        fprintf(log, "[Pool] Updated pools: %zu bytes, %zu vouchers, %zu packs, %zu tags\n",
                                PoolManager::buffer_size.load(),
                                snap.voucher.items.size(),
                                snap.pack1.items.size(),
                                snap.tag_small.items.size());
                        fclose(log);
                    }
                } else {
                    fprintf(stderr, "Error: Failed to copy pools to device\n");
                }
            } else {
                fprintf(stderr, "Error: Failed to allocate device memory for pools\n");
            }
        }
        
        PoolManager::initialized.store(true, std::memory_order_release);
        
    } catch (const std::exception& e) {
        fprintf(stderr, "Error updating pools: %s\n", e.what());
    }
}

// Get current pools for kernel
extern "C" const void* get_device_pools() {
    if (!PoolManager::initialized.load(std::memory_order_acquire)) return nullptr;
    
    // Return device pointer if CUDA available, otherwise host pointer
    if (PoolManager::cuda_available.load(std::memory_order_relaxed)) {
        int active_idx = PoolManager::active_pool_idx.load(std::memory_order_acquire);
        return PoolManager::device_pools[active_idx];
    }
    
    return PoolManager::host_buffer.data();  // CPU fallback readers only
}

// Get buffer size
extern "C" size_t get_pools_size() {
    return PoolManager::buffer_size.load(std::memory_order_acquire);
}

// Cleanup function for DLL unload
extern "C" void cleanup_pools() {
    std::lock_guard<std::mutex> lock(PoolManager::update_mutex);
    
    // Free device memory
    for (int i = 0; i < 2; i++) {
        if (PoolManager::device_pools[i]) {
            cuda_free_device_mem(PoolManager::device_pools[i]);
            PoolManager::device_pools[i] = nullptr;
        }
    }
    
    PoolManager::initialized.store(false, std::memory_order_release);
    PoolManager::cuda_available.store(false, std::memory_order_release);
}