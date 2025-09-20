/*
 * Diagnostics for In-Game Parity Testing
 * Provides detailed seed analysis for manual validation
 */

#ifndef DIAGNOSTICS_HPP
#define DIAGNOSTICS_HPP

#include <string>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include "gpu/gpu_types.h"
#include "gpu/seed_conversion.hpp"
#include "balatro_rng.hpp"
#include "pool_hash.hpp"

// Diagnostic result structure
struct SeedDiagnostics {
    char seed[9];
    
    // R-values (17 digits precision)
    double voucher_r;
    double pack1_r;
    double pack2_r;
    double tag_small_r;
    double tag_big_r;
    
    // Chosen indices
    uint32_t voucher_idx;
    uint32_t pack1_idx;
    uint32_t pack2_idx;
    uint32_t tag_small_idx;
    uint32_t tag_big_idx;
    
    // Item keys (if available)
    std::string voucher_key;
    std::string pack1_key;
    std::string pack2_key;
    std::string tag_small_key;
    std::string tag_big_key;
    
    // Pool metadata
    std::string pool_id;
    uint32_t pool_version;
    std::string ctx_keys[5];
};

// Generate diagnostics for a seed
extern "C" SeedDiagnostics generate_seed_diagnostics(const char* seed) {
    SeedDiagnostics diag;
    strncpy(diag.seed, seed, 8);
    diag.seed[8] = '\0';
    
    // Create RNG with this seed
    BalatroRNG rng(seed);
    
    // Generate r-values
    diag.voucher_r = rng.next("Voucher");
    diag.pack1_r = rng.next("shop_pack1");
    diag.pack2_r = rng.next("shop_pack1");  // Second call
    diag.tag_small_r = rng.next("Tag_small");
    diag.tag_big_r = rng.next("Tag_big");
    
    // Calculate indices (assuming standard pool sizes)
    // These would need actual pool sizes from PoolManager
    diag.voucher_idx = static_cast<uint32_t>(diag.voucher_r * 32);
    if (diag.voucher_idx >= 32) diag.voucher_idx = 31;
    
    diag.pack1_idx = static_cast<uint32_t>(diag.pack1_r * 15);
    if (diag.pack1_idx >= 15) diag.pack1_idx = 14;
    
    diag.pack2_idx = static_cast<uint32_t>(diag.pack2_r * 15);
    if (diag.pack2_idx >= 15) diag.pack2_idx = 14;
    
    diag.tag_small_idx = static_cast<uint32_t>(diag.tag_small_r * 24);
    if (diag.tag_small_idx >= 24) diag.tag_small_idx = 23;
    
    diag.tag_big_idx = static_cast<uint32_t>(diag.tag_big_r * 24);
    if (diag.tag_big_idx >= 24) diag.tag_big_idx = 23;
    
    // Pool metadata (would be filled from PoolManager)
    diag.pool_id = "pending_actual_pool_hash";
    diag.pool_version = 0;
    diag.ctx_keys[0] = "Voucher";
    diag.ctx_keys[1] = "shop_pack1";
    diag.ctx_keys[2] = "shop_pack1";
    diag.ctx_keys[3] = "Tag_small";
    diag.ctx_keys[4] = "Tag_big";
    
    return diag;
}

// Export diagnostics to JSON
extern "C" __declspec(dllexport)
const char* brainstorm_diag(const char* seed) {
    if (!seed || strlen(seed) != 8) {
        return strdup("{\"error\":\"Invalid seed (must be 8 chars)\"}");
    }
    
    SeedDiagnostics diag = generate_seed_diagnostics(seed);
    
    std::ostringstream json;
    json << std::fixed << std::setprecision(17);
    json << "{\n";
    json << "  \"seed\": \"" << diag.seed << "\",\n";
    json << "  \"pool_id\": \"" << diag.pool_id << "\",\n";
    json << "  \"pool_version\": " << diag.pool_version << ",\n";
    json << "  \"ctx_keys\": {\n";
    json << "    \"voucher\": \"" << diag.ctx_keys[0] << "\",\n";
    json << "    \"pack1\": \"" << diag.ctx_keys[1] << "\",\n";
    json << "    \"pack2\": \"" << diag.ctx_keys[2] << "\",\n";
    json << "    \"tag_small\": \"" << diag.ctx_keys[3] << "\",\n";
    json << "    \"tag_big\": \"" << diag.ctx_keys[4] << "\"\n";
    json << "  },\n";
    json << "  \"r_values\": {\n";
    json << "    \"voucher\": " << diag.voucher_r << ",\n";
    json << "    \"pack1\": " << diag.pack1_r << ",\n";
    json << "    \"pack2\": " << diag.pack2_r << ",\n";
    json << "    \"tag_small\": " << diag.tag_small_r << ",\n";
    json << "    \"tag_big\": " << diag.tag_big_r << "\n";
    json << "  },\n";
    json << "  \"indices\": {\n";
    json << "    \"voucher\": " << diag.voucher_idx << ",\n";
    json << "    \"pack1\": " << diag.pack1_idx << ",\n";
    json << "    \"pack2\": " << diag.pack2_idx << ",\n";
    json << "    \"tag_small\": " << diag.tag_small_idx << ",\n";
    json << "    \"tag_big\": " << diag.tag_big_idx << "\n";
    json << "  },\n";
    json << "  \"item_keys\": {\n";
    json << "    \"voucher\": \"" << diag.voucher_key << "\",\n";
    json << "    \"pack1\": \"" << diag.pack1_key << "\",\n";
    json << "    \"pack2\": \"" << diag.pack2_key << "\",\n";
    json << "    \"tag_small\": \"" << diag.tag_small_key << "\",\n";
    json << "    \"tag_big\": \"" << diag.tag_big_key << "\"\n";
    json << "  }\n";
    json << "}";
    
    return strdup(json.str().c_str());
}

#endif // DIAGNOSTICS_HPP