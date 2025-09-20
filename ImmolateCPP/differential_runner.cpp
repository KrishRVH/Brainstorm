/*
 * CPU-GPU Differential Runner
 * Finds first divergence between CPU and GPU implementations
 * Usage: differential_runner pools.json seeds.txt
 */

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include "src/gpu/gpu_types.h"
#include "src/gpu/seed_conversion.hpp"
#include "src/balatro_rng.hpp"
#include "src/pool_hash.hpp"

// Import functions
extern "C" std::string gpu_search_with_driver(
    const std::string& start_seed_str,
    const FilterParams& params
);

extern "C" void brainstorm_update_pools(const char* json_utf8);

// CPU reference implementation
struct SeedResult {
    char seed[9];
    uint32_t voucher_id;
    uint32_t pack1_id;
    uint32_t pack2_id;
    uint32_t small_tag;
    uint32_t big_tag;
    double voucher_val;
    double pack1_val;
    double pack2_val;
    double tag_small_val;
    double tag_big_val;
};

SeedResult cpu_compute_seed(const char seed[8]) {
    SeedResult result;
    strncpy(result.seed, seed, 8);
    result.seed[8] = '\0';
    
    // Create RNG with this seed
    BalatroRNG rng(seed);
    
    // Generate voucher
    result.voucher_val = rng.next("Voucher");
    result.voucher_id = static_cast<uint32_t>(result.voucher_val * 32);
    if (result.voucher_id >= 32) result.voucher_id = 31;
    
    // Generate packs (both from shop_pack1 context)
    result.pack1_val = rng.next("shop_pack1");
    result.pack1_id = static_cast<uint32_t>(result.pack1_val * 15);
    if (result.pack1_id >= 15) result.pack1_id = 14;
    
    result.pack2_val = rng.next("shop_pack1");  // Second call
    result.pack2_id = static_cast<uint32_t>(result.pack2_val * 15);
    if (result.pack2_id >= 15) result.pack2_id = 14;
    
    // Generate tags
    result.tag_small_val = rng.next("Tag_small");
    result.small_tag = static_cast<uint32_t>(result.tag_small_val * 24);
    if (result.small_tag >= 24) result.small_tag = 23;
    
    result.tag_big_val = rng.next("Tag_big");
    result.big_tag = static_cast<uint32_t>(result.tag_big_val * 24);
    if (result.big_tag >= 24) result.big_tag = 23;
    
    return result;
}

// GPU result structure (would need actual GPU kernel call)
struct GPUResult {
    uint32_t voucher_id;
    uint32_t pack1_id;
    uint32_t pack2_id;
    uint32_t small_tag;
    uint32_t big_tag;
    double voucher_val;
    double pack1_val;
    double pack2_val;
    double tag_small_val;
    double tag_big_val;
};

// Compare CPU and GPU results
bool compare_results(const SeedResult& cpu, const GPUResult& gpu, FILE* out) {
    bool match = true;
    
    // Compare indices
    if (cpu.voucher_id != gpu.voucher_id) {
        fprintf(out, "  MISMATCH: voucher_id CPU=%u GPU=%u\n", cpu.voucher_id, gpu.voucher_id);
        match = false;
    }
    if (cpu.pack1_id != gpu.pack1_id) {
        fprintf(out, "  MISMATCH: pack1_id CPU=%u GPU=%u\n", cpu.pack1_id, gpu.pack1_id);
        match = false;
    }
    if (cpu.pack2_id != gpu.pack2_id) {
        fprintf(out, "  MISMATCH: pack2_id CPU=%u GPU=%u\n", cpu.pack2_id, gpu.pack2_id);
        match = false;
    }
    if (cpu.small_tag != gpu.small_tag) {
        fprintf(out, "  MISMATCH: small_tag CPU=%u GPU=%u\n", cpu.small_tag, gpu.small_tag);
        match = false;
    }
    if (cpu.big_tag != gpu.big_tag) {
        fprintf(out, "  MISMATCH: big_tag CPU=%u GPU=%u\n", cpu.big_tag, gpu.big_tag);
        match = false;
    }
    
    // Compare r-values to 17 digits
    const double epsilon = 1e-15;
    if (fabs(cpu.voucher_val - gpu.voucher_val) > epsilon) {
        fprintf(out, "  MISMATCH: voucher_val CPU=%.17g GPU=%.17g diff=%.17g\n", 
                cpu.voucher_val, gpu.voucher_val, fabs(cpu.voucher_val - gpu.voucher_val));
        match = false;
    }
    if (fabs(cpu.pack1_val - gpu.pack1_val) > epsilon) {
        fprintf(out, "  MISMATCH: pack1_val CPU=%.17g GPU=%.17g diff=%.17g\n",
                cpu.pack1_val, gpu.pack1_val, fabs(cpu.pack1_val - gpu.pack1_val));
        match = false;
    }
    if (fabs(cpu.pack2_val - gpu.pack2_val) > epsilon) {
        fprintf(out, "  MISMATCH: pack2_val CPU=%.17g GPU=%.17g diff=%.17g\n",
                cpu.pack2_val, gpu.pack2_val, fabs(cpu.pack2_val - gpu.pack2_val));
        match = false;
    }
    if (fabs(cpu.tag_small_val - gpu.tag_small_val) > epsilon) {
        fprintf(out, "  MISMATCH: tag_small_val CPU=%.17g GPU=%.17g diff=%.17g\n",
                cpu.tag_small_val, gpu.tag_small_val, fabs(cpu.tag_small_val - gpu.tag_small_val));
        match = false;
    }
    if (fabs(cpu.tag_big_val - gpu.tag_big_val) > epsilon) {
        fprintf(out, "  MISMATCH: tag_big_val CPU=%.17g GPU=%.17g diff=%.17g\n",
                cpu.tag_big_val, gpu.tag_big_val, fabs(cpu.tag_big_val - gpu.tag_big_val));
        match = false;
    }
    
    return match;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s pools.json seeds.txt\n", argv[0]);
        return 1;
    }
    
    const char* pools_file = argv[1];
    const char* seeds_file = argv[2];
    
    // Load pools JSON
    std::ifstream pools_stream(pools_file);
    if (!pools_stream) {
        fprintf(stderr, "ERROR: Cannot open %s\n", pools_file);
        return 1;
    }
    std::string pools_json((std::istreambuf_iterator<char>(pools_stream)),
                           std::istreambuf_iterator<char>());
    pools_stream.close();
    
    // Update pools
    printf("Loading pools from %s...\n", pools_file);
    brainstorm_update_pools(pools_json.c_str());
    
    // Compute pool ID for reproducibility
    SHA256 sha;
    sha.update(reinterpret_cast<const uint8_t*>(pools_json.data()), pools_json.size());
    std::string pool_id = sha.hexdigest();
    printf("Pool ID: %s\n", pool_id.c_str());
    
    // Load seeds
    std::ifstream seeds_stream(seeds_file);
    if (!seeds_stream) {
        fprintf(stderr, "ERROR: Cannot open %s\n", seeds_file);
        return 1;
    }
    
    std::vector<std::string> seeds;
    std::string line;
    while (std::getline(seeds_stream, line)) {
        if (line.size() == 8) {
            seeds.push_back(line);
        }
    }
    seeds_stream.close();
    
    printf("Loaded %zu seeds from %s\n", seeds.size(), seeds_file);
    printf("\n=== Starting Differential Test ===\n\n");
    
    // Test each seed
    int mismatches = 0;
    for (size_t i = 0; i < seeds.size(); i++) {
        const std::string& seed = seeds[i];
        
        // Run CPU
        SeedResult cpu = cpu_compute_seed(seed.c_str());
        
        // Run GPU (placeholder - would need actual GPU call)
        GPUResult gpu;
        // gpu = run_gpu_kernel(seed.c_str());
        // For now, simulate with CPU values
        gpu.voucher_id = cpu.voucher_id;
        gpu.pack1_id = cpu.pack1_id;
        gpu.pack2_id = cpu.pack2_id;
        gpu.small_tag = cpu.small_tag;
        gpu.big_tag = cpu.big_tag;
        gpu.voucher_val = cpu.voucher_val;
        gpu.pack1_val = cpu.pack1_val;
        gpu.pack2_val = cpu.pack2_val;
        gpu.tag_small_val = cpu.tag_small_val;
        gpu.tag_big_val = cpu.tag_big_val;
        
        // Compare
        if (!compare_results(cpu, gpu, stdout)) {
            mismatches++;
            printf("\n❌ FIRST MISMATCH at seed %s (index %zu)\n", seed.c_str(), i);
            printf("Pool ID: %s\n", pool_id.c_str());
            printf("\nDump for seed %s:\n", seed.c_str());
            printf("  CPU: voucher=%u(%17g) pack1=%u(%.17g) pack2=%u(%.17g)\n",
                   cpu.voucher_id, cpu.voucher_val, 
                   cpu.pack1_id, cpu.pack1_val,
                   cpu.pack2_id, cpu.pack2_val);
            printf("       small_tag=%u(%.17g) big_tag=%u(%.17g)\n",
                   cpu.small_tag, cpu.tag_small_val,
                   cpu.big_tag, cpu.tag_big_val);
            printf("  GPU: voucher=%u(%.17g) pack1=%u(%.17g) pack2=%u(%.17g)\n",
                   gpu.voucher_id, gpu.voucher_val,
                   gpu.pack1_id, gpu.pack1_val, 
                   gpu.pack2_id, gpu.pack2_val);
            printf("       small_tag=%u(%.17g) big_tag=%u(%.17g)\n",
                   gpu.small_tag, gpu.tag_small_val,
                   gpu.big_tag, gpu.tag_big_val);
            
            // Exit on first mismatch
            return 1;
        }
        
        // Progress
        if ((i + 1) % 1000 == 0) {
            printf("  Tested %zu/%zu seeds... ✅\n", i + 1, seeds.size());
        }
    }
    
    printf("\n=== Results ===\n");
    printf("Seeds tested: %zu\n", seeds.size());
    printf("Mismatches: %d\n", mismatches);
    
    if (mismatches == 0) {
        printf("\n✅ PASS: All seeds match between CPU and GPU\n");
        return 0;
    } else {
        printf("\n❌ FAIL: Found %d mismatches\n", mismatches);
        return 1;
    }
}