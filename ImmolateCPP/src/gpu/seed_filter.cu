// CUDA kernel for parallel seed filtering
// Each thread tests one seed independently

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>

// Constants matching Balatro's internals
#define NUM_TAGS 27
#define NUM_VOUCHERS 32
#define NUM_PACKS 15
#define RETRY_VALUE 0xFFFFFFFF

// Filter parameters structure (must match host)
struct FilterParams {
    uint32_t tag1;
    uint32_t tag2;
    uint32_t voucher;
    uint32_t pack;
    bool require_souls;
    bool require_observatory;
    bool require_perkeo;
};

// Balatro's PRNG implementation ported to CUDA
// Based on analysis of Balatro source (functions/misc_functions.lua)
__device__ double pseudohash_device(const char* str, int len) {
    double num = 1.0;
    // Iterate backwards through string (matching Lua implementation)
    for (int i = len - 1; i >= 0; i--) {
        // Balatro's formula: ((1.1239285023/num)*char*pi + pi*(i+1))%1
        num = fmod((1.1239285023 / num) * str[i] * 3.14159265359 + 3.14159265359 * (i + 1), 1.0);
    }
    return num;
}

__device__ uint32_t pseudorandom_device(uint64_t seed, uint32_t key_hash) {
    // Convert seed and key to Balatro's format
    char seed_str[9];
    for (int i = 7; i >= 0; i--) {
        seed_str[i] = (char)((seed >> (i * 8)) & 0xFF);
    }
    seed_str[8] = '\0';
    
    // Combine key hash with seed string
    char combined[32];
    int pos = 0;
    
    // Add key hash as string prefix
    for (int i = 0; i < 8; i++) {
        combined[pos++] = '0' + ((key_hash >> (i * 4)) & 0xF);
    }
    
    // Add seed string
    for (int i = 0; i < 8; i++) {
        combined[pos++] = seed_str[i];
    }
    
    // Apply Balatro's pseudohash
    double hash = pseudohash_device(combined, pos);
    
    // Apply Balatro's pseudoseed transformation
    hash = fabs(fmod(2.134453429141 + hash * 1.72431234, 1.0));
    
    // Convert to integer range for tag/item selection
    return (uint32_t)(hash * 0xFFFFFFFF);
}

// Hash a string key (simplified)
__device__ uint32_t hash_key(const char* key) {
    uint32_t hash = 5381;
    for (int i = 0; i < 20 && key[i]; i++) {
        hash = ((hash << 5) + hash) + key[i];
    }
    return hash;
}

// Get tag for specific blind
__device__ uint32_t get_tag(uint64_t seed, int ante, int blind) {
    // Construct key like "Tag_ante_1_blind_0"
    char key[32];
    int pos = 0;
    
    // "Tag_ante_"
    key[pos++] = 'T'; key[pos++] = 'a'; key[pos++] = 'g';
    key[pos++] = '_'; key[pos++] = 'a'; key[pos++] = 'n';
    key[pos++] = 't'; key[pos++] = 'e'; key[pos++] = '_';
    
    // Add ante number
    key[pos++] = '0' + ante;
    
    // "_blind_"
    key[pos++] = '_'; key[pos++] = 'b'; key[pos++] = 'l';
    key[pos++] = 'i'; key[pos++] = 'n'; key[pos++] = 'd';
    key[pos++] = '_';
    
    // Add blind number
    key[pos++] = '0' + blind;
    key[pos] = '\0';
    
    uint32_t key_hash = hash_key(key);
    return pseudorandom_device(seed, key_hash) % NUM_TAGS;
}

// Get first voucher
__device__ uint32_t get_voucher(uint64_t seed) {
    uint32_t key_hash = hash_key("Voucher_1");
    return pseudorandom_device(seed, key_hash) % NUM_VOUCHERS;
}

// Get first pack
__device__ uint32_t get_pack(uint64_t seed) {
    uint32_t key_hash = hash_key("shop_pack_1");
    return pseudorandom_device(seed, key_hash) % NUM_PACKS;
}

// Main kernel - each thread tests one seed
__global__ void find_seeds_kernel(
    uint64_t start_seed,
    uint32_t count,
    FilterParams params,
    uint64_t* result,
    volatile int* found
) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check and early exit if already found
    if (idx >= count || *found) return;
    
    // Calculate seed for this thread
    uint64_t seed = start_seed + idx;
    
    // Check tags first (cheapest operation)
    if (params.tag1 != RETRY_VALUE || params.tag2 != RETRY_VALUE) {
        uint32_t small_tag = get_tag(seed, 1, 0);  // Ante 1, Small Blind
        uint32_t big_tag = get_tag(seed, 1, 1);    // Ante 1, Big Blind
        
        bool tags_match = false;
        
        if (params.tag2 == RETRY_VALUE) {
            // Only checking tag1
            tags_match = (small_tag == params.tag1 || big_tag == params.tag1);
        }
        else if (params.tag1 == params.tag2) {
            // Same tag twice - both positions must match
            tags_match = (small_tag == params.tag1 && big_tag == params.tag1);
        }
        else {
            // Two different tags - both must be present (order doesn't matter)
            bool has_tag1 = (small_tag == params.tag1 || big_tag == params.tag1);
            bool has_tag2 = (small_tag == params.tag2 || big_tag == params.tag2);
            tags_match = has_tag1 && has_tag2;
        }
        
        if (!tags_match) return;
    }
    
    // Check voucher if specified
    if (params.voucher != RETRY_VALUE) {
        uint32_t first_voucher = get_voucher(seed);
        if (first_voucher != params.voucher) return;
    }
    
    // Check pack if specified
    if (params.pack != RETRY_VALUE) {
        uint32_t first_pack = get_pack(seed);
        if (first_pack != params.pack) return;
    }
    
    // Special conditions
    if (params.require_observatory) {
        // Check for Telescope voucher and Mega Celestial pack
        uint32_t first_voucher = get_voucher(seed);
        uint32_t first_pack = get_pack(seed);
        
        // IDs from Balatro's item definitions (verified from source)
        const uint32_t TELESCOPE_ID = 24;      // Telescope voucher
        const uint32_t MEGA_CELESTIAL_ID = 12; // Mega Celestial Pack
        
        if (first_voucher != TELESCOPE_ID || first_pack != MEGA_CELESTIAL_ID) {
            return;
        }
    }
    
    // Found a match! Use atomic operation to ensure only one thread wins
    if (atomicCAS((int*)found, 0, 1) == 0) {
        *result = seed;
    }
}

// Host-callable function to launch kernel
extern "C" void launch_seed_search(
    uint64_t start_seed,
    uint32_t count,
    FilterParams* d_params,
    uint64_t* d_result,
    int* d_found
) {
    // Calculate optimal launch configuration for RTX 4090
    // RTX 4090 has 128 SMs, optimal is usually 256-512 threads per block
    int threads_per_block = 256;
    int blocks = (count + threads_per_block - 1) / threads_per_block;
    
    // Limit blocks to avoid overwhelming the GPU
    const int MAX_BLOCKS = 65536;
    if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;
    
    // Launch kernel
    find_seeds_kernel<<<blocks, threads_per_block>>>(
        start_seed, count, *d_params, d_result, d_found
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[GPU] Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}