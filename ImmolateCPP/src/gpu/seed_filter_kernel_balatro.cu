// Balatro-accurate CUDA kernel for GPU-accelerated seed filtering
// Implements exact pseudohash/pseudoseed algorithm from the game

#include <stdint.h>
#include "pool_types.h"

// Filter parameters structure (must match host)
struct FilterParams {
    uint32_t tag1_small;     // Tag1 index in tag_small pool
    uint32_t tag1_big;       // Tag1 index in tag_big pool
    uint32_t tag2_small;     // Tag2 index in tag_small pool
    uint32_t tag2_big;       // Tag2 index in tag_big pool
    uint32_t voucher;        // Voucher index in voucher pool
    uint32_t pack1;          // Pack index in pack1 pool
    uint32_t pack2;          // Pack index in pack2 pool
    uint32_t require_souls;
    uint32_t require_observatory;
    uint32_t require_perkeo;
};

// Constants from Balatro
__device__ const double PI = 3.14159265358979323846;
__device__ const double HASH_MULT = 1.1239285023;
__device__ const double LCG_A = 2.134453429141;
__device__ const double LCG_B = 1.72431234;

// Balatro's pseudohash function (GPU version)
__device__ double pseudohash_gpu(const char* str, int len) {
    double num = 1.0;
    for (int i = len - 1; i >= 0; i--) {
        double byte_val = (double)((unsigned char)str[i]);
        num = fmod((HASH_MULT / num) * byte_val * PI + PI * (double)(i + 1), 1.0);
    }
    return num;
}

// Simplified RNG state for GPU (single-use per seed)
struct BalatroRNGState {
    double hashed_seed;
    double voucher_state;
    double pack1_state;
    double pack2_state;
    double tag_small_state;
    double tag_big_state;
};

// Reverse-iterate two segments without building a temporary buffer:
// string = prefix[0..prefix_len-1] || seed[0..7]
__device__ double pseudohash_two_segments(const uint8_t* prefix,
                                          uint32_t prefix_len,
                                          const char seed8[8]) {
    double num = 1.0;
    int L = (int)prefix_len + 8;
    int k = L;
    // Process seed tail-to-head (positions L..prefix_len+1)
    for (int i = 7; i >= 0; --i, --k) {
        double byte_val = (double)((unsigned char)seed8[i]);
        num = fmod((HASH_MULT / num) * byte_val * PI + PI * (double)k, 1.0);
    }
    // Process prefix tail-to-head (positions prefix_len..1)
    for (int i = (int)prefix_len - 1; i >= 0; --i, --k) {
        double byte_val = (double)((unsigned char)prefix[i]);
        num = fmod((HASH_MULT / num) * byte_val * PI + PI * (double)k, 1.0);
    }
    return num;
}

// Initialize RNG state for a seed using dynamic pools
__device__ void init_rng_state_dynamic(BalatroRNGState* state, const char seed[8], const DevicePools* pools) {
    // Hash the base seed
    state->hashed_seed = pseudohash_gpu(seed, 8);
    
    // Get base pointers to blobs using header metadata
    const uint8_t* pools_bytes = reinterpret_cast<const uint8_t*>(pools);
    const uint8_t* prefixes = pools_bytes + sizeof(DevicePools);
    
    // Initialize each context using dynamic pool data with two-segment hash
    // Context 0: Voucher
    state->voucher_state = pseudohash_two_segments(
        prefixes + pools->ctx[0].prefix_off, 
        pools->ctx[0].prefix_len,
        seed);
    
    // Context 1: Pack1
    state->pack1_state = pseudohash_two_segments(
        prefixes + pools->ctx[1].prefix_off,
        pools->ctx[1].prefix_len,
        seed);
    
    // Context 2: Pack2 (usually same as Pack1)
    state->pack2_state = pseudohash_two_segments(
        prefixes + pools->ctx[2].prefix_off,
        pools->ctx[2].prefix_len,
        seed);
    
    // Context 3: Tag small
    state->tag_small_state = pseudohash_two_segments(
        prefixes + pools->ctx[3].prefix_off,
        pools->ctx[3].prefix_len,
        seed);
    
    // Context 4: Tag big
    state->tag_big_state = pseudohash_two_segments(
        prefixes + pools->ctx[4].prefix_off,
        pools->ctx[4].prefix_len,
        seed);
}

// Fallback for when pools not available
__device__ void init_rng_state(BalatroRNGState* state, const char seed[8]) {
    // Hash the base seed
    state->hashed_seed = pseudohash_gpu(seed, 8);
    
    // Pre-initialize states for each context (hardcoded)
    // Voucher context
    char voucher_ctx[16] = "Voucher";
    for (int i = 0; i < 8; i++) voucher_ctx[7 + i] = seed[i];
    voucher_ctx[15] = '\0';
    state->voucher_state = pseudohash_gpu(voucher_ctx, 15);
    
    // Pack contexts (shop_pack1)
    char pack1_ctx[17] = "shop_pack1";
    for (int i = 0; i < 8; i++) pack1_ctx[10 + i] = seed[i];
    pack1_ctx[18] = '\0';
    state->pack1_state = pseudohash_gpu(pack1_ctx, 18);
    
    // Second pack (shop_pack1 again for second call)
    state->pack2_state = state->pack1_state; // Will be updated with LCG
    
    // Tag contexts
    char tag_small_ctx[17] = "Tag_small";
    for (int i = 0; i < 8; i++) tag_small_ctx[9 + i] = seed[i];
    tag_small_ctx[17] = '\0';
    state->tag_small_state = pseudohash_gpu(tag_small_ctx, 17);
    
    char tag_big_ctx[15] = "Tag_big";
    for (int i = 0; i < 8; i++) tag_big_ctx[7 + i] = seed[i];
    tag_big_ctx[15] = '\0';
    state->tag_big_state = pseudohash_gpu(tag_big_ctx, 15);
}

// Get next value from pseudoseed (updates state)
__device__ double pseudoseed_next(double* state) {
    // LCG update
    *state = fmod(LCG_A + (*state) * LCG_B, 1.0);
    *state = fabs(*state);
    
    // Return HALF of the stored LCG value (critical discovery)
    return (*state) / 2.0;
}

// Uniform chooser for pool selection (0-based)
__device__ __forceinline__ uint32_t choose_uniform(double r, uint32_t n) {
    unsigned long long m = static_cast<unsigned long long>(r * static_cast<double>(n));
    if (m >= n) m = n - 1;
    return static_cast<uint32_t>(m);
}

// Weighted chooser for pool selection
__device__ __forceinline__ uint32_t choose_weighted(double r, const uint64_t* pref, uint32_t n) {
    if (n == 0) return 0;
    
    unsigned long long total = pref[n - 1];
    long double t = static_cast<long double>(r) * static_cast<long double>(total);
    
    uint32_t lo = 0, hi = n - 1, ans = hi;
    while (lo <= hi) {
        uint32_t mid = (lo + hi) >> 1;
        if (t < static_cast<long double>(pref[mid])) {
            ans = mid;
            if (mid == 0) break;
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    return ans;
}

// Base-36 odometer increment (handles 0-9, A-Z)
__device__ __forceinline__ void inc_base36(char s[8]) {
    #pragma unroll
    for (int i = 7; i >= 0; --i) {
        if (s[i] >= '0' && s[i] < '9') {
            s[i]++;
            break;
        } else if (s[i] == '9') {
            s[i] = 'A';
            break;
        } else if (s[i] >= 'A' && s[i] < 'Z') {
            s[i]++;
            break;
        } else if (s[i] == 'Z') {
            s[i] = '0';
            // Continue to next digit
        }
    }
}

// Convert numeric index to base-36 string
__device__ void idx_to_chars(uint64_t idx, char s[8]) {
    const char BASE36[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    #pragma unroll
    for (int i = 7; i >= 0; --i) {
        s[i] = BASE36[idx % 36];
        idx /= 36;
    }
}

// Main kernel using Balatro's actual RNG with dynamic pools and candidates
extern "C" __global__ void find_seeds_kernel_balatro(
    uint64_t start_seed_index,
    uint32_t total_seeds,
    uint32_t chunk_size,
    const FilterParams* params,
    const DevicePools* pools,     // Dynamic pool data (optional)
    uint64_t* candidates,         // NEW: Candidate output buffer
    uint32_t cap,                 // NEW: Capacity
    uint32_t* cand_count,         // NEW: Count
    volatile int* found           // Single-result flag for early stop
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;
    const uint32_t total_chunks = (total_seeds + chunk_size - 1) / chunk_size;
    
    // Check found flag every N iterations to reduce contention
    const int CHECK_EVERY = 16;  // Check every 16 seeds
    int local_counter = 0;
    
    // Process chunks in grid-stride pattern
    for (uint32_t chunk = tid; chunk < total_chunks; chunk += stride) {
        // Early exit check (reduced frequency)
        if ((local_counter & (CHECK_EVERY - 1)) == 0 && *found) return;
        
        uint32_t chunk_start = chunk * chunk_size;
        uint32_t chunk_end = min(chunk_start + chunk_size, total_seeds);
        
        // Initialize seed for this chunk
        char seed[8];
        idx_to_chars(start_seed_index + chunk_start, seed);
        
        // Process seeds in chunk
        for (uint32_t i = chunk_start; i < chunk_end; i++) {
            // Periodic found check
            if ((++local_counter & (CHECK_EVERY - 1)) == 0 && *found) return;
            // Initialize RNG state for this seed
            BalatroRNGState rng_state;
            if (pools) {
                init_rng_state_dynamic(&rng_state, seed, pools);
                
                // Safety check for empty pools
                if (pools->ctx[0].pool_len == 0 || pools->ctx[1].pool_len == 0 || 
                    pools->ctx[2].pool_len == 0 || pools->ctx[3].pool_len == 0 || 
                    pools->ctx[4].pool_len == 0) {
                    // Skip this seed if any pool is empty
                    inc_base36(seed);
                    continue;
                }
            } else {
                init_rng_state(&rng_state, seed);
                // When pools is null, we can't access weights
                // This path uses hardcoded indices only
            }
            
            // Generate voucher
            double voucher_val = pseudoseed_next(&rng_state.voucher_state);
            uint32_t voucher_id;
            if (pools) {
                const uint8_t* pools_bytes = reinterpret_cast<const uint8_t*>(pools);
                const uint64_t* weights = reinterpret_cast<const uint64_t*>(pools_bytes + sizeof(DevicePools) + pools->prefixes_size);
                voucher_id = pools->ctx[0].weighted ?
                    choose_weighted(voucher_val, weights + pools->ctx[0].pool_off, pools->ctx[0].pool_len) :
                    choose_uniform(voucher_val, pools->ctx[0].pool_len);
            } else {
                // Hardcoded pool sizes when no dynamic pools
                voucher_id = choose_uniform(voucher_val, 30); // Approximate voucher count
            }
            
            // Generate RNG values
            double pack1_val = pseudoseed_next(&rng_state.pack1_state);
            double pack2_val = pseudoseed_next(&rng_state.pack2_state);
            double tag_small_val = pseudoseed_next(&rng_state.tag_small_state);
            double tag_big_val = pseudoseed_next(&rng_state.tag_big_state);
            
            uint32_t pack1_id, pack2_id, small_tag, big_tag;
            
            if (pools) {
                // Dynamic pools available - use weights from pools data
                const uint8_t* pools_bytes = reinterpret_cast<const uint8_t*>(pools);
                const uint64_t* weights = reinterpret_cast<const uint64_t*>(pools_bytes + sizeof(DevicePools) + pools->prefixes_size);
                
                pack1_id = pools->ctx[1].weighted ?
                    choose_weighted(pack1_val, weights + pools->ctx[1].pool_off, pools->ctx[1].pool_len) :
                    choose_uniform(pack1_val, pools->ctx[1].pool_len);
                    
                pack2_id = pools->ctx[2].weighted ?
                    choose_weighted(pack2_val, weights + pools->ctx[2].pool_off, pools->ctx[2].pool_len) :
                    choose_uniform(pack2_val, pools->ctx[2].pool_len);
                    
                small_tag = pools->ctx[3].weighted ?
                    choose_weighted(tag_small_val, weights + pools->ctx[3].pool_off, pools->ctx[3].pool_len) :
                    choose_uniform(tag_small_val, pools->ctx[3].pool_len);
                    
                big_tag = pools->ctx[4].weighted ?
                    choose_weighted(tag_big_val, weights + pools->ctx[4].pool_off, pools->ctx[4].pool_len) :
                    choose_uniform(tag_big_val, pools->ctx[4].pool_len);
            } else {
                // No dynamic pools - use hardcoded pool sizes
                pack1_id = choose_uniform(pack1_val, 15);  // Approximate pack count
                pack2_id = choose_uniform(pack2_val, 15);  
                small_tag = choose_uniform(tag_small_val, 30);  // Approximate tag count
                big_tag = choose_uniform(tag_big_val, 30);
            }
            
            // Check filters
            bool match = true;
            
            // Tag filter (per-context indices; any placement qualifies)
            if (match) {
                bool want_t1 = (params->tag1_small != 0xFFFFFFFF) || (params->tag1_big != 0xFFFFFFFF);
                bool want_t2 = (params->tag2_small != 0xFFFFFFFF) || (params->tag2_big != 0xFFFFFFFF);
                
                if (want_t1) {
                    bool has_t1 = ((params->tag1_small != 0xFFFFFFFF) && (small_tag == params->tag1_small)) ||
                                  ((params->tag1_big   != 0xFFFFFFFF) && (big_tag   == params->tag1_big));
                    if (!has_t1) match = false;
                }
                
                if (match && want_t2) {
                    bool has_t2 = ((params->tag2_small != 0xFFFFFFFF) && (small_tag == params->tag2_small)) ||
                                  ((params->tag2_big   != 0xFFFFFFFF) && (big_tag   == params->tag2_big));
                    if (!has_t2) match = false;
                }
            }
            
            // Voucher filter
            if (match && params->voucher != 0xFFFFFFFF) {
                match = (voucher_id == params->voucher);
            }
            
            // Pack filter (OR semantics across slots)
            if (match) {
                bool want_p1 = (params->pack1 != 0xFFFFFFFF);
                bool want_p2 = (params->pack2 != 0xFFFFFFFF);
                if (want_p1 && want_p2) {
                    // Either slot may satisfy its respective index
                    match = (pack1_id == params->pack1) || (pack2_id == params->pack2);
                } else if (want_p1) {
                    match = (pack1_id == params->pack1);
                } else if (want_p2) {
                    match = (pack2_id == params->pack2);
                }
            }
            
            // If match found, write to candidate buffer
            if (match) {
                uint32_t idx = atomicAdd(cand_count, 1u);
                // Use modulo for ring buffer (per DE instructions)
                uint32_t pos = (cap > 0) ? (idx % cap) : 0;
                if (pos < cap) {  // Defensive check
                    candidates[pos] = start_seed_index + i;
                }
                // Optionally set found flag for early exit on first match
                if (idx == 0) {
                    atomicCAS((int*)found, 0, 1);
                }
            }
            
            // Increment seed using odometer
            inc_base36(seed);
        }
    }
}