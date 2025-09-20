#pragma once
#include <stdint.h>

#if defined(__CUDACC__)
#define GPU_HOST_DEVICE __host__ __device__
#else
#define GPU_HOST_DEVICE
#endif

// Force 4-byte fields to avoid ABI surprises between host and device
// All fields are uint32_t for consistent alignment and size
struct FilterParams {
    // Tag indices per context; 0xFFFFFFFF means "no filter"
    uint32_t tag1_small;     // Tag1 index in tag_small pool
    uint32_t tag1_big;       // Tag1 index in tag_big pool
    uint32_t tag2_small;     // Tag2 index in tag_small pool  
    uint32_t tag2_big;       // Tag2 index in tag_big pool
    uint32_t voucher;        // Voucher index in voucher pool or 0xFFFFFFFF for none
    uint32_t pack1;          // Pack index in pack1 pool or 0xFFFFFFFF for none
    uint32_t pack2;          // Pack index in pack2 pool or 0xFFFFFFFF for none
    uint32_t require_souls;  // 1 if souls required, 0 otherwise
    uint32_t require_observatory;  // 1 if observatory required, 0 otherwise
    uint32_t require_perkeo; // 1 if perkeo required, 0 otherwise
};

// Ensure consistent size across compilers
static_assert(sizeof(FilterParams) == 40, "FilterParams size mismatch");

// Debug statistics structure (optional)
struct DebugStats {
    uint64_t seeds_tested;
    uint64_t tag_matches;
    uint64_t tag_rejections;
    uint64_t voucher_matches;
    uint64_t voucher_rejections;
    uint64_t pack_matches;
    uint64_t pack_rejections;
    uint64_t observatory_matches;
    uint64_t observatory_rejections;
    uint64_t total_matches;
};

static_assert(sizeof(DebugStats) == 80, "DebugStats size mismatch");