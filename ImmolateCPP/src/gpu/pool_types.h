/*
 * Pool Types for Dynamic Shop Generation
 * Shared between host and device
 */

#ifndef POOL_TYPES_H
#define POOL_TYPES_H

#include <cstdint>

// Doubles have 53 bits of integer precision
#ifndef MAX_WEIGHT_TOTAL
#define MAX_WEIGHT_TOTAL (9007199254740992ULL) /* 2^53 */
#endif

#pragma pack(push, 1)

// Context specification for one pool (voucher, pack, tag, etc.)
typedef struct ContextSpec {
    uint32_t prefix_off;  // byte offset into prefixes blob
    uint32_t prefix_len;  // number of bytes in context key
    uint32_t pool_off;    // offset (in uint64 units) into weights blob; 0 if uniform
    uint32_t pool_len;    // number of items in pool
    uint32_t weighted;    // 0=uniform, 1=weighted
} ContextSpec;

// Device pools header followed by variable-length blobs
typedef struct DevicePools {
    // Fixed 5 contexts in order:
    // [0]=voucher, [1]=pack1, [2]=pack2, [3]=tag_small, [4]=tag_big
    ContextSpec ctx[5];
    
    // Metadata for computing blob addresses on device
    uint32_t prefixes_size;  // total size of prefixes blob in bytes (already 8-byte padded)
    uint32_t weights_count;  // number of uint64 entries in weights blob
    uint32_t reserved[2];    // future use / alignment
    
    // Followed by blobs (not in struct, but in memory layout):
    // uint8_t  prefixes_bytes[prefixes_size];
    // uint64_t weights_prefix_sums[weights_count];
} DevicePools;

#pragma pack(pop)

#ifdef __cplusplus
#include <type_traits>
static_assert(sizeof(ContextSpec) == 20, "ContextSpec size mismatch (packing required)");
static_assert(sizeof(DevicePools) == (5*20 + 16), "DevicePools size mismatch (packing required)");
#endif

// Constants for pool management
constexpr size_t MAX_PREFIX_LENGTH = 64;  // Safety limit for context keys
constexpr size_t MAX_POOL_ITEMS = 256;    // Safety limit per pool

#endif // POOL_TYPES_H