#pragma once
#include <cstdint>

// V2: Resolve UI filter names to per-context indices (separate small/big tags)
extern "C" bool brainstorm_resolve_filter_indices_v2(
    const char* voucher_key,
    const char* pack_key,
    const char* tag1_key,
    const char* tag2_key,
    // outputs:
    uint32_t* out_voucher_idx,
    uint32_t* out_pack1_idx,
    uint32_t* out_pack2_idx,
    uint32_t* out_tag1_small_idx,
    uint32_t* out_tag1_big_idx,
    uint32_t* out_tag2_small_idx,
    uint32_t* out_tag2_big_idx
);