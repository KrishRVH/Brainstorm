// Probe kernel to verify argument passing - per Distinguished Engineer's instructions
#include <stdint.h>

// Probe kernel that records what the device actually sees
extern "C" __global__ void probe_args_kernel(
    uint64_t start_seed_index,   // 0
    uint32_t total_seeds,        // 1
    uint32_t chunk_size,         // 2
    const void* params,          // 3
    const void* pools,           // 4
    uint64_t* candidates,        // 5
    uint32_t cap,                // 6
    uint32_t* cand_count,        // 7
    volatile int* found,         // 8
    uint64_t* dbg                // 9 (debug buffer)
) {
    // Only thread 0 writes
    if (threadIdx.x | blockIdx.x) return;

    // Record raw values as seen by device
    dbg[0] = start_seed_index;
    dbg[1] = (uint64_t)total_seeds;
    dbg[2] = (uint64_t)chunk_size;
    dbg[3] = (uint64_t)params;
    dbg[4] = (uint64_t)pools;
    dbg[5] = (uint64_t)candidates;
    dbg[6] = (uint64_t)cap;
    dbg[7] = (uint64_t)cand_count;
    dbg[8] = (uint64_t)found;

    // Nullness/alignment guards
    auto is_aligned = [](uint64_t p, uint64_t a) { return (p & (a - 1)) == 0; };
    dbg[9]  = (params     && is_aligned((uint64_t)params,     8)) ? 1 : (params ? 0xBAD00001ull : 0);
    dbg[10] = (candidates && is_aligned((uint64_t)candidates, 8)) ? 1 : (candidates ? 0xBAD00002ull : 0);
    dbg[11] = (cand_count && is_aligned((uint64_t)cand_count, 4)) ? 1 : (cand_count ? 0xBAD00003ull : 0);
    dbg[12] = (found      && is_aligned((uint64_t)found,      4)) ? 1 : (found ? 0xBAD00004ull : 0);

    // Safe test writes if non-null
    if (cand_count) { *cand_count = 0; }            
    if (found)      { *(int*)found = 0; }           
    if (candidates) { candidates[0] = 0xCAFEBABEDEADBEEF; }
    
    // Mark completion
    dbg[13] = 0xC0FFEE;
}

// Single struct argument version for robust passing
struct __align__(8) KernelArgs {
    uint64_t start_seed_index;
    uint32_t total_seeds;
    uint32_t chunk_size;
    const void* params;
    const void* pools;
    uint64_t* candidates;
    uint32_t cap;
    uint32_t* cand_count;
    volatile int* found;
};

// Probe kernel using single struct
extern "C" __global__ void probe_args_struct(KernelArgs a, uint64_t* dbg) {
    if (threadIdx.x | blockIdx.x) return;
    
    // Record values from struct
    dbg[0] = a.start_seed_index;
    dbg[1] = (uint64_t)a.total_seeds;
    dbg[2] = (uint64_t)a.chunk_size;
    dbg[3] = (uint64_t)a.params;
    dbg[4] = (uint64_t)a.pools;
    dbg[5] = (uint64_t)a.candidates;
    dbg[6] = (uint64_t)a.cap;
    dbg[7] = (uint64_t)a.cand_count;
    dbg[8] = (uint64_t)a.found;
    
    // Alignment checks
    auto is_aligned = [](uint64_t p, uint64_t a) { return (p & (a - 1)) == 0; };
    dbg[9]  = (a.params     && is_aligned((uint64_t)a.params,     8)) ? 1 : (a.params ? 0xBAD00001ull : 0);
    dbg[10] = (a.candidates && is_aligned((uint64_t)a.candidates, 8)) ? 1 : (a.candidates ? 0xBAD00002ull : 0);
    dbg[11] = (a.cand_count && is_aligned((uint64_t)a.cand_count, 4)) ? 1 : (a.cand_count ? 0xBAD00003ull : 0);
    dbg[12] = (a.found      && is_aligned((uint64_t)a.found,      4)) ? 1 : (a.found ? 0xBAD00004ull : 0);
    
    // Safe writes
    if (a.cand_count) *a.cand_count = 0;
    if (a.found) *(int*)a.found = 0;
    if (a.candidates && a.cap > 0) a.candidates[0] = 0xCAFEBABEDEADBEEF;
    
    dbg[13] = 0xC0FFEE;
}