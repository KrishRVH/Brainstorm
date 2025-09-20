/*
 * Discrete Choosers for Balatro Pool Selection
 * Exact implementation of the game's selection logic
 */

#ifndef CHOOSERS_HPP
#define CHOOSERS_HPP

#include <cstdint>

// CPU versions

// Uniform chooser (0-based index)
// r in [0,1); returns idx in [0, n-1]
inline uint32_t choose_uniform(double r, uint32_t n) {
    // r in [0,1); idx in [0, n-1]
    uint64_t m = static_cast<uint64_t>(r * static_cast<double>(n));
    if (m >= n) m = n - 1; // defensive
    return static_cast<uint32_t>(m);
}

// Weighted chooser (strict inequality)
// pref is monotonically increasing prefix sums; total = pref[n-1]
inline uint32_t choose_weighted(double r, const uint64_t* pref, uint32_t n) {
    if (n == 0) return 0; // defensive
    
    uint64_t total = pref[n - 1];
    long double t = static_cast<long double>(r) * static_cast<long double>(total);
    
    // Find first i with t < pref[i]
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

// CUDA versions
#ifdef __CUDA_ARCH__

__device__ __forceinline__ uint32_t choose_uniform_gpu(double r, uint32_t n) {
    unsigned long long m = static_cast<unsigned long long>(r * static_cast<double>(n));
    if (m >= n) m = n - 1;
    return static_cast<uint32_t>(m);
}

__device__ __forceinline__ uint32_t choose_weighted_gpu(double r, const unsigned long long* pref, uint32_t n) {
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

#endif // __CUDA_ARCH__

#endif // CHOOSERS_HPP