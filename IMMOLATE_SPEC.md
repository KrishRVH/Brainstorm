# Custom Immolate DLL Specification

## Overview
A high-performance native library for Balatro seed generation and validation, written in Rust or Zig for maximum performance.

## Current Limitations of Original Immolate.dll
1. **No Erratic Deck Support**: Doesn't understand Erratic deck's randomized face card and suit distributions
2. **Limited Validation**: Only checks vouchers, tags, and packs - not deck composition
3. **No Batch Processing**: Tests one seed at a time
4. **Black Box**: No visibility into what's happening internally

## Requirements for New Implementation

### Core Features

#### 1. Seed Generation Engine
- Generate deterministic seeds using Balatro's algorithm
- Support batch generation (test 100+ seeds without game restarts)
- Use same PRNG as Balatro (likely xoshiro256++ or similar)

#### 2. Deck Simulation
- **Normal Decks**: Standard 52-card deck generation
- **Erratic Decks**: Implement the randomization algorithm
  - Random face card count (varies widely)
  - Random suit distribution
  - Must match Balatro's exact algorithm

#### 3. Filter Validation
- **Voucher matching**: Check if seed generates specific vouchers
- **Tag matching**: Validate tag generation
- **Pack matching**: Check booster pack availability
- **Face card count**: Count Jack, Queen, King cards
- **Suit ratio**: Calculate top-2 suit percentage
- **Custom filters**: Extensible for future requirements

#### 4. Performance Optimizations
- **Parallel processing**: Use SIMD/multi-threading for batch validation
- **Memory efficiency**: Minimal allocations, stack-based where possible
- **Caching**: LRU cache for recently tested seeds
- **Early exit**: Stop testing once requirements fail

### API Design

```rust
// FFI-safe C interface
#[no_mangle]
pub extern "C" fn immolate_batch_test(
    start_seed: *const c_char,
    count: u32,
    filters: *const FilterConfig,
    results: *mut SeedResult,
) -> u32 {
    // Returns number of valid seeds found
}

#[repr(C)]
pub struct FilterConfig {
    // Voucher/Tag/Pack filters
    voucher_name: *const c_char,
    tag_name: *const c_char,
    pack_names: *const *const c_char,
    pack_count: u32,
    
    // Deck composition filters
    min_face_cards: u8,
    min_suit_ratio: f32,  // 0.0 to 1.0
    
    // Deck type
    is_erratic: bool,
    
    // Performance hints
    max_seeds_to_test: u32,
    early_exit: bool,
}

#[repr(C)]
pub struct SeedResult {
    seed: [c_char; 9],  // 8 chars + null terminator
    face_card_count: u8,
    suit_ratio: f32,
    matches_filters: bool,
    
    // Debug info
    spades: u8,
    hearts: u8,
    clubs: u8,
    diamonds: u8,
}
```

### Implementation Strategy

#### Phase 1: Research
1. **Reverse engineer Balatro's PRNG**: Determine exact algorithm
2. **Understand Erratic deck generation**: Map the randomization logic
3. **Profile current performance**: Establish baseline metrics

#### Phase 2: Core Implementation
1. **Rust/Zig scaffold**: Set up FFI-safe library structure
2. **PRNG implementation**: Match Balatro's seed generation
3. **Normal deck generation**: Implement standard logic
4. **Erratic deck generation**: Implement randomization

#### Phase 3: Optimization
1. **Batch processing**: Test multiple seeds in single call
2. **Parallel validation**: Use rayon or similar for multi-threading
3. **SIMD operations**: Vectorize suit counting and validation
4. **Memory optimization**: Zero-copy where possible

#### Phase 4: Integration
1. **FFI bindings**: Clean Lua interface
2. **Error handling**: Graceful failures
3. **Debug mode**: Detailed logging for development
4. **Testing suite**: Validate against known seeds

## Performance Targets
- **Speed**: 100,000+ seeds/second on modern hardware
- **Memory**: < 1MB runtime memory usage
- **Latency**: < 1ms for single seed validation
- **Batch size**: Test 1000+ seeds per call

## Technology Choice

### Rust (Recommended)
**Pros:**
- Excellent FFI support with `#[no_mangle]`
- Zero-cost abstractions
- Great SIMD support via `packed_simd`
- Mature ecosystem (rayon for parallelism)
- Memory safety without GC

**Cons:**
- Larger binary size
- Longer compile times

### Zig
**Pros:**
- Designed for FFI/C interop
- Smaller binaries
- Faster compilation
- Built-in cross-compilation

**Cons:**
- Less mature ecosystem
- Fewer libraries available
- Language still evolving

## Testing Strategy
1. **Unit tests**: Each component in isolation
2. **Integration tests**: Full seed validation pipeline
3. **Regression tests**: Known seed/result pairs
4. **Performance benchmarks**: Track speed improvements
5. **Fuzzing**: Random input validation

## Estimated Development Time
- **Research**: 1-2 days
- **Core implementation**: 2-3 days
- **Optimization**: 1-2 days
- **Testing & integration**: 1 day
- **Total**: ~1 week for MVP

## Future Enhancements
1. **GPU acceleration**: CUDA/OpenCL for massive parallelism
2. **Seed prediction**: ML model to predict good seeds
3. **Web service**: REST API for seed validation
4. **Seed database**: Pre-computed seeds with metadata
5. **Advanced filters**: Joker positions, shop contents, etc.