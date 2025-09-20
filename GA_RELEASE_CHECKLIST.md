# GA Release Checklist v1.0.0

Date: August 30, 2025  
Build: **3.2M**  
Status: **PRE-RELEASE VALIDATION**

---

## ✅ Pre-Release Audit Complete

### 1. FilterParams Size Verification
- [x] Host: 40 bytes (10 × uint32_t)
- [x] Device: 40 bytes (matches)
- [x] All HtoD copies use `sizeof(FilterParams)`
- [x] Static assert enforces size

### 2. ABI Exports
- [x] `brainstorm_resolve_filter_indices_v2` exported
- [x] No calls to v1 resolver remain
- [x] DLL exports verified with dumpbin

### 3. Kernel Arguments
- [x] 9 arguments in correct order:
  1. `uint64_t start_seed_index`
  2. `uint32_t total_seeds`
  3. `uint32_t chunk_size`
  4. `const FilterParams* params`
  5. `const DevicePools* pools`
  6. `uint64_t* candidates`
  7. `uint32_t cap`
  8. `uint32_t* cand_count`
  9. `volatile int* found`

### 4. Build Flags
- [x] NO `-use_fast_math` flag
- [x] `-Xptxas -fmad=false`
- [x] `-prec-div=true -prec-sqrt=true`
- [x] Compute 6.0+ target

### 5. Memory Safety
- [x] Synthetic pools freed after calibration
- [x] Double-buffered pool swapping
- [x] RAII ScopedContext for CUDA

---

## ✅ Correctness Patches Applied

### Pack OR Semantics
```cuda
// Line 323-333 in seed_filter_kernel_balatro.cu
if (want_p1 && want_p2) {
    match = (pack1_id == params->pack1) || (pack2_id == params->pack2);
}
```
**Status**: ✅ Implemented and tested

### Per-Context Tag Indices
```cpp
// FilterParams now has separate indices
uint32_t tag1_small, tag1_big;  // Different indices for same tag
uint32_t tag2_small, tag2_big;
```
**Status**: ✅ Implemented and tested

---

## ✅ Operational Features

### CPU Fallback
- [x] Automatic on CUDA init failure
- [x] One-line log message
- [x] Seamless to user

### Target MS Control
- [x] Default: 250ms
- [x] FFI setter: `brainstorm_set_target_ms()`
- [x] Range: [50, 1000] clamped

### Resume Support
- [x] Empty seed = resume
- [x] `brainstorm_reset_resume()` to reset
- [x] `brainstorm_get_resume_index()` to inspect

### Debug Mode
- [x] Off by default in production
- [x] 17-digit doubles when enabled
- [x] Per-context r-values logged

---

## ⏳ Validation Gates (In Progress)

### 1. In-Game Parity Test
**Seeds to test**: 20-50 across states
- [ ] AAAAAAAA (first shop)
- [ ] TESTTEST (mid-ante)
- [ ] ZZZZZZZZ (edge case)
- [ ] 10 random seeds
- [ ] 10 from actual gameplay

**For each seed**:
1. Call `Brainstorm.update_pools_from_game()`
2. Run GPU search with simple filter
3. Open shop in game
4. Verify exact match on voucher, pack1, pack2, tag_small, tag_big

**Pass criteria**: 0 mismatches

### 2. TDR Soak Test
**Duration**: 5 minutes continuous
- [ ] Set target_ms=250
- [ ] Run continuous searches
- [ ] Update pools at 2 and 4 minutes
- [ ] Monitor GPU responsiveness

**Pass criteria**: 
- 0 TDR timeouts
- No CUDA errors
- Stable throughput (~10M/s on RTX 4090)

---

## 📦 Release Package

### Files Structure
```
Brainstorm-v1.0.0-GA/
├── Immolate.dll          # Main DLL (3.2M)
├── Core/
│   ├── Brainstorm.lua
│   └── BrainstormPoolUpdate.lua
├── UI/
│   └── ui.lua
├── config.lua
├── docs/
│   ├── README.md         # User guide
│   ├── TECHNICAL.md      # Implementation details
│   └── CHANGELOG.md      # v1.0.0 changes
└── checksums.txt         # SHA-256 hashes
```

### Checksums
```
Immolate.dll: [TO BE CALCULATED AFTER FINAL BUILD]
```

---

## 🚀 Deployment Commands

### Build
```bash
cd ImmolateCPP
./build_driver.sh
```

### Calculate Checksum
```bash
sha256sum Immolate.dll > checksums.txt
```

### Tag Release
```bash
git tag -a v1.0.0-GA -m "General Availability Release"
git push origin v1.0.0-GA
```

### Create Archive
```bash
zip -r Brainstorm-v1.0.0-GA.zip \
    Immolate.dll \
    Core/ UI/ \
    config.lua \
    docs/ \
    checksums.txt
```

---

## 📊 Performance Benchmarks

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Throughput | >5M/s | 10.23M/s | ✅ |
| Kernel Time | <250ms | 245-248ms | ✅ |
| Memory | <100MB | 32MB | ✅ |
| TDR Margin | >80% | 87.5% | ✅ |

---

## 🐛 Known Limitations (Documented)

1. **Long double warnings**: GPU hardware limitation (cosmetic)
2. **JSON format**: Must be flat arrays
3. **Single GPU**: Multi-GPU in v2.0
4. **Pack matching**: Exact keys only (no wildcards yet)

---

## 📝 Post-Release Monitoring

### First 48 Hours
- [ ] Monitor Discord/GitHub for reports
- [ ] Collect any gpu_driver.log files
- [ ] Track seeds that fail
- [ ] Document GPU models with issues

### Triage Template
Users reporting issues should provide:
1. `gpu_driver.log`
2. `pool_update.log` 
3. Seed(s) that failed
4. GPU model and driver version
5. Whether first shop or mid-ante

### Hotfix Ready
Branch `hotfix/v1.0.1` prepared for:
- target_ms adjustment
- Logging verbosity changes
- Minor filter logic tweaks

---

## ✅ Sign-Off Criteria

Before shipping v1.0.0-GA:

1. [ ] In-game parity: 20-50 seeds tested, 0 mismatches
2. [ ] TDR soak: 5 minutes, 0 timeouts
3. [ ] Checksums calculated and documented
4. [ ] Release notes finalized
5. [ ] Documentation complete
6. [ ] Archive created and tested

**Ship Authorization**: _________________  
**Date**: _________________

---

## 🎯 Quick Validation Commands

```bash
# Test pack semantics
./test_pack_semantics

# Test tag contexts
./test_tag_contexts  

# Run differential test (needs pools.json and seeds.txt)
./differential_runner pools.json seeds.txt

# TDR soak (Windows)
test_tdr_soak.exe
```

---

**Status**: Awaiting final validation gates before GA release