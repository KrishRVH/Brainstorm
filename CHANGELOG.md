# Changelog

## v1.0.0-GA - General Availability Release
*August 30, 2025*

### 🎯 Major Features
- **GPU Acceleration**: 10M+ seeds/second on RTX 4090
- **Dynamic Pool Support**: Real-time pool updates from game state
- **Resume Support**: Continue searches across sessions
- **TDR Safety**: Automatic kernel slicing to prevent GPU timeouts
- **CPU Fallback**: Automatic fallback on CUDA errors

### ✅ Critical Fixes
- **Pack OR Semantics**: Searching for a pack now correctly finds it in EITHER slot
- **Per-Context Tag Indices**: Tags work correctly even when tag_small and tag_big have different orderings
- **FP Determinism**: Removed `-use_fast_math`, added precise math flags
- **Dynamic Index Resolution**: Filter names resolved at runtime, not compile-time
- **Null Pointer Safety**: Added synthetic pools for calibration

### 🔧 Technical Improvements
- FilterParams expanded to 40 bytes with per-slot/per-context fields
- Added v2 resolver function for complete index mapping
- Implemented SHA-256 pool IDs for reproducibility
- Added determinism self-test on initialization
- Created CPU-GPU differential runner for debugging
- Added shadow verification in production (every 50th call)
- Implemented rolling logs with JSON structure

### 📊 Performance
- Throughput: 10.23M seeds/sec (RTX 4090)
- Kernel Time: 245-248ms (under 250ms target)
- Memory Usage: 32MB
- TDR Margin: 87.5%

### 🐛 Known Limitations
- Long double warnings on GPU (cosmetic, hardware limitation)
- JSON must use flat array format
- Single GPU support only (multi-GPU in v2.0)
- Pack matching requires exact keys (no wildcards yet)

### 📦 Files
- `Immolate.dll` (3.2M) - Main DLL with GPU acceleration
- `Core/Brainstorm.lua` - Main mod integration
- `Core/BrainstormPoolUpdate.lua` - Pool management
- `UI/ui.lua` - Settings interface
- `config.lua` - User configuration

### 🔍 Verification
DLL SHA-256: `664b8e3c35c13bfe3cc6f1b171762daca3fe27a69521d3eb5de7429c994065b9`

---

## Previous Versions

### v0.9.0-RC1 - Release Candidate
*August 29, 2025*
- Initial GPU implementation
- Basic filtering support
- Known issues with pack and tag filtering

### v0.8.0-beta - Beta Release
*August 28, 2025*
- CPU-only implementation
- Proof of concept for seed filtering