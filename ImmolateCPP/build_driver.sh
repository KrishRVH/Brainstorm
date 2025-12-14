#!/bin/bash

# Brainstorm GPU Build Script with CUDA Driver API
# This uses PTX embedding for cross-compilation from WSL2 to Windows

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================="
echo "  Brainstorm Driver API Build v1.0"
echo "======================================="

# Check for CUDA
NVCC_PATH="/usr/local/cuda/bin/nvcc"
if [ ! -f "$NVCC_PATH" ]; then
    echo -e "${RED}✗ CUDA not found at $NVCC_PATH${NC}"
    exit 1
fi

CUDA_VERSION=$($NVCC_PATH --version | grep "release" | awk '{print $6}' | cut -c2-)
echo -e "${GREEN}✓ CUDA found:${NC} version $CUDA_VERSION at $NVCC_PATH"

# Check for MinGW
if ! command -v x86_64-w64-mingw32-g++ &> /dev/null; then
    echo -e "${RED}✗ MinGW not found${NC}"
    echo "Install with: sudo apt-get install mingw-w64"
    exit 1
fi
echo -e "${GREEN}✓ MinGW found${NC}"

# Clean and create build directory
rm -rf build
mkdir -p build
cd build

echo ""
echo -e "${YELLOW}Building GPU Driver API version...${NC}"

# Step 1: Compile CUDA kernel to PTX
echo "Step 1: Compiling CUDA kernel to PTX..."

$NVCC_PATH \
    -ptx \
    -O3 \
    -arch=compute_80 \
    -ccbin gcc-13 \
    -I ../src/gpu \
    -o seed_filter.ptx \
    ../src/gpu/seed_filter_kernel.cu \
    2>&1 | tee build_cuda.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}✗ PTX compilation failed${NC}"
    exit 1
fi

PTX_SIZE=$(du -h seed_filter.ptx | cut -f1)
echo -e "${GREEN}✓ PTX compiled successfully (${PTX_SIZE})${NC}"

# Step 2: Convert PTX to C header
echo "Step 2: Converting PTX to C header..."

xxd -i seed_filter.ptx > ../src/gpu/seed_filter_ptx.h

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ PTX conversion failed${NC}"
    exit 1
fi

# Fix the variable names in the header
sed -i 's/seed_filter_ptx/build_seed_filter_ptx/g' ../src/gpu/seed_filter_ptx.h

echo -e "${GREEN}✓ PTX embedded as C header${NC}"

# Step 3: Compile GPU driver bridge
echo "Step 3: Compiling GPU driver bridge..."

x86_64-w64-mingw32-g++ \
    -c \
    -O3 \
    -std=c++17 \
    -DGPU_ENABLED \
    -DBRAINSTORM_DEBUG \
    -I ../src \
    -o gpu_driver.o \
    ../src/gpu/gpu_kernel_driver.cpp \
    2>&1 | tee -a build_cuda.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}✗ GPU driver compilation failed${NC}"
    exit 1
fi

# Step 3b: Compile GPU worker client
echo "Step 3b: Compiling GPU worker client..."

x86_64-w64-mingw32-g++ \
    -c \
    -O3 \
    -std=c++17 \
    -DGPU_ENABLED \
    -I ../src \
    -o gpu_worker_client.o \
    ../src/gpu/gpu_worker_client.cpp \
    2>&1 | tee -a build_cuda.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}✗ GPU driver compilation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ GPU driver compiled${NC}"

# Step 4: Compile main brainstorm with Driver API support
echo "Step 4: Compiling main brainstorm..."

# Only create the wrapper if it doesn't exist
if [ ! -f ../src/brainstorm_driver.cpp ]; then
  echo "Creating brainstorm_driver.cpp..."
  cat > ../src/brainstorm_driver.cpp << 'EOF'
#include <windows.h>
#include <string>
#include <cstring>
#include <cstdint>
#include "gpu/gpu_types.h"

// External functions from gpu_kernel_driver.cpp
extern "C" std::string gpu_search_with_driver(
    const std::string& start_seed_str,
    const FilterParams& params
);

extern "C" void cleanup_gpu_driver();

// Main DLL entry point
extern "C" __declspec(dllexport) 
const char* brainstorm(
    const char* seed,
    const char* voucher,
    const char* pack,
    const char* tag1,
    const char* tag2,
    const char* souls,
    const char* observatory,
    const char* perkeo
) {
    // Convert string parameters to FilterParams
    FilterParams params;
    params.tag1 = (tag1 && strlen(tag1) > 0) ? std::stoul(tag1) : 0xFFFFFFFF;
    params.tag2 = (tag2 && strlen(tag2) > 0) ? std::stoul(tag2) : 0xFFFFFFFF;
    params.voucher = (voucher && strlen(voucher) > 0) ? std::stoul(voucher) : 0xFFFFFFFF;
    params.pack = (pack && strlen(pack) > 0) ? std::stoul(pack) : 0xFFFFFFFF;
    params.require_souls = (souls && strcmp(souls, "1") == 0) ? 1 : 0;
    params.require_observatory = (observatory && strcmp(observatory, "1") == 0) ? 1 : 0;
    params.require_perkeo = (perkeo && strcmp(perkeo, "1") == 0) ? 1 : 0;
    
    // Call GPU driver
    std::string result = gpu_search_with_driver(seed, params);
    
    if (!result.empty()) {
        // Return a copy that the caller can free
        char* result_copy = (char*)malloc(result.size() + 1);
        strcpy(result_copy, result.c_str());
        return result_copy;
    }
    
    return nullptr;
}

// Free result memory
extern "C" __declspec(dllexport)
void free_result(const char* result) {
    if (result) {
        free((void*)result);
    }
}

// Get hardware info
extern "C" __declspec(dllexport)
const char* get_hardware_info() {
    static char info[256];
    snprintf(info, sizeof(info), "CUDA Driver API (PTX JIT)");
    return info;
}

// DLL cleanup
BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpReserved) {
    if (fdwReason == DLL_PROCESS_DETACH) {
        cleanup_gpu_driver();
    }
    return TRUE;
}
EOF
fi

x86_64-w64-mingw32-g++ \
    -c \
    -O3 \
    -std=c++17 \
    -DBUILDING_DLL \
    -I ../src \
    -o brainstorm_driver.o \
    ../src/brainstorm_driver.cpp \
    2>&1 | tee -a build_cuda.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}✗ Brainstorm compilation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Brainstorm wrapper compiled${NC}"

# Step 5: Compile required support files
echo "Step 5: Compiling support files..."

# Compile seed.cpp (needed for Seed class)
x86_64-w64-mingw32-g++ \
    -c \
    -O3 \
    -std=c++17 \
    -I ../src \
    -o seed.o \
    ../src/seed.cpp \
    2>&1 | tee -a build_cuda.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}✗ Seed compilation failed${NC}"
    exit 1
fi

# Compile util.cpp (needed for pseudostep)
x86_64-w64-mingw32-g++ \
    -c \
    -O3 \
    -std=c++17 \
    -I ../src \
    -o util.o \
    ../src/util.cpp \
    2>&1 | tee -a build_cuda.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}✗ Util compilation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Support files compiled${NC}"

# Step 6: Link everything into DLL
echo "Step 6: Linking DLL with Driver API..."

x86_64-w64-mingw32-g++ \
    -shared \
    -o ../Immolate.dll \
    brainstorm_driver.o \
    gpu_driver.o \
    gpu_worker_client.o \
    seed.o \
    util.o \
    -static-libgcc \
    -static-libstdc++ \
    -Wl,--export-all-symbols \
    2>&1 | tee -a build_cuda.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}✓ GPU Driver API build successful${NC}"
    DLL_SIZE=$(du -h ../Immolate.dll | cut -f1)
    echo -e "  DLL size: ${DLL_SIZE}"
    
    # Copy PTX files for debugging
    cp seed_filter.ptx ../
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Build Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "The DLL uses CUDA Driver API with embedded PTX."
    echo "It will JIT-compile on the target system."
    echo ""
    echo "Requirements on target system:"
    echo "  - NVIDIA GPU with driver installed"
    echo "  - nvcuda.dll (comes with driver)"
    echo "  - No CUDA toolkit needed!"
    echo ""
    echo "Debug logs will be written to:"
    echo "  %AppData%\\Roaming\\Balatro\\Mods\\Brainstorm\\gpu_driver.log"
else
    echo -e "${RED}✗ Linking failed. Check build_cuda.log${NC}"
    exit 1
fi