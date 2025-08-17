#!/bin/bash

# Build script for safe GPU-accelerated Brainstorm DLL
# This version has safe CUDA initialization with fallback

echo "======================================="
echo "  Brainstorm Safe Build Script v1.0"
echo "======================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for MinGW
if ! command -v x86_64-w64-mingw32-g++ &> /dev/null; then
    echo -e "${RED}✗ MinGW not found. Please install mingw-w64${NC}"
    echo "  sudo apt-get install mingw-w64"
    exit 1
fi

echo -e "${GREEN}✓ MinGW found${NC}"

# Create build directory
mkdir -p build
cd build

echo -e "\n${YELLOW}Building safe hybrid version with GPU fallback...${NC}"

# Compile safe brainstorm with optional GPU support
x86_64-w64-mingw32-g++ \
    -shared \
    -O3 \
    -std=c++17 \
    -DBUILDING_DLL \
    -DGPU_ENABLED \
    -o ../Immolate.dll \
    ../src/brainstorm_safe.cpp \
    ../src/items.cpp \
    ../src/rng.cpp \
    ../src/seed.cpp \
    ../src/util.cpp \
    ../src/functions.cpp \
    -I ../src/ \
    -static-libgcc \
    -static-libstdc++ \
    -pthread \
    -Wl,--export-all-symbols \
    2>&1 | tee build_safe.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}✓ Safe build successful${NC}"
    DLL_SIZE=$(du -h ../Immolate.dll | cut -f1)
    echo -e "  DLL size: ${DLL_SIZE}"
    echo -e "  Features: CPU + Safe GPU fallback"
else
    echo -e "${RED}✗ Build failed. Check build_safe.log${NC}"
    exit 1
fi

echo -e "\n======================================="
echo -e "${GREEN}Build complete!${NC}"
echo ""
echo "This DLL includes:"
echo "  - Safe CUDA initialization with timeout"
echo "  - Automatic fallback to CPU if GPU fails"
echo "  - No crashes from CUDA issues"
echo ""
echo "To deploy: ./deploy.sh"
echo "======================================="