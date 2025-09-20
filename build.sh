#!/bin/bash

# Clean build script - just the essentials
cd "$(dirname "$0")"

echo "Building Brainstorm GPU DLL..."

# Step 1: Compile CUDA kernel to PTX (with exact floating-point)
echo "Compiling CUDA kernel..."
/usr/local/cuda/bin/nvcc \
    -ptx \
    -arch=sm_80 \
    -O3 -lineinfo \
    -Xptxas -fmad=false \
    -prec-div=true \
    -prec-sqrt=true \
    -o seed_filter.ptx \
    src/gpu/seed_filter_kernel_optimized.cu

# Step 2: Embed PTX in C header
echo "Embedding PTX..."
xxd -i seed_filter.ptx > src/gpu/seed_filter_ptx.h
sed -i 's/unsigned char seed_filter_ptx/unsigned char seed_filter_kernel_ptx/g' src/gpu/seed_filter_ptx.h
sed -i 's/unsigned int seed_filter_ptx_len/unsigned int seed_filter_kernel_ptx_len/g' src/gpu/seed_filter_ptx.h

# Step 3: Compile DLL (with exact floating-point)
echo "Compiling DLL..."
x86_64-w64-mingw32-g++ \
    -shared \
    -static \
    -O3 \
    -fno-fast-math \
    -ffp-contract=off \
    -fexcess-precision=standard \
    -march=native \
    -flto \
    -o Immolate.dll \
    src/brainstorm_driver.cpp \
    src/gpu/gpu_kernel_driver_prod.cpp \
    src/gpu/gpu_worker_client.cpp \
    src/seed.cpp \
    src/filters.cpp \
    -I src \
    -lws2_32 \
    -std=c++17

echo "✓ Build complete! DLL size: $(du -h Immolate.dll | cut -f1)"