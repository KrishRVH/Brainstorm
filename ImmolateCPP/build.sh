#!/bin/bash
set -e

echo "Building Brainstorm DLL..."

# Create combined CUDA source with both kernels
cat src/gpu/seed_filter_kernel_balatro.cu > combined_kernels.cu
echo "" >> combined_kernels.cu
echo "// Probe kernel functions" >> combined_kernels.cu
cat src/gpu/probe_kernel.cu >> combined_kernels.cu

# Compile combined kernels to PTX (with exact floating-point)
/usr/local/cuda/bin/nvcc \
    -ptx \
    -arch=sm_80 \
    -O3 -lineinfo \
    -Xptxas -fmad=false \
    -prec-div=true \
    -prec-sqrt=true \
    -ccbin gcc-13 \
    -I src/gpu \
    -o seed_filter.ptx \
    combined_kernels.cu

# Clean up
rm -f combined_kernels.cu

# Embed PTX in header
xxd -i seed_filter.ptx > src/gpu/seed_filter_ptx.h
sed -i 's/unsigned char seed_filter_ptx/unsigned char seed_filter_kernel_ptx/g' src/gpu/seed_filter_ptx.h
sed -i 's/unsigned int seed_filter_ptx_len/unsigned int seed_filter_kernel_ptx_len/g' src/gpu/seed_filter_ptx.h

# Compile and link DLL (with exact floating-point)
x86_64-w64-mingw32-g++ \
    -shared \
    -static \
    -O3 \
    -fno-fast-math \
    -ffp-contract=off \
    -march=native \
    -flto \
    -o Immolate.dll \
    src/brainstorm_driver.cpp \
    src/gpu/gpu_kernel_driver_prod.cpp \
    src/cpu_fallback_balatro.cpp \
    src/pool_manager.cpp \
    src/seed.cpp \
    src/util.cpp \
    -I src \
    -lws2_32 \
    -std=c++17

echo "✓ Built: $(du -h Immolate.dll | cut -f1)"