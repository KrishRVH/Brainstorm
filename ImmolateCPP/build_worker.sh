#!/bin/bash

echo "Building GPU Worker Process..."

# First ensure we have the PTX header
if [ ! -f "src/gpu/seed_filter_ptx.h" ]; then
    echo "Error: seed_filter_ptx.h not found. Run build_driver.sh first."
    exit 1
fi

# Compile worker executable with all necessary object files
x86_64-w64-mingw32-g++ \
    -O3 \
    -std=c++17 \
    -DGPU_ENABLED \
    -I src \
    -o ../gpu_worker.exe \
    src/gpu_worker.cpp \
    src/seed.cpp \
    src/util.cpp \
    -static-libgcc \
    -static-libstdc++ \
    -lws2_32

if [ $? -eq 0 ]; then
    echo "✓ Worker built successfully: gpu_worker.exe"
    ls -lh ../gpu_worker.exe
else
    echo "✗ Worker build failed"
    exit 1
fi