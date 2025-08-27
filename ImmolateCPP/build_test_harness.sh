#!/bin/bash

echo "Building CUDA Driver API test harness..."

# Compile test harness
x86_64-w64-mingw32-g++ \
    -O2 \
    -std=c++17 \
    -I src \
    -o tests/cuda_drv_probe.exe \
    tests/cuda_drv_probe.cpp \
    -static-libgcc \
    -static-libstdc++

if [ $? -eq 0 ]; then
    echo "✓ Build successful: tests/cuda_drv_probe.exe"
    echo ""
    echo "To test on Windows:"
    echo "  1. Copy cuda_drv_probe.exe to Windows"
    echo "  2. Run: cuda_drv_probe.exe"
    echo "  3. Check if it prints 'All tests PASSED' or fails at cuMemAlloc"
else
    echo "✗ Build failed"
    exit 1
fi