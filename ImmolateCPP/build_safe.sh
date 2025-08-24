#!/bin/bash

# Safe build with optional GPU support that won't crash if CUDA is unavailable
echo "Building safe DLL with optional GPU support..."

# Clean previous builds
rm -f Immolate.dll

# Build without GPU by default, but with GPU stubs
echo "Step 1: Compiling CPU-only with GPU stubs..."

x86_64-w64-mingw32-g++ -shared -O2 -std=c++17 \
    -DBUILDING_DLL \
    -DGPU_SAFE_MODE \
    -o Immolate.dll \
    src/brainstorm_enhanced.cpp \
    src/items.cpp \
    src/rng.cpp \
    src/seed.cpp \
    src/util.cpp \
    src/functions.cpp \
    -I src/ \
    -static-libgcc -static-libstdc++ \
    -Wl,--export-all-symbols 2>&1 | grep -v "warning:"

if [ $? -eq 0 ]; then
    echo "✓ Safe build successful"
    ls -lh Immolate.dll
else
    echo "✗ Build failed"
    exit 1
fi

echo ""
echo "======================================="
echo "Safe build complete!"
echo ""
echo "This build will:"
echo "  - Work on all systems (no CUDA required)"
echo "  - Use CPU for all seed searches"
echo "  - Not crash if GPU is unavailable"
echo "======================================="
