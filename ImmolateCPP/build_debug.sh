#!/bin/bash
# Build Brainstorm DLL with debug instrumentation

echo "Building Brainstorm DLL (DEBUG MODE)..."

# Clean old build
rm -f Immolate.dll

# Build with debug driver
x86_64-w64-mingw32-g++ \
    -shared \
    -static \
    -O2 \
    -march=native \
    -o Immolate.dll \
    src/brainstorm_driver.cpp \
    src/gpu/gpu_kernel_driver_debug.cpp \
    src/seed.cpp \
    src/util.cpp \
    -lkernel32 -luser32 -lgdi32 -lwinspool -lshell32 \
    -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32 \
    -static-libgcc -static-libstdc++ \
    -Wl,--no-undefined \
    -Wl,--enable-runtime-pseudo-reloc \
    -Wl,--export-all-symbols

# Check if build succeeded
if [ -f Immolate.dll ]; then
    echo "✓ Built (DEBUG): $(du -h Immolate.dll | cut -f1)"
else
    echo "✗ Build failed"
    exit 1
fi