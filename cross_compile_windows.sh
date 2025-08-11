#!/bin/bash

echo "Cross-compiling for Windows from WSL2"
echo "======================================"

# Install MinGW cross-compiler if needed
if ! command -v x86_64-w64-mingw32-gcc &> /dev/null; then
    echo "Installing MinGW cross-compiler..."
    sudo apt-get update
    sudo apt-get install -y mingw-w64
fi

# Build Windows DLL for the simple analyzer
echo "Building Windows DLL..."
x86_64-w64-mingw32-gcc -shared -O3 -o immolate_analyzer.dll \
    -DWINDOWS_DLL \
    balatro_rng_analyzer.c \
    -lm \
    -static-libgcc \
    -Wl,--out-implib,immolate_analyzer.lib

if [ $? -eq 0 ]; then
    echo "Success! Created immolate_analyzer.dll"
    echo ""
    echo "This DLL can be used on Windows directly."
    echo "Copy to Windows with:"
    echo "  cp immolate_analyzer.dll /mnt/c/Users/YOUR_USERNAME/Desktop/"
else
    echo "Build failed!"
    exit 1
fi