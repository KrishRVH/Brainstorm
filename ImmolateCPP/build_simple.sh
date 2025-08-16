#!/bin/bash

# Simple build script that compiles just the essential files
echo "Simple DLL build for Brainstorm"

# First install MinGW if not present
if ! command -v x86_64-w64-mingw32-g++ &> /dev/null; then
    echo "Installing MinGW cross-compiler..."
    sudo apt-get update
    sudo apt-get install -y mingw-w64
fi

echo "Compiling DLL..."

# Single command compilation
x86_64-w64-mingw32-g++ -shared -O2 -std=c++17 \
    -DBUILDING_DLL \
    -o ../Immolate_new.dll \
    src/brainstorm_enhanced.cpp \
    src/items.cpp \
    src/rng.cpp \
    src/seed.cpp \
    src/util.cpp \
    src/functions.cpp \
    -I src/ \
    -static-libgcc -static-libstdc++ \
    -Wl,--export-all-symbols

if [ $? -eq 0 ]; then
    echo "Success! Created Immolate_new.dll"
else
    echo "Compilation failed"
fi