#!/bin/bash

echo "========================================"
echo "Building Enhanced Immolate from WSL2"
echo "========================================"

# Check if we're in WSL
if ! grep -q Microsoft /proc/version; then
    echo "Warning: This doesn't appear to be WSL2"
fi

# Navigate to source directory
cd ImmolateSourceCode || exit 1

# Install dependencies if needed
echo "Checking dependencies..."
if ! command -v cmake &> /dev/null; then
    echo "Installing cmake..."
    sudo apt-get update
    sudo apt-get install -y cmake
fi

if ! command -v g++ &> /dev/null; then
    echo "Installing build tools..."
    sudo apt-get install -y build-essential
fi

# Install OpenCL headers and ICD loader
echo "Installing OpenCL dependencies..."
sudo apt-get install -y ocl-icd-opencl-dev opencl-headers

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring build..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo "Building..."
make -j$(nproc)

if [ -f "Immolate" ]; then
    echo "Build successful!"
    echo ""
    echo "========================================"
    echo "IMPORTANT: GPU Access Limitations"
    echo "========================================"
    echo ""
    echo "WSL2 has LIMITED GPU support:"
    echo "- NVIDIA GPUs: Requires WSL2 GPU support + CUDA drivers"
    echo "- AMD GPUs: Generally NOT supported for OpenCL"
    echo "- Intel GPUs: Limited support"
    echo ""
    echo "For best performance, build and run on Windows directly."
    echo ""
    echo "To test if OpenCL works in WSL2:"
    echo "  ./Immolate --list_devices"
    echo ""
else
    echo "Build failed!"
    exit 1
fi