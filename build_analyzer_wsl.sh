#!/bin/bash

echo "Building Balatro RNG Analyzer for WSL2"
echo "========================================"

# Simple C compilation - this will work fine in WSL2
gcc -O3 -o balatro_rng_analyzer balatro_rng_analyzer.c -lm

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo ""
    echo "Run with: ./balatro_rng_analyzer"
    echo ""
    echo "This will test the glitched seed 7LB2WVPK"
else
    echo "Build failed!"
    exit 1
fi