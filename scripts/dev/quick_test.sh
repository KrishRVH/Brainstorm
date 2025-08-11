#!/bin/bash

# Quick test script for development iteration
echo "Quick Development Test"
echo "====================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../.."

cd "$PROJECT_ROOT" || exit 1

# Test specific seed
SEED=${1:-7LB2WVPK}

echo "Testing seed: $SEED"
echo

# Build if needed
if [ ! -f balatro_rng_analyzer ]; then
    echo "Building analyzer..."
    gcc -O3 -o balatro_rng_analyzer balatro_rng_analyzer.c -lm
fi

# Test with analyzer
echo "RNG Analysis:"
./balatro_rng_analyzer | grep "$SEED" -A 10

# Test with Immolate if available
if [ -f tools/Immolate ]; then
    echo
    echo "GPU Test:"
    tools/Immolate -f ImmolateSourceCode/filters/erratic_brainstorm -s $SEED -n 1
fi

echo
echo "Done!"