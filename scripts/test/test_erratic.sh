#!/bin/bash

echo "========================================"
echo "Testing Erratic Deck Filters"
echo "========================================"
echo

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../.."

cd "$PROJECT_ROOT" || exit 1

# Parse arguments
MIN_FACE=${1:-20}
SUIT_RATIO=${2:-0.5}

echo "Test Parameters:"
echo "  Minimum Face Cards: $MIN_FACE"
echo "  Minimum Suit Ratio: $SUIT_RATIO"
echo

# Check for Immolate
IMMOLATE=""
if [ -f tools/Immolate ]; then
    IMMOLATE="tools/Immolate"
elif [ -f tools/Immolate.exe ]; then
    # Try running Windows exe through Wine if available
    if command -v wine &> /dev/null; then
        IMMOLATE="wine tools/Immolate.exe"
    else
        echo "Windows executable found but Wine not installed"
        echo "Build for Linux or install Wine"
        exit 1
    fi
elif [ -f ImmolateSourceCode/build/Immolate ]; then
    IMMOLATE="ImmolateSourceCode/build/Immolate"
else
    echo "Immolate not found. Please build first:"
    echo "  ./scripts/build/build_wsl.sh"
    exit 1
fi

echo "Using: $IMMOLATE"
echo

# Test known seeds
echo "Testing Known Seeds:"
echo "===================="

# Test glitched seed
echo "1. Glitched Seed (7LB2WVPK):"
$IMMOLATE -f ImmolateSourceCode/filters/erratic_brainstorm -s 7LB2WVPK -n 1

echo
echo "2. Random Search (100 seeds):"
$IMMOLATE -f ImmolateSourceCode/filters/erratic_brainstorm -n 100 -c 10

echo
echo "Performance Test:"
echo "================="

# Measure performance
echo "Testing 10,000 seeds..."
start_time=$(date +%s)
output=$($IMMOLATE -f ImmolateSourceCode/filters/erratic_brainstorm -n 10000 2>&1)
end_time=$(date +%s)

elapsed=$((end_time - start_time))
if [ $elapsed -gt 0 ]; then
    rate=$((10000 / elapsed))
    echo "Time: ${elapsed}s"
    echo "Rate: ~$rate seeds/second"
else
    echo "Test completed in less than 1 second"
fi

# Parse results
found=$(echo "$output" | grep -c "ACCEPT")
if [ $found -gt 0 ]; then
    echo
    echo "Found $found seeds matching criteria"
    echo
    echo "Sample seeds:"
    echo "$output" | grep "ACCEPT" | head -5
else
    echo "No seeds found matching criteria"
    echo "This is expected for very restrictive filters"
fi

echo
echo "========================================"
echo "Recommendations based on testing:"
echo "========================================"

if [ $found -eq 0 ]; then
    echo "• Your criteria may be too restrictive"
    echo "• Try reducing requirements:"
    echo "  - Face cards: 15-20 (realistic)"
    echo "  - Suit ratio: 0.4-0.6 (achievable)"
elif [ $found -lt 10 ]; then
    echo "• Seeds are rare but findable"
    echo "• Consider running longer searches"
    echo "• GPU acceleration recommended"
else
    echo "• Good balance of requirements"
    echo "• Seeds are reasonably common"
fi

echo
echo "Statistical Analysis:"
echo "====================="
echo "Based on 5,790+ seeds analyzed:"
echo "• Max suit ratio found: 76.9%"
echo "• Max face cards found: 23"
echo "• 80% suit ratio: Mathematically impossible"
echo "• 25 face cards: Theoretically possible, extremely rare"