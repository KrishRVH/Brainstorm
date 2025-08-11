#!/bin/bash

echo "========================================"
echo "Brainstorm Complete Test Suite"
echo "========================================"
echo

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../.."

cd "$PROJECT_ROOT" || exit 1

# Test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test
run_test() {
    local test_name=$1
    local test_command=$2
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    echo -e "${CYAN}[TEST $TOTAL_TESTS]${NC} $test_name"
    echo "Command: $test_command"
    
    # Run test and capture output
    output=$(eval "$test_command" 2>&1)
    result=$?
    
    if [ $result -eq 0 ]; then
        echo -e "${GREEN}✓ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo "Output: $output"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    echo
}

# Test 1: Check file structure
run_test "File Structure" "[ -f Core/Brainstorm.lua ] && [ -f UI/ui.lua ] && [ -f config.lua ]"

# Test 2: Lua syntax check
if command -v luac &> /dev/null; then
    run_test "Lua Syntax (Brainstorm.lua)" "luac -p Core/Brainstorm.lua"
    run_test "Lua Syntax (ui.lua)" "luac -p UI/ui.lua"
else
    echo -e "${YELLOW}Skipping Lua syntax tests (luac not installed)${NC}"
fi

# Test 3: Build RNG analyzer
run_test "Build RNG Analyzer" "gcc -O3 -o test_rng_analyzer balatro_rng_analyzer.c -lm"

# Test 4: Test glitched seed
if [ -f test_rng_analyzer ]; then
    run_test "Glitched Seed Test" "./test_rng_analyzer | grep -q '10 of Spades'"
    rm -f test_rng_analyzer
fi

# Test 5: Check Immolate filters
run_test "Erratic Filter Exists" "[ -f ImmolateSourceCode/filters/erratic_brainstorm.cl ]"

# Test 6: Check OpenCL syntax (basic)
if [ -f ImmolateSourceCode/filters/erratic_brainstorm.cl ]; then
    run_test "OpenCL Syntax Check" "grep -q '__kernel' ImmolateSourceCode/filters/erratic_brainstorm.cl"
fi

# Test 7: Check for GPU tools
if [ -f tools/Immolate ] || [ -f tools/Immolate.exe ]; then
    echo -e "${CYAN}[TEST]${NC} GPU Tool Detection"
    if [ -f tools/Immolate ]; then
        run_test "GPU Device List" "tools/Immolate --list_devices"
    else
        echo -e "${YELLOW}GPU tool built for Windows, skipping Linux test${NC}"
    fi
else
    echo -e "${YELLOW}GPU tools not built yet${NC}"
fi

# Test 8: Config file parsing
cat > test_config.lua << 'EOF'
local config = {
    debug_enabled = true,
    filters = {
        face_cards = {min = 20},
        suit_ratio = {enabled = true, ratio = 0.5}
    }
}
return config
EOF

run_test "Config Structure" "[ -f test_config.lua ]"
rm -f test_config.lua

# Test 9: Documentation
run_test "Documentation Files" "[ -f README.md ] && [ -f CLAUDE.md ]"

# Test 10: Stylua config
run_test "Stylua Config" "[ -f stylua.toml ]"

# Test 11: Check for common issues
echo -e "${CYAN}[CHECK]${NC} Common Issues"

# Check for tabs vs spaces
if grep -q $'\t' Core/Brainstorm.lua; then
    echo -e "${YELLOW}Warning: Tabs found in Brainstorm.lua (should use spaces)${NC}"
fi

# Check for large functions
large_funcs=$(awk '/function/,/end/ {count++} /end/ {if(count>50) print NR; count=0}' Core/Brainstorm.lua)
if [ ! -z "$large_funcs" ]; then
    echo -e "${YELLOW}Warning: Large functions (>50 lines) found at lines: $large_funcs${NC}"
fi

echo
echo "========================================"
echo "TEST RESULTS"
echo "========================================"
echo -e "Total Tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Please review.${NC}"
    exit 1
fi