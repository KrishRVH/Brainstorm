#!/bin/bash

echo "========================================"
echo "GPU Detection and Performance Test"
echo "========================================"
echo

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../.."

cd "$PROJECT_ROOT" || exit 1

# Check for Immolate
IMMOLATE=""
if [ -f tools/Immolate ]; then
    IMMOLATE="tools/Immolate"
elif [ -f ImmolateSourceCode/build/Immolate ]; then
    IMMOLATE="ImmolateSourceCode/build/Immolate"
else
    echo -e "${RED}Immolate not found${NC}"
    echo "Please build first: ./scripts/build/build_wsl.sh"
    exit 1
fi

echo "Using: $IMMOLATE"
echo

# Test 1: List OpenCL devices
echo "OpenCL Device Detection:"
echo "========================"
$IMMOLATE --list_devices

if [ $? -ne 0 ]; then
    echo
    echo -e "${YELLOW}No OpenCL devices detected${NC}"
    echo
    echo "Possible solutions:"
    echo "1. For WSL2 + NVIDIA:"
    echo "   - Install CUDA on Windows"
    echo "   - Install nvidia-docker2 in WSL2"
    echo "   - Update WSL2: wsl --update"
    echo
    echo "2. For native Linux + NVIDIA:"
    echo "   - Install CUDA toolkit"
    echo "   - Install nvidia-opencl-icd"
    echo
    echo "3. For AMD:"
    echo "   - Install rocm-opencl-runtime"
    echo
    echo "4. For Intel:"
    echo "   - Install intel-opencl-icd"
    echo
    echo "5. Use Windows build instead"
    exit 1
fi

echo
echo "Performance Benchmarks:"
echo "======================="

# Test different workloads
test_performance() {
    local filter=$1
    local seeds=$2
    local desc=$3
    
    echo "Test: $desc"
    echo "Filter: $filter"
    echo "Seeds: $seeds"
    
    start_time=$(date +%s.%N)
    $IMMOLATE -f ImmolateSourceCode/filters/$filter -n $seeds > /dev/null 2>&1
    end_time=$(date +%s.%N)
    
    elapsed=$(echo "$end_time - $start_time" | bc)
    rate=$(echo "scale=0; $seeds / $elapsed" | bc)
    
    echo "Time: ${elapsed}s"
    echo -e "${GREEN}Rate: $rate seeds/second${NC}"
    echo
}

# Run benchmarks
echo "Running benchmarks..."
echo

# Basic test
test_performance "test" 10000 "Basic filter (10k seeds)"

# Erratic test
if [ -f ImmolateSourceCode/filters/erratic_brainstorm.cl ]; then
    test_performance "erratic_brainstorm" 10000 "Erratic deck filter (10k seeds)"
fi

# Large test
echo "Large-scale test (100k seeds):"
test_performance "test" 100000 "Basic filter (100k seeds)"

echo
echo "Memory Usage:"
echo "============="
# Monitor memory during test
$IMMOLATE -f ImmolateSourceCode/filters/test -n 10000 &
PID=$!
sleep 1

if [ -d /proc/$PID ]; then
    MEM_KB=$(cat /proc/$PID/status | grep VmRSS | awk '{print $2}')
    MEM_MB=$((MEM_KB / 1024))
    echo "Process memory: ${MEM_MB}MB"
    wait $PID
else
    echo "Could not measure memory usage"
fi

echo
echo "========================================"
echo "Performance Analysis"
echo "========================================"

echo -e "${GREEN}Expected Performance Targets:${NC}"
echo "• CPU-only: 1,000-5,000 seeds/second"
echo "• Integrated GPU: 10,000-50,000 seeds/second"
echo "• Dedicated GPU: 100,000+ seeds/second"
echo
echo "Your system's performance will depend on:"
echo "• GPU model and drivers"
echo "• OpenCL implementation"
echo "• Filter complexity"
echo "• System memory bandwidth"