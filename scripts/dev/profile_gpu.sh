#!/bin/bash

echo "========================================"
echo "GPU Performance Profiler"
echo "========================================"
echo

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../.."

cd "$PROJECT_ROOT" || exit 1

# Find Immolate
IMMOLATE=""
if [ -f tools/Immolate ]; then
    IMMOLATE="tools/Immolate"
elif [ -f ImmolateSourceCode/build/Immolate ]; then
    IMMOLATE="ImmolateSourceCode/build/Immolate"
else
    echo "Immolate not found. Build first."
    exit 1
fi

# Create results directory
mkdir -p profiling_results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="profiling_results/gpu_profile_$TIMESTAMP.txt"

echo "Profiling GPU performance..." | tee $RESULT_FILE
echo "Results will be saved to: $RESULT_FILE" | tee -a $RESULT_FILE
echo | tee -a $RESULT_FILE

# System info
echo "System Information:" | tee -a $RESULT_FILE
echo "==================" | tee -a $RESULT_FILE
uname -a | tee -a $RESULT_FILE
echo | tee -a $RESULT_FILE

# GPU info
echo "GPU Information:" | tee -a $RESULT_FILE
echo "===============" | tee -a $RESULT_FILE
$IMMOLATE --list_devices 2>&1 | tee -a $RESULT_FILE
echo | tee -a $RESULT_FILE

# Performance tests
echo "Performance Tests:" | tee -a $RESULT_FILE
echo "=================" | tee -a $RESULT_FILE

test_sizes=(100 1000 10000 100000 1000000)
filters=("test" "erratic_brainstorm")

for filter in "${filters[@]}"; do
    if [ ! -f "ImmolateSourceCode/filters/${filter}.cl" ] && [ "$filter" != "test" ]; then
        continue
    fi
    
    echo | tee -a $RESULT_FILE
    echo "Filter: $filter" | tee -a $RESULT_FILE
    echo "-------------------" | tee -a $RESULT_FILE
    
    for size in "${test_sizes[@]}"; do
        echo -n "Seeds: $size - " | tee -a $RESULT_FILE
        
        # Run test 3 times and average
        total_time=0
        for i in {1..3}; do
            start_time=$(date +%s.%N)
            $IMMOLATE -f ImmolateSourceCode/filters/$filter -n $size > /dev/null 2>&1
            end_time=$(date +%s.%N)
            elapsed=$(echo "$end_time - $start_time" | bc)
            total_time=$(echo "$total_time + $elapsed" | bc)
        done
        
        avg_time=$(echo "scale=3; $total_time / 3" | bc)
        rate=$(echo "scale=0; $size / $avg_time" | bc)
        
        echo "Time: ${avg_time}s, Rate: $rate seeds/sec" | tee -a $RESULT_FILE
    done
done

# Memory profiling
echo | tee -a $RESULT_FILE
echo "Memory Usage Test:" | tee -a $RESULT_FILE
echo "==================" | tee -a $RESULT_FILE

$IMMOLATE -f ImmolateSourceCode/filters/test -n 100000 > /dev/null 2>&1 &
PID=$!
sleep 2

if [ -d /proc/$PID ]; then
    cat /proc/$PID/status | grep -E "VmRSS|VmPeak" | tee -a $RESULT_FILE
    kill $PID 2>/dev/null
fi

# Analysis
echo | tee -a $RESULT_FILE
echo "Analysis:" | tee -a $RESULT_FILE
echo "=========" | tee -a $RESULT_FILE

# Calculate scaling efficiency
echo "Scaling efficiency (should be near linear):" | tee -a $RESULT_FILE
# This would need the actual times from above to calculate

echo | tee -a $RESULT_FILE
echo "Profile complete. Results saved to: $RESULT_FILE"

# Generate summary
echo
echo "Quick Summary:"
echo "=============="
grep "Rate:" $RESULT_FILE | tail -5