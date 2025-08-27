#!/bin/bash
# Quick test to validate DLL before deployment

echo "========================================"
echo "  Quick DLL Validation"
echo "========================================"

# Check DLL exists and size
if [ -f "Immolate.dll" ]; then
    SIZE=$(stat -c%s "Immolate.dll")
    SIZE_MB=$(echo "scale=2; $SIZE/1048576" | bc)
    echo "✓ DLL found: ${SIZE_MB} MB"
else
    echo "✗ DLL not found!"
    exit 1
fi

# Check for common crash indicators in logs
if [ -f "gpu_driver.log" ]; then
    if grep -q "invalid device context\|CUDA_ERROR\|Failed to allocate\|crash\|segfault" gpu_driver.log 2>/dev/null; then
        echo "✗ GPU driver log contains errors:"
        tail -10 gpu_driver.log
        exit 1
    fi
fi

# Run basic Lua tests
echo ""
echo "Running basic tests..."
if lua basic_test.lua 2>&1 | grep -q "Tests Complete"; then
    echo "✓ Basic tests passed"
else
    echo "✗ Basic tests failed"
    exit 1
fi

# Check if LuaJIT is available for FFI test
if command -v luajit &> /dev/null; then
    echo ""
    echo "Testing with LuaJIT FFI (simulates Balatro)..."
    if timeout 5 luajit test_ffi.lua 2>&1 | grep -q "SUCCESS"; then
        echo "✓ FFI tests passed - DLL is safe!"
    else
        echo "✗ FFI tests failed - would crash in Balatro!"
        exit 1
    fi
else
    echo ""
    echo "⚠ LuaJIT not installed - can't test FFI"
    echo "  Install with: sudo apt-get install luajit"
fi

echo ""
echo "========================================"
echo "  Validation Complete"
echo "========================================"
echo ""
echo "DLL appears safe to deploy. Run ./deploy.sh to install."