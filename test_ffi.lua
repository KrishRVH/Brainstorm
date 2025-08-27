#!/usr/bin/env luajit
-- FFI Test - simulates how Balatro calls the DLL
-- Requires LuaJIT to run (same as Balatro uses)

local ffi = require("ffi")

-- Define the C functions exactly as Brainstorm.lua does
ffi.cdef([[
    const char* brainstorm(
        const char* seed,
        const char* voucher,
        const char* pack,
        const char* tag1,
        const char* tag2,
        double souls,
        bool observatory,
        bool perkeo
    );
    void free_result(const char* result);
    const char* get_hardware_info();
    void set_use_cuda(bool enable);
    int get_acceleration_type();
    const char* get_tags(const char* seed);
]])

-- Color codes for output
local RED = "\27[31m"
local GREEN = "\27[32m"
local YELLOW = "\27[33m"
local RESET = "\27[0m"

local function test(name, func)
  io.write(string.format("[TEST] %s... ", name))
  io.flush()

  local ok, err = pcall(func)
  if ok then
    print(GREEN .. "PASS" .. RESET)
    return true
  else
    print(RED .. "FAIL" .. RESET)
    print("  Error: " .. tostring(err))
    return false
  end
end

print("========================================")
print("  Brainstorm FFI Test Suite")
print("========================================")

-- Try to load the DLL
local dll_path = arg[1] or "./Immolate.dll"
print("\nLoading DLL: " .. dll_path)

local ok, immolate = pcall(ffi.load, dll_path)
if not ok then
  print(RED .. "[ERROR] Failed to load DLL: " .. tostring(immolate) .. RESET)
  os.exit(1)
end

print(GREEN .. "[OK] DLL loaded successfully" .. RESET)

local tests_passed = 0
local tests_failed = 0

-- Test 1: Get hardware info
if
  test("Get hardware info", function()
    local info = immolate.get_hardware_info()
    if info ~= nil then
      print("    Hardware: " .. ffi.string(info))
    end
  end)
then
  tests_passed = tests_passed + 1
else
  tests_failed = tests_failed + 1
end

-- Test 2: Get acceleration type
if
  test("Get acceleration type", function()
    local accel_type = immolate.get_acceleration_type()
    print("    Type: " .. (accel_type == 1 and "GPU" or "CPU"))
  end)
then
  tests_passed = tests_passed + 1
else
  tests_failed = tests_failed + 1
end

-- Test 3: Set CUDA (should not crash)
if test("Set CUDA enabled", function()
  immolate.set_use_cuda(true)
end) then
  tests_passed = tests_passed + 1
else
  tests_failed = tests_failed + 1
end

if test("Set CUDA disabled", function()
  immolate.set_use_cuda(false)
end) then
  tests_passed = tests_passed + 1
else
  tests_failed = tests_failed + 1
end

-- Test 4: Basic brainstorm call
if
  test("Brainstorm with no filters", function()
    local result =
      immolate.brainstorm("TESTTEST", nil, nil, nil, nil, 0, false, false)
    if result ~= nil then
      local seed = ffi.string(result)
      print("    Result: " .. seed)
      if immolate.free_result then
        pcall(immolate.free_result, result)
      end
    else
      print("    Result: nil (no match)")
    end
  end)
then
  tests_passed = tests_passed + 1
else
  tests_failed = tests_failed + 1
end

-- Test 5: Brainstorm with tag filter (GPU init happens here)
if
  test("Brainstorm with Speed Tag", function()
    local result = immolate.brainstorm(
      "TESTTEST",
      nil,
      nil,
      "Speed Tag",
      nil,
      0,
      false,
      false
    )
    if result ~= nil then
      local seed = ffi.string(result)
      print("    Result: " .. seed)
      if immolate.free_result then
        pcall(immolate.free_result, result)
      end
    else
      print("    Result: nil (no match)")
    end
  end)
then
  tests_passed = tests_passed + 1
else
  tests_failed = tests_failed + 1
end

-- Test 6: Brainstorm with dual tags
if
  test("Brainstorm with dual tags", function()
    local result = immolate.brainstorm(
      "TESTTEST",
      nil,
      nil,
      "Speed Tag",
      "Economy Tag",
      0,
      false,
      false
    )
    if result ~= nil then
      local seed = ffi.string(result)
      print("    Result: " .. seed)
      if immolate.free_result then
        pcall(immolate.free_result, result)
      end
    else
      print("    Result: nil (no match)")
    end
  end)
then
  tests_passed = tests_passed + 1
else
  tests_failed = tests_failed + 1
end

-- Test 7: Get tags for seed
if
  test("Get tags for seed", function()
    local tags = immolate.get_tags("TESTTEST")
    if tags ~= nil then
      local tag_str = ffi.string(tags)
      print("    Tags: " .. (tag_str ~= "" and tag_str or "empty"))
    end
  end)
then
  tests_passed = tests_passed + 1
else
  tests_failed = tests_failed + 1
end

-- Test 8: Memory leak test
if
  test("Memory leak test (100 calls)", function()
    for i = 1, 100 do
      local result = immolate.brainstorm(
        "TEST" .. string.format("%04d", i),
        nil,
        nil,
        nil,
        nil,
        0,
        false,
        false
      )
      if result ~= nil and immolate.free_result then
        pcall(immolate.free_result, result)
      end
    end
    print("    Completed 100 calls without crash")
  end)
then
  tests_passed = tests_passed + 1
else
  tests_failed = tests_failed + 1
end

-- Test 9: Edge cases
if
  test("Invalid seed format", function()
    -- Should handle gracefully
    local result =
      immolate.brainstorm("INVALID!", nil, nil, nil, nil, 0, false, false)
    if result ~= nil and immolate.free_result then
      pcall(immolate.free_result, result)
    end
  end)
then
  tests_passed = tests_passed + 1
else
  tests_failed = tests_failed + 1
end

-- Test 10: Stress test - rapid successive calls
if
  test("Stress test (1000 rapid calls)", function()
    local start_time = os.clock()
    for i = 1, 1000 do
      local result =
        immolate.brainstorm("RAPIDTST", nil, nil, nil, nil, 0, false, false)
      if result ~= nil and immolate.free_result then
        pcall(immolate.free_result, result)
      end
    end
    local elapsed = os.clock() - start_time
    print(
      string.format(
        "    Completed in %.2f seconds (%.0f calls/sec)",
        elapsed,
        1000 / elapsed
      )
    )
  end)
then
  tests_passed = tests_passed + 1
else
  tests_failed = tests_failed + 1
end

-- Summary
print("\n========================================")
print("  TEST SUMMARY")
print("========================================")
print("Passed: " .. GREEN .. tests_passed .. RESET)
print("Failed: " .. RED .. tests_failed .. RESET)

if tests_failed == 0 then
  print(
    "\n"
      .. GREEN
      .. "[SUCCESS] All FFI tests passed! DLL is Balatro-ready."
      .. RESET
  )
  os.exit(0)
else
  print(
    "\n"
      .. RED
      .. "[WARNING] Some tests failed. Fix issues before using in Balatro."
      .. RESET
  )
  os.exit(1)
end
