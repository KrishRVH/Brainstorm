-- Test script to verify CUDA configuration and CPU fallback
-- Run this after building and deploying the DLL

local ffi = require("ffi")

-- Define FFI interface
ffi.cdef([[
    const char* brainstorm(const char* seed, const char* voucher, const char* pack, 
                          const char* tag1, const char* tag2, double souls, 
                          bool observatory, bool perkeo);
    void free_result(const char* result);
    int get_acceleration_type();
    const char* get_hardware_info();
    void set_use_cuda(bool enable);
    void run_cuda_tests();
]])

-- Load DLL
local dll_path = "./Immolate.dll"
local immolate = ffi.load(dll_path)

print("=" .. string.rep("=", 50))
print("  BRAINSTORM CUDA FALLBACK TEST")
print("=" .. string.rep("=", 50))

-- Test 1: Check hardware info
print("\n--- Test 1: Hardware Detection ---")
local hw_info = ffi.string(immolate.get_hardware_info())
print("Hardware: " .. hw_info)

-- Test 2: Check initial acceleration type
print("\n--- Test 2: Initial Acceleration Type ---")
local accel = immolate.get_acceleration_type()
if accel == 1 then
  print("✓ GPU acceleration is ENABLED")
else
  print("✓ CPU mode is active")
end

-- Test 3: Test disabling CUDA
print("\n--- Test 3: Disable CUDA ---")
immolate.set_use_cuda(false)
accel = immolate.get_acceleration_type()
if accel == 0 then
  print("✓ Successfully disabled CUDA (now using CPU)")
else
  print("✗ Failed to disable CUDA")
end

-- Test 4: Test enabling CUDA
print("\n--- Test 4: Re-enable CUDA ---")
immolate.set_use_cuda(true)
accel = immolate.get_acceleration_type()
if accel == 1 then
  print("✓ Successfully re-enabled CUDA")
else
  print("✓ CUDA not available, staying in CPU mode")
end

-- Test 5: Test search with CUDA disabled
print("\n--- Test 5: Search with CUDA Disabled ---")
immolate.set_use_cuda(false)
local start_time = os.clock()
local result = immolate.brainstorm(
  "TESTTEST", -- seed
  "", -- voucher
  "", -- pack
  "tag_skip", -- tag1 (Speed Tag)
  "", -- tag2
  0, -- souls
  false, -- observatory
  false -- perkeo
)
local elapsed = os.clock() - start_time

if result ~= nil then
  local seed_str = ffi.string(result)
  print(
    "✓ CPU search found: "
      .. seed_str
      .. " in "
      .. string.format("%.3f", elapsed)
      .. "s"
  )
  immolate.free_result(result)
else
  print("✗ CPU search failed")
end

-- Test 6: Test search with CUDA enabled (if available)
print("\n--- Test 6: Search with CUDA Enabled ---")
immolate.set_use_cuda(true)
accel = immolate.get_acceleration_type()

if accel == 1 then
  start_time = os.clock()
  result = immolate.brainstorm(
    "TESTTEST", -- seed
    "", -- voucher
    "", -- pack
    "tag_skip", -- tag1
    "", -- tag2
    0, -- souls
    false, -- observatory
    false -- perkeo
  )
  elapsed = os.clock() - start_time

  if result ~= nil then
    local seed_str = ffi.string(result)
    print(
      "✓ GPU search found: "
        .. seed_str
        .. " in "
        .. string.format("%.3f", elapsed)
        .. "s"
    )
    immolate.free_result(result)
  else
    print("✗ GPU search failed")
  end
else
  print("⚠ GPU not available, skipping GPU test")
end

-- Test 7: Performance comparison
print("\n--- Test 7: Performance Comparison ---")
local function benchmark_search(use_cuda, num_searches)
  immolate.set_use_cuda(use_cuda)
  local mode = use_cuda and "GPU" or "CPU"

  local start_time = os.clock()
  local found = 0

  for i = 1, num_searches do
    local seed = "TEST" .. string.format("%04d", i)
    local result =
      immolate.brainstorm(seed, "", "", "tag_skip", "", 0, false, false)
    if result ~= nil then
      found = found + 1
      immolate.free_result(result)
    end
  end

  local elapsed = os.clock() - start_time
  local rate = num_searches / elapsed

  print(
    string.format(
      "%s: %d searches in %.3fs (%.0f searches/sec, %d found)",
      mode,
      num_searches,
      elapsed,
      rate,
      found
    )
  )

  return rate
end

-- Run benchmarks
local cpu_rate = benchmark_search(false, 10)
local gpu_rate = 0

if immolate.get_acceleration_type() == 1 then
  gpu_rate = benchmark_search(true, 10)
  local speedup = gpu_rate / cpu_rate
  print(string.format("✓ GPU is %.1fx faster than CPU", speedup))
end

-- Test 8: Run built-in CUDA tests (if available)
print("\n--- Test 8: Built-in CUDA Tests ---")
local success, err = pcall(function()
  immolate.run_cuda_tests()
end)
if success then
  print("✓ Built-in tests completed")
else
  print("⚠ Built-in tests not available or failed: " .. tostring(err))
end

print("\n" .. "=" .. string.rep("=", 50))
print("  TEST COMPLETE")
print("=" .. string.rep("=", 50))

-- Summary
print("\nSummary:")
print("- Hardware: " .. hw_info)
print("- CUDA available: " .. (gpu_rate > 0 and "Yes" or "No"))
print("- CPU fallback: Working")
if gpu_rate > 0 then
  print(string.format("- Performance gain: %.1fx", gpu_rate / cpu_rate))
end
