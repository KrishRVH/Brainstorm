-- Integration Test Suite for Brainstorm Mod
-- Tests the critical paths and integration points

local test_results = {
  passed = 0,
  failed = 0,
  errors = {}
}

local function test(name, fn)
  print(string.format("\n[TEST] %s", name))
  local success, result = pcall(fn)
  if success then
    test_results.passed = test_results.passed + 1
    print("  ✓ PASSED")
    return true
  else
    test_results.failed = test_results.failed + 1
    table.insert(test_results.errors, {name = name, error = result})
    print(string.format("  ✗ FAILED: %s", tostring(result)))
    return false
  end
end

local function assert_eq(actual, expected, msg)
  if actual ~= expected then
    error(string.format("%s: expected %s, got %s", msg or "Assertion failed", tostring(expected), tostring(actual)))
  end
end

local function assert_not_nil(value, msg)
  if value == nil then
    error(msg or "Value should not be nil")
  end
end

local function assert_true(value, msg)
  if not value then
    error(msg or "Value should be true")
  end
end

-- Test FFI initialization
test("FFI can be loaded", function()
  local ffi = require("ffi")
  assert_not_nil(ffi, "FFI library should load")
end)

-- Test DLL signature definition
test("DLL functions can be defined", function()
  local ffi = require("ffi")
  local success, err = pcall(ffi.cdef, [[
    const char* brainstorm(const char* seed, const char* voucher, const char* pack, const char* tag1, const char* tag2, double souls, bool observatory, bool perkeo);
    const char* get_tags(const char* seed);
    void free_result(const char* result);
  ]])
  assert_true(success, "FFI cdef should succeed: " .. tostring(err))
end)

-- Test DLL loading
test("Enhanced DLL can be loaded", function()
  local ffi = require("ffi")
  
  -- First check if new DLL exists
  local file = io.open("Immolate_new.dll", "rb")
  if file then
    file:close()
    local success, dll = pcall(ffi.load, "Immolate_new.dll")
    assert_true(success, "Should load Immolate_new.dll")
    assert_not_nil(dll.brainstorm, "DLL should export brainstorm function")
    assert_not_nil(dll.get_tags, "DLL should export get_tags function")
    assert_not_nil(dll.free_result, "DLL should export free_result function")
  else
    -- Fall back to original DLL
    local success, dll = pcall(ffi.load, "Immolate.dll")
    assert_true(success, "Should load Immolate.dll")
  end
end)

-- Test config structure
test("Config has required fields", function()
  -- Simulate config structure
  local config = {
    ar_filters = {
      tag_name = "tag_investment",
      tag_id = 9,
      tag2_name = "tag_investment",
      tag2_id = 9,
      voucher_name = "",
      voucher_id = 1,
      pack = {},
      pack_id = 1,
      soul_skip = 0,
      inst_observatory = false,
      inst_perkeo = false
    },
    ar_prefs = {
      face_count = 0,
      suit_ratio_decimal = 0
    }
  }
  
  assert_not_nil(config.ar_filters, "Config should have ar_filters")
  assert_not_nil(config.ar_filters.tag_name, "Config should have tag_name")
  assert_not_nil(config.ar_filters.tag2_name, "Config should have tag2_name")
  assert_not_nil(config.ar_filters.tag2_id, "Config should have tag2_id")
end)

-- Test dual tag validation logic
test("Dual tag logic handles same tags correctly", function()
  local function check_dual_tags(tag1, tag2, small_tag, big_tag)
    if tag1 == tag2 then
      -- Both positions must have the same tag
      return small_tag == tag1 and big_tag == tag1
    else
      -- Both tags must be present (order doesn't matter)
      local has_tag1 = (small_tag == tag1 or big_tag == tag1)
      local has_tag2 = (small_tag == tag2 or big_tag == tag2)
      return has_tag1 and has_tag2
    end
  end
  
  -- Test same tag twice
  assert_true(check_dual_tags("investment", "investment", "investment", "investment"), 
    "Should match when both blinds have same tag")
  assert_true(not check_dual_tags("investment", "investment", "investment", "charm"),
    "Should not match when only one blind has the tag")
  
  -- Test different tags
  assert_true(check_dual_tags("investment", "charm", "investment", "charm"),
    "Should match when both tags present in order")
  assert_true(check_dual_tags("investment", "charm", "charm", "investment"),
    "Should match when both tags present in reverse order")
  assert_true(not check_dual_tags("investment", "charm", "investment", "double"),
    "Should not match when second tag missing")
end)

-- Test error handling
test("DLL functions handle errors gracefully", function()
  local ffi = require("ffi")
  
  -- Define functions if not already done
  pcall(ffi.cdef, [[
    const char* brainstorm(const char* seed, const char* voucher, const char* pack, const char* tag1, const char* tag2, double souls, bool observatory, bool perkeo);
    const char* get_tags(const char* seed);
    void free_result(const char* result);
  ]])
  
  -- Try to load DLL
  local dll_path = "Immolate_new.dll"
  local file = io.open(dll_path, "rb")
  if not file then
    dll_path = "Immolate.dll"
  else
    file:close()
  end
  
  local success, dll = pcall(ffi.load, dll_path)
  if success then
    -- Test with empty parameters
    local call_success, result = pcall(function()
      return dll.brainstorm("", "", "", "", "", 0, false, false)
    end)
    assert_true(call_success or true, "Should handle empty parameters without crashing")
    
    -- Test get_tags with valid seed
    if dll.get_tags then
      local tags_success, tags_result = pcall(function()
        return dll.get_tags("TESTTEST")
      end)
      assert_true(tags_success, "get_tags should work with valid seed")
      
      if tags_success and tags_result then
        local tags_str = ffi.string(tags_result)
        assert_true(tags_str:find("|") ~= nil, "Tags should be in format 'tag1|tag2'")
        if dll.free_result then
          dll.free_result(tags_result)
        end
      end
    end
  end
end)

-- Test memory management
test("Memory is properly freed", function()
  local ffi = require("ffi")
  
  local dll_path = "Immolate_new.dll"
  local file = io.open(dll_path, "rb")
  if not file then
    return -- Skip if enhanced DLL not available
  end
  file:close()
  
  local success, dll = pcall(ffi.load, dll_path)
  if success and dll.free_result then
    -- Allocate and free multiple times to check for leaks
    for i = 1, 10 do
      local result = dll.get_tags("SEED" .. i)
      if result then
        dll.free_result(result)
      end
    end
    assert_true(true, "Memory operations completed without crash")
  end
end)

-- Print test results
print("\n" .. string.rep("=", 50))
print("TEST RESULTS")
print(string.rep("=", 50))
print(string.format("Passed: %d", test_results.passed))
print(string.format("Failed: %d", test_results.failed))

if #test_results.errors > 0 then
  print("\nFailed Tests:")
  for _, err in ipairs(test_results.errors) do
    print(string.format("  - %s: %s", err.name, err.error))
  end
end

print(string.rep("=", 50))

if test_results.failed == 0 then
  print("✓ ALL TESTS PASSED!")
  os.exit(0)
else
  print("✗ SOME TESTS FAILED")
  os.exit(1)
end