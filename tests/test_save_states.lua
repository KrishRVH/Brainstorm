-- Integration tests for save state functionality
-- Tests save/load operations, data integrity, and error handling

-- Mock STR_PACK and STR_UNPACK if not available
if not STR_PACK then
  STR_PACK = function(data)
    -- Simple serialization for testing
    local success, result = pcall(function()
      local str = ""
      for k, v in pairs(data) do
        str = str .. tostring(k) .. "=" .. tostring(v) .. ";"
      end
      return str
    end)
    return success and result or ""
  end
end

if not STR_UNPACK then
  STR_UNPACK = function(str)
    -- Simple deserialization for testing
    return { mocked = true, original_length = #str }
  end
end

local function run_save_state_tests()
  print("=" .. string.rep("=", 50))
  print("  SAVE STATE INTEGRATION TESTS")
  print("=" .. string.rep("=", 50))

  local passed = 0
  local failed = 0

  -- Mock game state for testing
  local function create_mock_game_state(id)
    return {
      pseudorandom = { seed = "TEST" .. id },
      round_resets = {
        ante = id,
        blind_tags = {
          Small = "tag_skip",
          Big = "tag_investment",
        },
      },
      starting_deck_size = 52,
      dollars = 100 + id * 10,
      interest_amount = id,
      jokers = {
        { name = "Joker" .. id, edition = id % 2 == 0 and "foil" or nil },
      },
    }
  end

  -- Test 1: Basic save and load
  local function test_basic_save_load()
    print("\n--- Test 1: Basic Save/Load ---")

    local original = create_mock_game_state(1)

    -- Simulate save
    local saved_data = STR_PACK(original)
    if type(saved_data) == "string" and #saved_data > 0 then
      print("✓ Save: Data serialized (" .. #saved_data .. " bytes)")
      passed = passed + 1
    else
      print("✗ Save: Serialization failed")
      failed = failed + 1
      return false
    end

    -- Simulate load
    local loaded = STR_UNPACK(saved_data)
    if loaded and loaded.pseudorandom.seed == original.pseudorandom.seed then
      print("✓ Load: Data restored correctly")
      passed = passed + 1
    else
      print("✗ Load: Data mismatch")
      failed = failed + 1
      return false
    end

    return true
  end

  -- Test 2: Multiple save slots
  local function test_multiple_slots()
    print("\n--- Test 2: Multiple Save Slots ---")

    local slots = {}

    -- Save to 5 slots
    for i = 1, 5 do
      local state = create_mock_game_state(i)
      slots[i] = STR_PACK(state)
    end

    -- Verify each slot
    local all_unique = true
    for i = 1, 5 do
      for j = i + 1, 5 do
        if slots[i] == slots[j] then
          all_unique = false
          break
        end
      end
    end

    if all_unique then
      print("✓ All 5 slots contain unique data")
      passed = passed + 1
    else
      print("✗ Save slots have duplicate data")
      failed = failed + 1
    end

    -- Load and verify each slot
    local all_correct = true
    for i = 1, 5 do
      local loaded = STR_UNPACK(slots[i])
      if not loaded or loaded.dollars ~= (100 + i * 10) then
        all_correct = false
        break
      end
    end

    if all_correct then
      print("✓ All slots load with correct data")
      passed = passed + 1
    else
      print("✗ Some slots have incorrect data")
      failed = failed + 1
    end
  end

  -- Test 3: Compression
  local function test_compression()
    print("\n--- Test 3: Compression ---")

    -- Create large state
    local large_state = create_mock_game_state(1)
    large_state.cards = {}
    for i = 1, 52 do
      table.insert(large_state.cards, {
        suit = "Hearts",
        rank = "Ace",
        seal = i % 3 == 0 and "Gold" or nil,
        edition = i % 5 == 0 and "polychrome" or nil,
      })
    end

    local uncompressed = STR_PACK(large_state)

    -- Simulate compression (using zlib deflate)
    local compressed = love
      and love.data
      and love.data.compress
      and love.data.compress("string", "deflate", uncompressed, 1)

    if compressed then
      local ratio = #compressed / #uncompressed
      print(
        string.format("✓ Compression: %.1f%% of original size", ratio * 100)
      )
      passed = passed + 1

      -- Test decompression
      local decompressed = love.data.decompress("string", "deflate", compressed)
      if decompressed == uncompressed then
        print("✓ Decompression: Data integrity maintained")
        passed = passed + 1
      else
        print("✗ Decompression: Data corrupted")
        failed = failed + 1
      end
    else
      print("⚠ Compression: Not available in test environment")
    end
  end

  -- Test 4: Error handling
  local function test_error_handling()
    print("\n--- Test 4: Error Handling ---")

    -- Test corrupted data
    local corrupted = "return {corrupted = 'data" -- Missing closing brace
    local success, result = pcall(STR_UNPACK, corrupted)

    if not success then
      print("✓ Corrupted data: Caught error correctly")
      passed = passed + 1
    else
      print("✗ Corrupted data: Failed to catch error")
      failed = failed + 1
    end

    -- Test nil/empty data
    local nil_success = pcall(STR_UNPACK, nil)
    local empty_success = pcall(STR_UNPACK, "")

    if not nil_success and not empty_success then
      print("✓ Nil/empty data: Handled correctly")
      passed = passed + 1
    else
      print("✗ Nil/empty data: Not handled properly")
      failed = failed + 1
    end

    -- Test malicious input
    local malicious = "os.execute('rm -rf /')"
    local mal_success, mal_result = pcall(STR_UNPACK, malicious)

    if mal_success and type(mal_result) ~= "table" then
      print("✓ Malicious input: Safely rejected")
      passed = passed + 1
    else
      print("⚠ Malicious input: Check security")
    end
  end

  -- Test 5: Data integrity
  local function test_data_integrity()
    print("\n--- Test 5: Data Integrity ---")

    -- Test special characters
    local special_state = {
      name = "Test\"Name'With`Special",
      unicode = "🃏🎰💰",
      nested = {
        deep = {
          value = math.pi,
          bool = true,
          null = nil,
        },
      },
    }

    local packed = STR_PACK(special_state)
    local unpacked = STR_UNPACK(packed)

    if
      unpacked
      and unpacked.name == special_state.name
      and unpacked.unicode == special_state.unicode
      and unpacked.nested.deep.value == special_state.nested.deep.value
    then
      print("✓ Special characters and nested data preserved")
      passed = passed + 1
    else
      print("✗ Data integrity lost for special cases")
      failed = failed + 1
    end

    -- Test large numbers
    local number_state = {
      small = 0.0000001,
      large = 999999999999,
      negative = -123456.789,
      scientific = 1.23e-10,
    }

    packed = STR_PACK(number_state)
    unpacked = STR_UNPACK(packed)

    if
      unpacked
      and math.abs(unpacked.small - number_state.small) < 0.0000001
      and unpacked.large == number_state.large
    then
      print("✓ Numeric precision maintained")
      passed = passed + 1
    else
      print("✗ Numeric precision lost")
      failed = failed + 1
    end
  end

  -- Test 6: Performance
  local function test_performance()
    print("\n--- Test 6: Performance ---")

    local iterations = 100
    local state = create_mock_game_state(1)

    -- Test save performance
    local save_start = os.clock()
    for i = 1, iterations do
      local saved = STR_PACK(state)
    end
    local save_time = os.clock() - save_start

    print(
      string.format(
        "✓ Save performance: %d ops in %.3fs (%.0f ops/sec)",
        iterations,
        save_time,
        iterations / save_time
      )
    )

    -- Test load performance
    local saved_data = STR_PACK(state)
    local load_start = os.clock()
    for i = 1, iterations do
      local loaded = STR_UNPACK(saved_data)
    end
    local load_time = os.clock() - load_start

    print(
      string.format(
        "✓ Load performance: %d ops in %.3fs (%.0f ops/sec)",
        iterations,
        load_time,
        iterations / load_time
      )
    )

    if save_time < 1 and load_time < 1 then
      passed = passed + 2
    else
      print("⚠ Performance may be slow")
    end
  end

  -- Run all tests
  test_basic_save_load()
  test_multiple_slots()
  test_compression()
  test_error_handling()
  test_data_integrity()
  test_performance()

  print("\n" .. "=" .. string.rep("=", 50))
  print(string.format("  RESULTS: %d passed, %d failed", passed, failed))
  print("=" .. string.rep("=", 50))

  return failed == 0
end

-- Mock STR_PACK and STR_UNPACK if not available
if not STR_PACK then
  function STR_PACK(data)
    -- Simple serialization for testing
    local function serialize(t, indent)
      indent = indent or ""
      local result = "{"
      for k, v in pairs(t) do
        local key = type(k) == "string" and string.format("[%q]", k)
          or "[" .. k .. "]"
        if type(v) == "table" then
          result = result .. key .. "=" .. serialize(v, indent .. "  ") .. ","
        elseif type(v) == "string" then
          result = result .. key .. "=" .. string.format("%q", v) .. ","
        else
          result = result .. key .. "=" .. tostring(v) .. ","
        end
      end
      return result .. "}"
    end
    return "return " .. serialize(data)
  end
end

if not STR_UNPACK then
  function STR_UNPACK(str)
    if not str or str == "" then
      error("Empty string")
    end
    local fn, err = loadstring(str)
    if not fn then
      error(err)
    end
    return fn()
  end
end

-- Export for use in other tests
return {
  run = run_save_state_tests,
}
