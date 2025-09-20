-- Super simple test to verify console works

print("===== SIMPLE TEST STARTED =====")

-- Test 1: Basic Lua
print("Test 1: Basic Lua - WORKS")

-- Test 2: Check if game is loaded
if G then
    print("Test 2: Game object G exists - WORKS")
else
    print("Test 2: Game object G not found - FAILED")
    return false
end

-- Test 3: Check RNG functions
if pseudohash and pseudoseed then
    print("Test 3: RNG functions exist - WORKS")
else
    print("Test 3: RNG functions not found - FAILED")
    return false
end

-- Test 4: Test pseudohash
local hash = pseudohash("AAAAAAAA")
print(string.format("Test 4: pseudohash('AAAAAAAA') = %.17g", hash))

-- Test 5: Check game state
if G.GAME and G.GAME.pseudorandom then
    print("Test 5: Game state exists - WORKS")
    print("  Current seed: " .. tostring(G.GAME.pseudorandom.seed))
else
    print("Test 5: Game state not found - FAILED")
    return false
end

-- Test 6: Test pseudoseed
local seed_val = pseudoseed("test_key")
print(string.format("Test 6: pseudoseed('test_key') = %.17g", seed_val))

print("===== ALL TESTS PASSED =====")
return true