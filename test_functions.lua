-- Direct test functions for console

-- Add global functions that can be called from console
function test_basic()
    print("===== BASIC TEST =====")
    print("This function works!")
    if G and G.GAME then
        print("Game is loaded: YES")
        if G.GAME.pseudorandom then
            print("Current seed: " .. tostring(G.GAME.pseudorandom.seed))
        end
    else
        print("Game is loaded: NO")
    end
    return true
end

function test_rng_simple()
    print("===== RNG TEST =====")
    if not pseudohash then
        print("ERROR: pseudohash not found")
        return false
    end
    
    local result = pseudohash("AAAAAAAA")
    print(string.format("pseudohash('AAAAAAAA') = %.17g", result))
    print("Expected: 0.43257138351543745")
    
    if pseudoseed and G and G.GAME and G.GAME.pseudorandom then
        G.GAME.pseudorandom = {seed = "AAAAAAAA", hashed = result}
        local seed_result = pseudoseed("Voucher")
        print(string.format("pseudoseed('Voucher') = %.17g", seed_result))
        print("Expected: 0.46530388624939389")
    end
    
    return true
end

function trace_to_console()
    print("===== TRACING TO CONSOLE =====")
    
    if not G or not G.GAME or not G.GAME.pseudorandom then
        print("ERROR: Game not initialized")
        return false
    end
    
    local test_seeds = {"AAAAAAAA", "00000000", "7NTPKW6P"}
    
    for _, seed in ipairs(test_seeds) do
        print("\n--- Testing seed: " .. seed .. " ---")
        
        -- Save state
        local old_state = {}
        for k, v in pairs(G.GAME.pseudorandom) do
            old_state[k] = v
        end
        
        -- Set test seed
        local hashed = pseudohash(seed)
        G.GAME.pseudorandom = {seed = seed, hashed = hashed}
        
        print(string.format("  hashed = %.17g", hashed))
        
        -- Test RNG calls
        local v = pseudoseed("Voucher")
        print(string.format("  pseudoseed('Voucher') = %.17g", v))
        
        local p1 = pseudoseed("shop_pack1")
        print(string.format("  pseudoseed('shop_pack1') = %.17g", p1))
        
        local p2 = pseudoseed("shop_pack1")
        print(string.format("  pseudoseed('shop_pack1') #2 = %.17g", p2))
        
        local ts = pseudoseed("Tag_small")
        print(string.format("  pseudoseed('Tag_small') = %.17g", ts))
        
        local tb = pseudoseed("Tag_big")
        print(string.format("  pseudoseed('Tag_big') = %.17g", tb))
        
        -- Restore state
        G.GAME.pseudorandom = old_state
    end
    
    print("\n===== TRACE COMPLETE =====")
    return true
end

-- Register the functions globally
_G.test_basic = test_basic
_G.test_rng_simple = test_rng_simple
_G.trace_to_console = trace_to_console

print("[TEST] Functions loaded: test_basic(), test_rng_simple(), trace_to_console()")
return true