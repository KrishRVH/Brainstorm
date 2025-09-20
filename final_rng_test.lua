-- Final test to understand the exact pseudoseed behavior
local function final_rng_test()
    if not G or not G.GAME or not pseudohash or not pseudoseed then
        return "Not ready"
    end
    
    local output = "=== FINAL RNG TEST ===\n"
    output = output .. os.date("%Y-%m-%d %H:%M:%S") .. "\n\n"
    
    -- Test 1: Create completely fresh state
    output = output .. "Test 1: Fresh state with AAAAAAAA\n"
    G.GAME.pseudorandom = {}  -- Completely empty
    G.GAME.pseudorandom.seed = "AAAAAAAA"
    
    -- Check what happens when we set seed
    output = output .. "After setting seed:\n"
    for k, v in pairs(G.GAME.pseudorandom) do
        if type(v) == "number" then
            output = output .. string.format("  %s = %.17g\n", k, v)
        else
            output = output .. string.format("  %s = %s\n", k, tostring(v))
        end
    end
    
    -- Now call pseudoseed for the first time
    output = output .. "\nFirst pseudoseed('Voucher') call:\n"
    local v1 = pseudoseed("Voucher")
    output = output .. string.format("  Result: %.17g\n", v1)
    output = output .. "  State after:\n"
    for k, v in pairs(G.GAME.pseudorandom) do
        if type(v) == "number" then
            output = output .. string.format("    %s = %.17g\n", k, v)
        else
            output = output .. string.format("    %s = %s\n", k, tostring(v))
        end
    end
    
    -- Call again
    output = output .. "\nSecond pseudoseed('Voucher') call:\n"
    local v2 = pseudoseed("Voucher")
    output = output .. string.format("  Result: %.17g\n", v2)
    
    output = output .. "\n"
    
    -- Test 2: Check shop_pack behavior
    output = output .. "Test 2: shop_pack with fresh state\n"
    G.GAME.pseudorandom = {seed = "AAAAAAAA"}
    
    local p1 = pseudoseed("shop_pack1")
    output = output .. string.format("  First shop_pack1: %.17g\n", p1)
    
    local p2 = pseudoseed("shop_pack1")
    output = output .. string.format("  Second shop_pack1: %.17g\n", p2)
    
    local p3 = pseudoseed("shop_pack1")
    output = output .. string.format("  Third shop_pack1: %.17g\n", p3)
    
    output = output .. "\n"
    
    -- Test 3: Check the actual formula by examining internals
    output = output .. "Test 3: Check hashed_seed\n"
    G.GAME.pseudorandom = {seed = "AAAAAAAA"}
    
    -- Try to trigger hashed_seed creation
    local _ = pseudoseed("test_key")
    
    output = output .. "After pseudoseed call:\n"
    if G.GAME.pseudorandom.hashed_seed then
        output = output .. string.format("  hashed_seed = %.17g\n", G.GAME.pseudorandom.hashed_seed)
        output = output .. string.format("  pseudohash(seed) = %.17g\n", pseudohash("AAAAAAAA"))
        output = output .. "  Are they equal? " .. tostring(G.GAME.pseudorandom.hashed_seed == pseudohash("AAAAAAAA")) .. "\n"
    else
        output = output .. "  No hashed_seed found\n"
    end
    
    return output
end

-- Hook into reroll
if Brainstorm and Brainstorm.reroll then
    local original_reroll = Brainstorm.reroll
    Brainstorm.reroll = function()
        local capture = final_rng_test()
        
        pcall(function()
            love.filesystem.write("Brainstorm_Final_Test.txt", capture)
            print("[Final Test] Saved to Brainstorm_Final_Test.txt")
        end)
        
        print(capture)
        
        return original_reroll()
    end
    
    print("[Final Test] Hooked - Press Ctrl+R")
end

return true