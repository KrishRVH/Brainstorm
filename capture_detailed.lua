-- Detailed RNG capture to understand pseudoseed behavior
local function capture_detailed_rng()
    if not G or not G.GAME or not G.GAME.pseudorandom then
        return "Game not ready"
    end
    
    local output = "=== DETAILED RNG CAPTURE ===\n"
    output = output .. os.date("%Y-%m-%d %H:%M:%S") .. "\n\n"
    
    -- Current state before any changes
    output = output .. "Initial State:\n"
    output = output .. string.format("  Seed: %s\n", tostring(G.GAME.pseudorandom.seed))
    output = output .. string.format("  Hashed: %.17g\n", G.GAME.pseudorandom.hashed or 0)
    output = output .. "\n"
    
    -- Capture initial state details
    output = output .. "G.GAME.pseudorandom contents:\n"
    for k, v in pairs(G.GAME.pseudorandom) do
        output = output .. string.format("  %s = %s (type: %s)\n", k, tostring(v), type(v))
    end
    output = output .. "\n"
    
    -- Test hash values
    if pseudohash then
        output = output .. "Hash Values (verified):\n"
        output = output .. string.format("  pseudohash('AAAAAAAA') = %.17g\n", pseudohash("AAAAAAAA"))
        output = output .. string.format("  pseudohash('7NTPKW6P') = %.17g\n", pseudohash("7NTPKW6P"))
        output = output .. string.format("  pseudohash('00000000') = %.17g\n", pseudohash("00000000"))
        output = output .. "\n"
    end
    
    -- Save current state
    local saved_state = {}
    for k, v in pairs(G.GAME.pseudorandom) do
        saved_state[k] = v
    end
    
    -- Test setting seed and checking hashed
    output = output .. "Testing seed setting:\n"
    G.GAME.pseudorandom.seed = "AAAAAAAA"
    output = output .. string.format("  After setting seed='AAAAAAAA', hashed = %.17g\n", G.GAME.pseudorandom.hashed or 0)
    
    -- Now set hashed manually
    G.GAME.pseudorandom.hashed = pseudohash("AAAAAAAA")
    output = output .. string.format("  After setting hashed manually = %.17g\n", G.GAME.pseudorandom.hashed)
    output = output .. "\n"
    
    -- Check state after setting
    output = output .. "State after manual setup:\n"
    for k, v in pairs(G.GAME.pseudorandom) do
        output = output .. string.format("  %s = %s\n", k, tostring(v))
    end
    output = output .. "\n"
    
    -- Now test pseudoseed calls
    if pseudoseed then
        output = output .. "Pseudoseed calls with AAAAAAAA:\n"
        
        -- Reset to clean state
        G.GAME.pseudorandom = {seed = "AAAAAAAA"}
        G.GAME.pseudorandom.hashed = pseudohash("AAAAAAAA")
        output = output .. string.format("  State before first call: hashed = %.17g\n", G.GAME.pseudorandom.hashed)
        
        local v1 = pseudoseed("Voucher")
        output = output .. string.format("  pseudoseed('Voucher') = %.17g\n", v1)
        output = output .. string.format("  State after: hashed = %.17g\n", G.GAME.pseudorandom.hashed)
        
        local p1 = pseudoseed("shop_pack1")
        output = output .. string.format("  pseudoseed('shop_pack1') #1 = %.17g\n", p1)
        output = output .. string.format("  State after: hashed = %.17g\n", G.GAME.pseudorandom.hashed)
        
        local p2 = pseudoseed("shop_pack1")
        output = output .. string.format("  pseudoseed('shop_pack1') #2 = %.17g\n", p2)
        output = output .. string.format("  State after: hashed = %.17g\n", G.GAME.pseudorandom.hashed)
        
        output = output .. "\n"
        
        -- Test with fresh state each time
        output = output .. "Testing with fresh state each call:\n"
        
        G.GAME.pseudorandom = {seed = "AAAAAAAA", hashed = pseudohash("AAAAAAAA")}
        local v_fresh = pseudoseed("Voucher")
        output = output .. string.format("  Fresh state: pseudoseed('Voucher') = %.17g\n", v_fresh)
        
        G.GAME.pseudorandom = {seed = "AAAAAAAA", hashed = pseudohash("AAAAAAAA")}
        local p_fresh = pseudoseed("shop_pack1")
        output = output .. string.format("  Fresh state: pseudoseed('shop_pack1') = %.17g\n", p_fresh)
    end
    
    -- Restore original state
    G.GAME.pseudorandom = saved_state
    
    return output
end

-- Hook into reroll
if Brainstorm and Brainstorm.reroll then
    local original_reroll = Brainstorm.reroll
    Brainstorm.reroll = function()
        local capture = capture_detailed_rng()
        
        pcall(function()
            love.filesystem.write("Brainstorm_Detailed_Capture.txt", capture)
            print("[Detailed Capture] Saved to Brainstorm_Detailed_Capture.txt")
        end)
        
        print(capture)
        
        return original_reroll()
    end
    
    print("[Detailed Capture] Hooked - Press Ctrl+R to capture")
end

return true