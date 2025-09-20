-- Simple RNG capture that runs when we do a reroll
-- This hooks into the existing Brainstorm reroll functionality

local function capture_rng_values()
    if not G or not G.GAME or not G.GAME.pseudorandom then
        return "Game not ready"
    end
    
    local output = "=== RNG CAPTURE (via Reroll) ===\n"
    output = output .. os.date("%Y-%m-%d %H:%M:%S") .. "\n\n"
    
    -- Current state
    output = output .. "Current Game State:\n"
    output = output .. string.format("  Seed: %s\n", tostring(G.GAME.pseudorandom.seed))
    output = output .. string.format("  Hashed: %.17g\n", G.GAME.pseudorandom.hashed or 0)
    output = output .. "\n"
    
    -- Test pseudohash if available
    if pseudohash then
        output = output .. "pseudohash tests:\n"
        
        local test_a = pseudohash("AAAAAAAA")
        output = output .. string.format("  pseudohash('AAAAAAAA') = %.17g\n", test_a)
        output = output .. "    Expected: 0.43257138351543745\n"
        output = output .. string.format("    Match: %s\n", tostring(math.abs(test_a - 0.43257138351543745) < 1e-15))
        
        local test_7 = pseudohash("7NTPKW6P") 
        output = output .. string.format("  pseudohash('7NTPKW6P') = %.17g\n", test_7)
        
        local test_0 = pseudohash("00000000")
        output = output .. string.format("  pseudohash('00000000') = %.17g\n", test_0)
    else
        output = output .. "pseudohash NOT AVAILABLE\n"
    end
    
    output = output .. "\n"
    
    -- Test pseudoseed with known seed
    if pseudoseed and pseudohash then
        output = output .. "pseudoseed tests with AAAAAAAA:\n"
        
        -- Save current state
        local saved_seed = G.GAME.pseudorandom.seed
        local saved_hashed = G.GAME.pseudorandom.hashed
        
        -- Set test seed
        G.GAME.pseudorandom.seed = "AAAAAAAA"
        G.GAME.pseudorandom.hashed = pseudohash("AAAAAAAA")
        
        -- Test values
        local voucher = pseudoseed("Voucher")
        output = output .. string.format("  pseudoseed('Voucher') = %.17g\n", voucher)
        output = output .. "    Expected: 0.46530388624939389\n"
        output = output .. string.format("    Match: %s\n", tostring(math.abs(voucher - 0.46530388624939389) < 1e-15))
        
        local pack1 = pseudoseed("shop_pack1")
        output = output .. string.format("  pseudoseed('shop_pack1') #1 = %.17g\n", pack1)
        output = output .. "    Expected: 0.60309655729733147\n"
        
        local pack2 = pseudoseed("shop_pack1")
        output = output .. string.format("  pseudoseed('shop_pack1') #2 = %.17g\n", pack2)
        output = output .. "    Expected: 0.45049515502425352\n"
        
        -- Restore state
        G.GAME.pseudorandom.seed = saved_seed
        G.GAME.pseudorandom.hashed = saved_hashed
    else
        output = output .. "pseudoseed NOT AVAILABLE\n"
    end
    
    return output
end

-- Hook into Brainstorm's reroll function
if Brainstorm and Brainstorm.reroll then
    local original_reroll = Brainstorm.reroll
    Brainstorm.reroll = function()
        -- Capture RNG values before reroll
        local capture = capture_rng_values()
        
        -- Try to save capture
        pcall(function()
            love.filesystem.write("Brainstorm_RNG_Capture.txt", capture)
            print("[RNG Capture] Saved to Brainstorm_RNG_Capture.txt")
        end)
        
        -- Also print to console
        print(capture)
        
        -- Do the original reroll
        return original_reroll()
    end
    
    print("[RNG Capture] Hooked into Brainstorm.reroll() - Press Ctrl+R to capture RNG values")
end

return true