-- Manual RNG capture that can be triggered
local ManualCapture = {}

function ManualCapture.capture_now()
    local output = "=== MANUAL RNG CAPTURE ===\n"
    output = output .. "Timestamp: " .. os.date("%Y-%m-%d %H:%M:%S") .. "\n\n"
    
    -- Check what exists
    output = output .. "Environment Check:\n"
    output = output .. "  pseudohash exists: " .. tostring(pseudohash ~= nil) .. "\n"
    output = output .. "  pseudoseed exists: " .. tostring(pseudoseed ~= nil) .. "\n"
    
    if G and G.GAME and G.GAME.pseudorandom then
        output = output .. "  Current seed: " .. tostring(G.GAME.pseudorandom.seed) .. "\n"
        output = output .. "  Current hashed: " .. tostring(G.GAME.pseudorandom.hashed) .. "\n"
    end
    
    output = output .. "\n"
    
    -- Test with known seeds
    if pseudohash then
        output = output .. "Testing pseudohash:\n"
        local hash_a = pseudohash("AAAAAAAA")
        output = output .. string.format("  pseudohash('AAAAAAAA') = %.17g\n", hash_a)
        output = output .. "    Expected: 0.43257138351543745\n"
        
        local hash_7 = pseudohash("7NTPKW6P")
        output = output .. string.format("  pseudohash('7NTPKW6P') = %.17g\n", hash_7)
        
        local hash_0 = pseudohash("00000000")
        output = output .. string.format("  pseudohash('00000000') = %.17g\n", hash_0)
    end
    
    output = output .. "\n"
    
    -- Test pseudoseed
    if pseudoseed and pseudohash and G and G.GAME and G.GAME.pseudorandom then
        -- Save state
        local old_seed = G.GAME.pseudorandom.seed
        local old_hashed = G.GAME.pseudorandom.hashed
        
        -- Test AAAAAAAA
        G.GAME.pseudorandom.seed = "AAAAAAAA"
        G.GAME.pseudorandom.hashed = pseudohash("AAAAAAAA")
        
        output = output .. "Testing pseudoseed with AAAAAAAA:\n"
        
        local v = pseudoseed("Voucher")
        output = output .. string.format("  pseudoseed('Voucher') = %.17g\n", v)
        output = output .. "    Expected: 0.46530388624939389\n"
        
        local p1 = pseudoseed("shop_pack1")
        output = output .. string.format("  pseudoseed('shop_pack1') first = %.17g\n", p1)
        output = output .. "    Expected: 0.60309655729733147\n"
        
        local p2 = pseudoseed("shop_pack1")
        output = output .. string.format("  pseudoseed('shop_pack1') second = %.17g\n", p2)
        output = output .. "    Expected: 0.45049515502425352\n"
        
        local ts = pseudoseed("Tag_small")
        output = output .. string.format("  pseudoseed('Tag_small') = %.17g\n", ts)
        output = output .. "    Expected: 0.45670112024272935\n"
        
        local tb = pseudoseed("Tag_big")
        output = output .. string.format("  pseudoseed('Tag_big') = %.17g\n", tb)
        output = output .. "    Expected: 0.50626440489001245\n"
        
        output = output .. "\n"
        
        -- Test 7NTPKW6P
        G.GAME.pseudorandom.seed = "7NTPKW6P"
        G.GAME.pseudorandom.hashed = pseudohash("7NTPKW6P")
        
        output = output .. "Testing pseudoseed with 7NTPKW6P:\n"
        
        v = pseudoseed("Voucher")
        output = output .. string.format("  pseudoseed('Voucher') = %.17g\n", v)
        
        p1 = pseudoseed("shop_pack1")
        output = output .. string.format("  pseudoseed('shop_pack1') first = %.17g\n", p1)
        
        p2 = pseudoseed("shop_pack1")
        output = output .. string.format("  pseudoseed('shop_pack1') second = %.17g\n", p2)
        
        -- Restore state
        G.GAME.pseudorandom.seed = old_seed
        G.GAME.pseudorandom.hashed = old_hashed
    end
    
    return output
end

-- Export as global function
_G.capture_rng = function()
    local result = ManualCapture.capture_now()
    print(result)
    
    -- Try to save to file
    local success = pcall(function()
        love.filesystem.write("Brainstorm_Manual_Capture.txt", result)
    end)
    
    if success then
        print("[ManualCapture] Saved to Brainstorm_Manual_Capture.txt")
    else
        print("[ManualCapture] Could not save to file")
    end
    
    return result
end

return ManualCapture