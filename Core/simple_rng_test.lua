-- Simple RNG test that writes directly to a file
local SimpleRNGTest = {}

function SimpleRNGTest.test_and_log()
    -- Test if we can access RNG functions
    local success, err = pcall(function()
        local output = "=== RNG TEST RESULTS ===\n"
        output = output .. "Timestamp: " .. os.date("%Y-%m-%d %H:%M:%S") .. "\n\n"
        
        -- Test 1: Check if functions exist
        output = output .. "Function existence:\n"
        output = output .. "  pseudohash exists: " .. tostring(pseudohash ~= nil) .. "\n"
        output = output .. "  pseudoseed exists: " .. tostring(pseudoseed ~= nil) .. "\n"
        output = output .. "  G exists: " .. tostring(G ~= nil) .. "\n"
        
        if G then
            output = output .. "  G.GAME exists: " .. tostring(G.GAME ~= nil) .. "\n"
            if G.GAME then
                output = output .. "  G.GAME.pseudorandom exists: " .. tostring(G.GAME.pseudorandom ~= nil) .. "\n"
                if G.GAME.pseudorandom then
                    output = output .. "  Current seed: " .. tostring(G.GAME.pseudorandom.seed) .. "\n"
                    output = output .. "  Current hashed: " .. tostring(G.GAME.pseudorandom.hashed) .. "\n"
                end
            end
        end
        
        output = output .. "\n"
        
        -- Test 2: Try to call pseudohash if it exists
        if pseudohash then
            output = output .. "pseudohash tests:\n"
            local test_seeds = {"AAAAAAAA", "00000000", "7NTPKW6P"}
            for _, seed in ipairs(test_seeds) do
                local hash = pseudohash(seed)
                output = output .. string.format("  pseudohash('%s') = %.17g\n", seed, hash)
            end
            output = output .. "\nExpected values:\n"
            output = output .. "  pseudohash('AAAAAAAA') = 0.43257138351543745\n"
        end
        
        output = output .. "\n"
        
        -- Test 3: Try pseudoseed if available
        if pseudoseed and G and G.GAME and G.GAME.pseudorandom then
            output = output .. "pseudoseed tests:\n"
            
            -- Save current state
            local old_seed = G.GAME.pseudorandom.seed
            local old_hashed = G.GAME.pseudorandom.hashed
            
            -- Test with AAAAAAAA
            if pseudohash then
                G.GAME.pseudorandom.seed = "AAAAAAAA"
                G.GAME.pseudorandom.hashed = pseudohash("AAAAAAAA")
                
                output = output .. "With seed AAAAAAAA:\n"
                local v1 = pseudoseed("Voucher")
                output = output .. string.format("  pseudoseed('Voucher') = %.17g\n", v1)
                output = output .. "    Expected: 0.46530388624939389\n"
                
                local p1 = pseudoseed("shop_pack1")
                output = output .. string.format("  pseudoseed('shop_pack1') #1 = %.17g\n", p1)
                output = output .. "    Expected: 0.60309655729733147\n"
                
                local p2 = pseudoseed("shop_pack1")
                output = output .. string.format("  pseudoseed('shop_pack1') #2 = %.17g\n", p2)
                output = output .. "    Expected: 0.45049515502425352\n"
            end
            
            -- Restore state
            G.GAME.pseudorandom.seed = old_seed
            G.GAME.pseudorandom.hashed = old_hashed
        end
        
        -- Write to file using nativefs
        local nfs = require("nativefs")
        local file_path = "Mods/Brainstorm/rng_test_results.txt"
        nfs.write(file_path, output)
        
        -- Also try love.filesystem
        love.filesystem.write("Brainstorm_RNG_Test.txt", output)
        
        return output
    end)
    
    if success then
        return err -- err contains the output string in this case
    else
        return "ERROR: " .. tostring(err)
    end
end

-- Auto-run the test when loaded
local result = SimpleRNGTest.test_and_log()
print("[SimpleRNGTest] Test completed. Check rng_test_results.txt")
print(result)

return SimpleRNGTest