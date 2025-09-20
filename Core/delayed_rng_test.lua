-- Delayed RNG test that waits for game to be ready
local DelayedRNGTest = {}

DelayedRNGTest.tested = false
DelayedRNGTest.test_count = 0

function DelayedRNGTest.run_test()
    if DelayedRNGTest.tested then
        return
    end
    
    -- Check if game is ready
    if not G or not G.GAME or not G.GAME.pseudorandom then
        DelayedRNGTest.test_count = DelayedRNGTest.test_count + 1
        if DelayedRNGTest.test_count < 100 then
            return -- Still waiting
        end
    end
    
    DelayedRNGTest.tested = true
    
    local output = "=== DELAYED RNG TEST RESULTS ===\n"
    output = output .. "Timestamp: " .. os.date("%Y-%m-%d %H:%M:%S") .. "\n"
    output = output .. "Test attempt count: " .. DelayedRNGTest.test_count .. "\n\n"
    
    -- Test current game state
    output = output .. "Game State:\n"
    output = output .. "  G exists: " .. tostring(G ~= nil) .. "\n"
    output = output .. "  G.GAME exists: " .. tostring(G and G.GAME ~= nil) .. "\n"
    output = output .. "  pseudohash exists: " .. tostring(pseudohash ~= nil) .. "\n"
    output = output .. "  pseudoseed exists: " .. tostring(pseudoseed ~= nil) .. "\n"
    
    if G and G.GAME and G.GAME.pseudorandom then
        output = output .. "  Current seed: " .. tostring(G.GAME.pseudorandom.seed) .. "\n"
        output = output .. "  Current hashed: " .. tostring(G.GAME.pseudorandom.hashed) .. "\n"
    end
    
    output = output .. "\n"
    
    -- Test pseudohash
    if pseudohash then
        output = output .. "Testing pseudohash:\n"
        
        local tests = {
            {"AAAAAAAA", 0.43257138351543745},
            {"00000000", nil},
            {"7NTPKW6P", nil}
        }
        
        for _, test in ipairs(tests) do
            local seed, expected = test[1], test[2]
            local result = pseudohash(seed)
            output = output .. string.format("  pseudohash('%s') = %.17g\n", seed, result)
            if expected then
                output = output .. string.format("    Expected: %.17g\n", expected)
                local match = math.abs(result - expected) < 1e-15
                output = output .. "    Match: " .. tostring(match) .. "\n"
            end
        end
    else
        output = output .. "pseudohash NOT FOUND\n"
    end
    
    output = output .. "\n"
    
    -- Test pseudoseed with specific seed
    if pseudoseed and pseudohash and G and G.GAME and G.GAME.pseudorandom then
        output = output .. "Testing pseudoseed with AAAAAAAA:\n"
        
        -- Save current state
        local old_seed = G.GAME.pseudorandom.seed
        local old_hashed = G.GAME.pseudorandom.hashed
        
        -- Set test seed
        G.GAME.pseudorandom.seed = "AAAAAAAA"
        G.GAME.pseudorandom.hashed = pseudohash("AAAAAAAA")
        
        local tests = {
            {"Voucher", 0.46530388624939389},
            {"shop_pack1", 0.60309655729733147},
            {"shop_pack1", 0.45049515502425352},
            {"Tag_small", 0.45670112024272935},
            {"Tag_big", 0.50626440489001245}
        }
        
        for i, test in ipairs(tests) do
            local key, expected = test[1], test[2]
            local result = pseudoseed(key)
            local label = key
            if i == 3 then label = key .. " (2nd call)" end
            output = output .. string.format("  pseudoseed('%s') = %.17g\n", label, result)
            output = output .. string.format("    Expected: %.17g\n", expected)
            local match = math.abs(result - expected) < 1e-15
            output = output .. "    Match: " .. tostring(match) .. "\n"
        end
        
        -- Restore state
        G.GAME.pseudorandom.seed = old_seed
        G.GAME.pseudorandom.hashed = old_hashed
    else
        output = output .. "Cannot test pseudoseed - missing dependencies\n"
    end
    
    -- Try to write output multiple ways
    print("[DelayedRNGTest] Test complete, attempting to save...")
    print(output)
    
    -- Method 1: love.filesystem
    local success1 = pcall(function()
        love.filesystem.write("Brainstorm_Delayed_Test.txt", output)
    end)
    
    -- Method 2: nativefs
    local success2 = pcall(function()
        local nfs = require("nativefs")
        nfs.write("Mods/Brainstorm/delayed_test.txt", output)
    end)
    
    -- Method 3: Direct file write
    local success3 = pcall(function()
        local file = io.open("delayed_rng_test.txt", "w")
        if file then
            file:write(output)
            file:close()
        end
    end)
    
    print("[DelayedRNGTest] Save attempts - love.fs: " .. tostring(success1) .. 
          ", nativefs: " .. tostring(success2) .. ", io: " .. tostring(success3))
end

-- Hook into update cycle
function DelayedRNGTest.install()
    -- Try to hook into the game update
    if G and G.FUNCS then
        local old_update = G.FUNCS.common_UIBox_hover
        G.FUNCS.common_UIBox_hover = function(e)
            DelayedRNGTest.run_test()
            if old_update then
                return old_update(e)
            end
        end
        print("[DelayedRNGTest] Hooked into common_UIBox_hover")
    end
    
    -- Also try hooking into skip_blind
    if G and G.FUNCS and G.FUNCS.skip_blind then
        local old_skip = G.FUNCS.skip_blind  
        G.FUNCS.skip_blind = function(e)
            DelayedRNGTest.run_test()
            if old_skip then
                return old_skip(e)
            end
        end
        print("[DelayedRNGTest] Hooked into skip_blind")
    end
end

return DelayedRNGTest