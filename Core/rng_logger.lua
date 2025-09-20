-- RNG Logger for Brainstorm
-- Automatically logs RNG values when entering shop

local RNGLogger = {}

-- Store RNG values in a table that gets saved to config
RNGLogger.data = {}

-- Hook into shop generation to capture RNG values
function RNGLogger.install()
    -- Hook when entering shop
    local old_shop_enter = G.FUNCS.skip_shop
    G.FUNCS.skip_shop = function(e)
        RNGLogger.capture_shop_rng()
        if old_shop_enter then
            return old_shop_enter(e)
        end
    end
    
    -- Also hook when shop is created
    local old_create_shop = create_shop
    if old_create_shop then
        create_shop = function(...)
            RNGLogger.capture_shop_rng()
            return old_create_shop(...)
        end
    end
    
    -- Hook the shop voucher generation
    local old_get_next_voucher = get_next_voucher_key
    if old_get_next_voucher then
        get_next_voucher_key = function(...)
            local result = old_get_next_voucher(...)
            RNGLogger.log_event("voucher_selected", result)
            return result
        end
    end
end

-- Capture current RNG state
function RNGLogger.capture_shop_rng()
    if not G or not G.GAME or not G.GAME.pseudorandom then
        return
    end
    
    local seed = G.GAME.pseudorandom.seed or "UNKNOWN"
    local hashed = G.GAME.pseudorandom.hashed or 0
    
    -- Create a log entry
    local entry = {
        timestamp = os.time(),
        seed = seed,
        hashed = hashed,
        tests = {}
    }
    
    -- Test pseudohash values
    if pseudohash then
        entry.tests.hash_AAAAAAAA = pseudohash("AAAAAAAA")
        entry.tests.hash_00000000 = pseudohash("00000000")
        entry.tests.hash_7NTPKW6P = pseudohash("7NTPKW6P")
    end
    
    -- Test pseudoseed values (save and restore state)
    if pseudoseed then
        local old_state = {}
        for k, v in pairs(G.GAME.pseudorandom) do
            old_state[k] = v
        end
        
        -- Test with AAAAAAAA
        G.GAME.pseudorandom = {seed = "AAAAAAAA", hashed = pseudohash("AAAAAAAA")}
        entry.tests.AAAAAAAA = {
            voucher = pseudoseed("Voucher"),
            pack1 = pseudoseed("shop_pack1"),
            pack2 = pseudoseed("shop_pack1"),
            tag_small = pseudoseed("Tag_small"),
            tag_big = pseudoseed("Tag_big")
        }
        
        -- Test with 7NTPKW6P
        G.GAME.pseudorandom = {seed = "7NTPKW6P", hashed = pseudohash("7NTPKW6P")}
        entry.tests["7NTPKW6P"] = {
            voucher = pseudoseed("Voucher"),
            pack1 = pseudoseed("shop_pack1"),
            pack2 = pseudoseed("shop_pack1"),
            tag_small = pseudoseed("Tag_small"),
            tag_big = pseudoseed("Tag_big")
        }
        
        -- Restore state
        G.GAME.pseudorandom = old_state
    end
    
    -- Add to data
    table.insert(RNGLogger.data, entry)
    
    -- Save to file
    RNGLogger.save()
    
    -- Also display as alert
    attention_text({
        text = "RNG Logged!",
        scale = 0.5,
        hold = 2,
        major = G.play
    })
end

-- Log specific event
function RNGLogger.log_event(event_type, value)
    local entry = {
        timestamp = os.time(),
        event = event_type,
        value = value,
        seed = G.GAME.pseudorandom and G.GAME.pseudorandom.seed or "UNKNOWN"
    }
    table.insert(RNGLogger.data, entry)
end

-- Save data to config file
function RNGLogger.save()
    if not RNGLogger.data or #RNGLogger.data == 0 then
        return
    end
    
    -- Convert to string format
    local output = "RNG LOG DATA\n"
    output = output .. "============\n\n"
    
    for i, entry in ipairs(RNGLogger.data) do
        output = output .. string.format("Entry %d (seed: %s):\n", i, entry.seed or "?")
        
        if entry.hashed then
            output = output .. string.format("  hashed = %.17g\n", entry.hashed)
        end
        
        if entry.tests then
            -- Hash tests
            if entry.tests.hash_AAAAAAAA then
                output = output .. string.format("  pseudohash('AAAAAAAA') = %.17g\n", entry.tests.hash_AAAAAAAA)
                output = output .. "    Expected: 0.43257138351543745\n"
            end
            if entry.tests.hash_00000000 then
                output = output .. string.format("  pseudohash('00000000') = %.17g\n", entry.tests.hash_00000000)
            end
            if entry.tests.hash_7NTPKW6P then
                output = output .. string.format("  pseudohash('7NTPKW6P') = %.17g\n", entry.tests.hash_7NTPKW6P)
            end
            
            -- Seed tests
            if entry.tests.AAAAAAAA then
                output = output .. "\n  Test AAAAAAAA:\n"
                output = output .. string.format("    pseudoseed('Voucher') = %.17g\n", entry.tests.AAAAAAAA.voucher)
                output = output .. "      Expected: 0.46530388624939389\n"
                output = output .. string.format("    pseudoseed('shop_pack1') = %.17g\n", entry.tests.AAAAAAAA.pack1)
                output = output .. "      Expected: 0.60309655729733147\n"
                output = output .. string.format("    pseudoseed('shop_pack1') #2 = %.17g\n", entry.tests.AAAAAAAA.pack2)
                output = output .. "      Expected: 0.45049515502425352\n"
                output = output .. string.format("    pseudoseed('Tag_small') = %.17g\n", entry.tests.AAAAAAAA.tag_small)
                output = output .. string.format("    pseudoseed('Tag_big') = %.17g\n", entry.tests.AAAAAAAA.tag_big)
            end
            
            if entry.tests["7NTPKW6P"] then
                output = output .. "\n  Test 7NTPKW6P:\n"
                output = output .. string.format("    pseudoseed('Voucher') = %.17g\n", entry.tests["7NTPKW6P"].voucher)
                output = output .. string.format("    pseudoseed('shop_pack1') = %.17g\n", entry.tests["7NTPKW6P"].pack1)
                output = output .. string.format("    pseudoseed('shop_pack1') #2 = %.17g\n", entry.tests["7NTPKW6P"].pack2)
                output = output .. string.format("    pseudoseed('Tag_small') = %.17g\n", entry.tests["7NTPKW6P"].tag_small)
                output = output .. string.format("    pseudoseed('Tag_big') = %.17g\n", entry.tests["7NTPKW6P"].tag_big)
            end
        end
        
        output = output .. "\n"
    end
    
    -- Save to Brainstorm config directory
    love.filesystem.write("Brainstorm_RNG_Log.txt", output)
    
    -- Also try to save to mod directory
    local nfs = require("nativefs")
    if nfs then
        local path = "Mods/Brainstorm/rng_log.txt"
        nfs.write(path, output)
    end
end

-- Get the log as a string
function RNGLogger.get_log()
    local path = "Brainstorm_RNG_Log.txt"
    if love.filesystem.getInfo(path) then
        return love.filesystem.read(path)
    end
    return "No log file found"
end

-- Clear the log
function RNGLogger.clear()
    RNGLogger.data = {}
    love.filesystem.remove("Brainstorm_RNG_Log.txt")
end

return RNGLogger