-- Standalone RNG Tracer for Balatro
-- This file can be run directly without dependencies

local trace_file = nil
local trace_path = "C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm\\rng_trace.jsonl"

-- Helper to format doubles with full precision
local function f(x)
    if x == nil then return "null" end
    return string.format("%.17g", x)
end

-- Helper to escape strings for JSON
local function escape_json(str)
    if not str then return "" end
    return str:gsub('"', '\\"'):gsub('\n', '\\n'):gsub('\r', '\\r')
end

-- Open trace file
local function init_trace()
    trace_file = io.open(trace_path, "a")
    if trace_file then
        trace_file:write("\n-- NEW TRACE SESSION --\n")
        trace_file:flush()
        print("[TRACE] File opened: " .. trace_path)
        return true
    end
    print("[TRACE] Failed to open file: " .. trace_path)
    return false
end

-- Append to trace
local function append_trace(line)
    if trace_file then
        trace_file:write(line)
        trace_file:flush()
    end
end

-- Close trace file
local function close_trace()
    if trace_file then
        trace_file:close()
        trace_file = nil
    end
end

-- Main tracing function
local function install_traces()
    print("[TRACE] Installing RNG traces...")
    
    if not init_trace() then
        return false
    end
    
    -- Check if game is loaded
    if not G or not G.GAME or not G.GAME.pseudorandom then
        print("[TRACE] ERROR: Game not initialized")
        close_trace()
        return false
    end
    
    -- Save original functions
    local _pseudohash = pseudohash
    local _pseudoseed = pseudoseed
    
    if not _pseudohash or not _pseudoseed then
        print("[TRACE] ERROR: RNG functions not found")
        close_trace()
        return false
    end
    
    -- Patch pseudohash
    _G.pseudohash = function(str)
        local result = _pseudohash(str)
        local line = string.format(
            '{"type":"pseudohash","input":"%s","output":%s}\n',
            escape_json(str),
            f(result)
        )
        append_trace(line)
        return result
    end
    
    -- Patch pseudoseed
    _G.pseudoseed = function(key)
        local pr = G.GAME.pseudorandom
        local first = pr[key] == nil
        local before = first and "null" or f(pr[key])
        
        local ret = _pseudoseed(key)
        
        local after = f(pr[key])
        local hashed = f(pr.hashed)
        local seed = pr.seed or "unknown"
        
        local line = string.format(
            '{"type":"pseudoseed","key":"%s","seed":"%s","first":%s,"before":%s,"after":%s,"ret":%s,"hashed":%s}\n',
            escape_json(key),
            escape_json(seed),
            tostring(first),
            before,
            after,
            f(ret),
            hashed
        )
        append_trace(line)
        
        return ret
    end
    
    print("[TRACE] Patches installed successfully")
    
    -- Generate test traces
    print("[TRACE] Generating test traces...")
    
    local test_seeds = {"AAAAAAAA", "00000000", "7NTPKW6P", "ZZZZZZZZ"}
    
    for _, test_seed in ipairs(test_seeds) do
        print("[TRACE] Testing seed: " .. test_seed)
        
        -- Save current state
        local old_seed = G.GAME.pseudorandom.seed
        local old_state = {}
        for k, v in pairs(G.GAME.pseudorandom) do
            old_state[k] = v
        end
        
        -- Set test seed
        G.GAME.pseudorandom = {seed = test_seed, hashed = pseudohash(test_seed)}
        
        append_trace(string.format('{"type":"test_seed_start","seed":"%s"}\n', test_seed))
        
        -- Test basic RNG calls
        local voucher_val = pseudoseed("Voucher")
        local pack1_val = pseudoseed("shop_pack1")
        local pack2_val = pseudoseed("shop_pack1")  -- Second call
        local tag_small_val = pseudoseed("Tag_small")
        local tag_big_val = pseudoseed("Tag_big")
        
        append_trace(string.format('{"type":"test_summary","seed":"%s","voucher":%s,"pack1":%s,"pack2":%s,"tag_small":%s,"tag_big":%s}\n',
            test_seed, f(voucher_val), f(pack1_val), f(pack2_val), f(tag_small_val), f(tag_big_val)))
        
        append_trace('{"type":"test_seed_end"}\n')
        
        -- Restore state
        G.GAME.pseudorandom = old_state
    end
    
    close_trace()
    
    print("[TRACE] ✓ Complete! Trace file saved to:")
    print("  " .. trace_path)
    
    return true
end

-- Run immediately when loaded
print("[TRACE] Starting RNG trace...")
local success = install_traces()

if success then
    print("[TRACE] SUCCESS - Check the trace file")
else
    print("[TRACE] FAILED - See errors above")
end

return success