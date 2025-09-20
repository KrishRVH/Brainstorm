-- RNG Tracing Module for Brainstorm
-- Captures golden test vectors from the game for RNG parity testing

local RNGTrace = {}

-- File handle for logging
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
function RNGTrace.init()
    trace_file = io.open(trace_path, "a")
    if trace_file then
        trace_file:write("\n-- NEW TRACE SESSION --\n")
        trace_file:flush()
    end
end

-- Close trace file
function RNGTrace.close()
    if trace_file then
        trace_file:close()
        trace_file = nil
    end
end

-- Append line to trace
function RNGTrace.append(line)
    if trace_file then
        trace_file:write(line)
        trace_file:flush()
    end
end

-- Install pseudohash trace
function RNGTrace.patch_pseudohash()
    if not pseudohash then return end
    
    local _pseudohash = pseudohash
    _G.pseudohash = function(str)
        local result = _pseudohash(str)
        
        -- Log the call
        local line = string.format(
            '{"type":"pseudohash","input":"%s","output":%s}\n',
            escape_json(str),
            f(result)
        )
        RNGTrace.append(line)
        
        return result
    end
end

-- Install pseudoseed trace
function RNGTrace.patch_pseudoseed()
    if not pseudoseed then return end
    
    local _pseudoseed = pseudoseed
    _G.pseudoseed = function(key)
        local pr = G.GAME.pseudorandom
        local first = pr[key] == nil
        local before = first and "null" or f(pr[key])
        
        -- Call original
        local ret = _pseudoseed(key)
        
        -- Get state after
        local after = f(pr[key])
        local hashed = f(pr.hashed)
        local seed = pr.seed or "unknown"
        
        -- Log the call
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
        RNGTrace.append(line)
        
        return ret
    end
end

-- Install pseudorandom_element trace
function RNGTrace.patch_pseudorandom_element()
    if not pseudorandom_element then return end
    
    local _pseudorandom_element = pseudorandom_element
    _G.pseudorandom_element = function(pool, seed_val)
        -- Capture pool info
        local sum_w = 0
        local items = {}
        
        if pool then
            for i, e in ipairs(pool) do
                local item_key = nil
                local weight = 1
                
                -- Handle different pool formats
                if type(e) == "table" then
                    item_key = e.key or e.name or tostring(e)
                    weight = e.weight or 1
                else
                    item_key = tostring(e)
                end
                
                sum_w = sum_w + weight
                items[#items + 1] = {key = item_key, weight = weight}
            end
        end
        
        -- Call original
        local sel = _pseudorandom_element(pool, seed_val)
        
        -- Find selected index
        local sel_idx = nil
        local sel_key = nil
        if pool then
            for i, e in ipairs(pool) do
                local check_key = nil
                if type(e) == "table" then
                    check_key = e.key or e.name or tostring(e)
                else
                    check_key = tostring(e)
                end
                
                if check_key == sel or e == sel then
                    sel_idx = i
                    sel_key = check_key
                    break
                end
            end
        end
        
        -- Log the call
        local line = string.format(
            '{"type":"choose","seed":%s,"sum_w":%s,"index":%s,"selected":"%s","pool_size":%d,"pool_key":"%s"}\n',
            f(seed_val),
            f(sum_w),
            sel_idx or "null",
            escape_json(sel_key or tostring(sel)),
            #items,
            escape_json(G._last_pool_key or "unknown")
        )
        RNGTrace.append(line)
        
        -- Also log full pool if small enough
        if #items <= 50 then
            local pool_line = '{"type":"pool_detail","pool_key":"' .. escape_json(G._last_pool_key or "unknown") .. '","items":['
            for i, item in ipairs(items) do
                if i > 1 then pool_line = pool_line .. "," end
                pool_line = pool_line .. string.format('{"key":"%s","weight":%s}', escape_json(item.key), f(item.weight))
            end
            pool_line = pool_line .. ']}\n'
            RNGTrace.append(pool_line)
        end
        
        return sel
    end
end

-- Install get_current_pool trace
function RNGTrace.patch_get_current_pool()
    if not get_current_pool then return end
    
    local _get_current_pool = get_current_pool
    _G.get_current_pool = function(_type, _rarity, _legendary, _append)
        -- Call original
        local pool, pool_key = _get_current_pool(_type, _rarity, _legendary, _append)
        
        -- Store for pseudorandom_element to use
        G._last_pool_key = pool_key or (_type and tostring(_type)) or "unknown"
        
        -- Log the pool request
        local line = string.format(
            '{"type":"get_pool","type":"%s","rarity":"%s","legendary":%s,"append":"%s","pool_key":"%s","pool_size":%d}\n',
            escape_json(_type or "nil"),
            escape_json(_rarity and tostring(_rarity) or "nil"),
            tostring(_legendary or false),
            escape_json(_append or "nil"),
            escape_json(G._last_pool_key),
            pool and #pool or 0
        )
        RNGTrace.append(line)
        
        return pool, pool_key
    end
end

-- Install get_pack trace
function RNGTrace.patch_get_pack()
    if not get_pack then return end
    
    local _get_pack = get_pack
    _G.get_pack = function(_key, _type)
        -- Store the key being used
        G._last_pack_key = _key or "pack_generic"
        
        -- Call original
        local result = _get_pack(_key, _type)
        
        -- Log the pack selection
        local line = string.format(
            '{"type":"get_pack","key":"%s","type":"%s","result_key":"%s"}\n',
            escape_json(_key or "nil"),
            escape_json(_type or "nil"),
            escape_json(result and result.key or "nil")
        )
        RNGTrace.append(line)
        
        return result
    end
end

-- Install get_next_voucher_key trace
function RNGTrace.patch_get_next_voucher_key()
    if not get_next_voucher_key then return end
    
    local _get_next_voucher_key = get_next_voucher_key
    _G.get_next_voucher_key = function(force_key, delayed_key)
        -- Call original
        local result = _get_next_voucher_key(force_key, delayed_key)
        
        -- Log the voucher selection
        local line = string.format(
            '{"type":"get_voucher","force_key":"%s","delayed_key":"%s","result":"%s"}\n',
            escape_json(force_key or "nil"),
            escape_json(delayed_key or "nil"),
            escape_json(result or "nil")
        )
        RNGTrace.append(line)
        
        return result
    end
end

-- Capture a complete shop generation trace
function RNGTrace.trace_shop_generation(seed)
    if not G or not G.GAME then
        RNGTrace.append('{"type":"error","msg":"Game not initialized"}\n')
        return
    end
    
    -- Set the seed
    local old_seed = G.GAME.pseudorandom.seed
    G.GAME.pseudorandom.seed = seed
    
    -- Clear pseudorandom state to get fresh generation
    G.GAME.pseudorandom = {seed = seed, hashed = pseudohash(seed)}
    
    -- Log start of shop trace
    local line = string.format(
        '{"type":"shop_start","seed":"%s","hashed":%s}\n',
        escape_json(seed),
        f(G.GAME.pseudorandom.hashed)
    )
    RNGTrace.append(line)
    
    -- Simulate shop generation (calling the actual game functions)
    if get_next_voucher_key then
        local voucher = get_next_voucher_key()
        RNGTrace.append(string.format('{"type":"shop_voucher","key":"%s"}\n', escape_json(voucher or "nil")))
    end
    
    -- Get packs (usually 2 in shop)
    if get_pack then
        local pack1 = get_pack('shop_pack1')
        if pack1 then
            RNGTrace.append(string.format('{"type":"shop_pack1","key":"%s"}\n', escape_json(pack1.key or "nil")))
        end
        
        local pack2 = get_pack('shop_pack2')
        if pack2 then
            RNGTrace.append(string.format('{"type":"shop_pack2","key":"%s"}\n', escape_json(pack2.key or "nil")))
        end
    end
    
    -- Get tags
    if get_next_tag_key then
        local tag_small = get_next_tag_key('Small')
        local tag_big = get_next_tag_key('Big')
        
        RNGTrace.append(string.format('{"type":"shop_tag_small","key":"%s"}\n', escape_json(tag_small or "nil")))
        RNGTrace.append(string.format('{"type":"shop_tag_big","key":"%s"}\n', escape_json(tag_big or "nil")))
    end
    
    -- Log end of shop trace
    RNGTrace.append('{"type":"shop_end"}\n')
    
    -- Restore original seed
    G.GAME.pseudorandom.seed = old_seed
end

-- Install all patches
function RNGTrace.install_all_patches()
    RNGTrace.init()
    RNGTrace.patch_pseudohash()
    RNGTrace.patch_pseudoseed()
    RNGTrace.patch_pseudorandom_element()
    RNGTrace.patch_get_current_pool()
    RNGTrace.patch_get_pack()
    RNGTrace.patch_get_next_voucher_key()
    
    RNGTrace.append('{"type":"patches_installed","timestamp":"' .. os.date() .. '"}\n')
end

-- Test function to generate traces for known seeds
function RNGTrace.generate_test_traces()
    local test_seeds = {"AAAAAAAA", "00000000", "7NTPKW6P", "ZZZZZZZZ"}
    
    for _, seed in ipairs(test_seeds) do
        RNGTrace.append(string.format('\n{"type":"test_seed_start","seed":"%s"}\n', seed))
        RNGTrace.trace_shop_generation(seed)
        RNGTrace.append('{"type":"test_seed_end"}\n')
    end
end

return RNGTrace