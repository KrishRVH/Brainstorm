-- BrainstormPoolUpdate.lua
-- Exact pool update implementation for release-safe operation
-- Uses exact ctx_key from game for each slot, handles first-shop Buffoon

local ffi = require("ffi")

-- FFI declarations for pool and resume control
ffi.cdef[[
    void brainstorm_update_pools(const char* json_utf8);
    void brainstorm_reset_resume(const char* seed);
    void brainstorm_set_target_ms(uint32_t ms);
    uint64_t brainstorm_get_resume_index(void);
    void brainstorm_calibrate(void);
]]

local function to_flat_items_and_weights(pool)
    local items, weights = {}, {}
    local total = 0
    for i, e in ipairs(pool) do
        local key = e.key or e
        local w = e.weight or 1
        items[#items + 1] = key
        weights[#weights + 1] = w
        total = total + w
    end
    return items, weights, total
end

local function get_ctx(name)
    -- get_current_pool must return (pool, ctx_key) - the exact key the game uses
    local pool, key = get_current_pool(name)
    if not pool or not key then
        -- Fallback if get_current_pool not available
        return { ctx_key = name, items = {"default"}, weights = nil }
    end
    local items, weights = to_flat_items_and_weights(pool)
    return { ctx_key = key, items = items, weights = (#weights > 0) and weights or nil }
end

local function is_first_shop()
    -- Check if we're in the first shop of a run
    if G and G.GAME then
        -- Method 1: Check round number
        if G.GAME.round and G.GAME.round == 1 then 
            return true 
        end
        -- Method 2: Check ante
        if G.GAME.current_round and G.GAME.current_round.ante == 1 then 
            return true 
        end
        -- Method 3: Check shop visit count
        if G.GAME.shop_visits and G.GAME.shop_visits == 0 then
            return true
        end
    end
    return false
end

local function force_buffoon_in_one_slot(ctx_pack1, ctx_pack2)
    -- First shop guarantees a Buffoon pack in one slot
    local function find_buffoon_index(items)
        for i, k in ipairs(items) do
            -- Check various possible keys for Buffoon pack
            if k == "Buffoon Pack" or 
               k == "p_buffoon" or 
               k == "p_buffoon_1" or
               k:match("[Bb]uffoon") then
                return i
            end
        end
        return nil
    end
    
    local i1 = find_buffoon_index(ctx_pack1.items)
    local i2 = find_buffoon_index(ctx_pack2.items)
    
    if i1 then
        -- Force pack1 to only have Buffoon
        ctx_pack1.items = { ctx_pack1.items[i1] }
        ctx_pack1.weights = nil
        Brainstorm.log:debug("First shop: forcing pack1 to Buffoon")
    elseif i2 then
        -- Force pack2 to only have Buffoon
        ctx_pack2.items = { ctx_pack2.items[i2] }
        ctx_pack2.weights = nil
        Brainstorm.log:debug("First shop: forcing pack2 to Buffoon")
    else
        -- Buffoon not found - log warning but continue
        if Brainstorm and Brainstorm.log then
            Brainstorm.log:warn("First shop: Buffoon pack not found in pools")
        end
    end
end

function Brainstorm.update_pools_from_game()
    if not immolate then
        if Brainstorm and Brainstorm.log then
            Brainstorm.log:error("immolate DLL not loaded")
        end
        return
    end
    
    local contexts = {}
    
    -- Get exact context keys from the game for each slot
    -- The game should return the exact strings it uses internally
    
    -- Voucher context
    contexts.voucher = get_ctx("Voucher")
    if Brainstorm and Brainstorm.log then
        Brainstorm.log:debug("Voucher ctx_key: '%s'", contexts.voucher.ctx_key)
    end
    
    -- Pack contexts - use exact slot identifiers
    -- These may be "shop_pack_1"/"shop_pack_2" or similar
    -- CRITICAL: Do not hardcode these - get from game
    local pack1 = get_ctx("shop_pack_1")
    local pack2 = get_ctx("shop_pack_2")
    
    -- Log the exact keys we got
    if Brainstorm and Brainstorm.log then
        Brainstorm.log:debug("Pack1 ctx_key: '%s'", pack1.ctx_key)
        Brainstorm.log:debug("Pack2 ctx_key: '%s'", pack2.ctx_key)
    end
    
    contexts.pack1 = pack1
    contexts.pack2 = pack2
    
    -- Tag contexts
    contexts.tag_small = get_ctx("Tag_small")
    contexts.tag_big = get_ctx("Tag_big")
    
    if Brainstorm and Brainstorm.log then
        Brainstorm.log:debug("Tag_small ctx_key: '%s'", contexts.tag_small.ctx_key)
        Brainstorm.log:debug("Tag_big ctx_key: '%s'", contexts.tag_big.ctx_key)
    end
    
    -- Apply business rules
    
    -- 1. First-shop handling: force Buffoon in one slot
    if is_first_shop() then
        if Brainstorm and Brainstorm.log then
            Brainstorm.log:info("Applying first-shop Buffoon rule")
        end
        force_buffoon_in_one_slot(contexts.pack1, contexts.pack2)
    end
    
    -- 2. Voucher gating: remove locked vouchers
    -- TODO: Filter contexts.voucher.items based on unlock status
    
    -- 3. Pack gating: remove unavailable packs
    -- TODO: Filter pack items based on unlock status
    
    -- Build JSON
    local snap = { 
        version = 1, 
        contexts = contexts 
    }
    
    -- Simple JSON encoding (if dkjson not available)
    local function encode_simple(t)
        local function encode_value(v)
            if type(v) == "string" then
                return '"' .. v:gsub('"', '\\"') .. '"'
            elseif type(v) == "number" then
                return tostring(v)
            elseif type(v) == "boolean" then
                return v and "true" or "false"
            elseif type(v) == "table" then
                if #v > 0 then
                    -- Array
                    local parts = {}
                    for i, item in ipairs(v) do
                        parts[#parts + 1] = encode_value(item)
                    end
                    return "[" .. table.concat(parts, ",") .. "]"
                else
                    -- Object
                    local parts = {}
                    for k, val in pairs(v) do
                        parts[#parts + 1] = '"' .. k .. '":' .. encode_value(val)
                    end
                    return "{" .. table.concat(parts, ",") .. "}"
                end
            else
                return "null"
            end
        end
        return encode_value(t)
    end
    
    local json = encode_simple(snap)
    
    -- Log the JSON for debugging
    if Brainstorm and Brainstorm.log then
        Brainstorm.log:debug("Sending pools JSON: %s", json:sub(1, 200) .. "...")
    end
    
    -- Send to DLL
    immolate.brainstorm_update_pools(json)
end

-- Export functions for external use
return {
    update_pools_from_game = Brainstorm.update_pools_from_game,
    
    -- Calibration helper
    calibrate = function()
        if immolate then
            immolate.brainstorm_calibrate()
        end
    end,
    
    -- Resume control
    reset_resume = function(seed)
        if immolate then
            immolate.brainstorm_reset_resume(seed or "")
        end
    end,
    
    get_resume_index = function()
        if immolate then
            return tonumber(immolate.brainstorm_get_resume_index())
        end
        return 0
    end,
    
    set_target_ms = function(ms)
        if immolate then
            immolate.brainstorm_set_target_ms(ms or 250)
        end
    end,
    
    -- Search with resume support
    search_with_resume = function(start_seed, filters, max_iterations)
        if not immolate then return nil end
        
        max_iterations = max_iterations or 100
        
        -- Reset to start seed if provided
        if start_seed then
            immolate.brainstorm_reset_resume(start_seed)
        end
        
        for i = 1, max_iterations do
            -- Call with empty seed to use resume
            local result = immolate.brainstorm(
                "",  -- Empty seed means use resume
                filters.voucher or "",
                filters.pack or "",
                filters.tag1 or "",
                filters.tag2 or "",
                filters.souls or 0,
                filters.observatory or false,
                filters.perkeo or false
            )
            
            if result ~= nil and result ~= ffi.NULL then
                local seed_str = ffi.string(result)
                immolate.free_result(result)
                if seed_str ~= "RETRY" then
                    return seed_str
                end
            end
            
            -- Check if we've covered enough space
            local current_index = tonumber(immolate.brainstorm_get_resume_index())
            if current_index > 1000000000 then  -- 1B seeds
                break
            end
        end
        
        return nil
    end
}