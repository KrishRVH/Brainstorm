local ImmolateIntegration = {}

local success, immolate_lib = pcall(require, "immolate_ffi")

if not success then
    print("[Brainstorm] Failed to load new Immolate library: " .. tostring(immolate_lib))
    print("[Brainstorm] Falling back to legacy implementation")
    ImmolateIntegration.available = false
else
    ImmolateIntegration.available = true
    ImmolateIntegration.lib = immolate_lib
end

function ImmolateIntegration.test_seed(seed, config)
    if not ImmolateIntegration.available then
        return nil
    end
    
    local brainstorm_config = {
        erratic_deck = config.deck_type == "erratic",
        no_faces = config.no_faces,
        face_cards = {},
        suit_ratio = {},
        vouchers = {},
        tags = {}
    }
    
    if config.face_cards_enabled then
        brainstorm_config.face_cards.min = config.face_cards_min or 0
        brainstorm_config.face_cards.max = config.face_cards_max or 52
    end
    
    if config.suit_ratio_enabled then
        brainstorm_config.suit_ratio.min = (config.suit_ratio_min or 0) / 100
        brainstorm_config.suit_ratio.max = (config.suit_ratio_max or 100) / 100
        brainstorm_config.suit_ratio.target = config.suit_ratio_target
    end
    
    if config.voucher_enabled then
        for voucher_key, enabled in pairs(config.vouchers or {}) do
            if enabled then
                table.insert(brainstorm_config.vouchers, voucher_key)
            end
        end
    end
    
    if config.tag_enabled then
        for tag_key, enabled in pairs(config.tags or {}) do
            if enabled then
                table.insert(brainstorm_config.tags, tag_key)
            end
        end
    end
    
    local ffi_config = ImmolateIntegration.lib.configure_for_brainstorm(brainstorm_config)
    local result = ImmolateIntegration.lib.test_single_seed(seed, ffi_config)
    
    if result then
        return {
            matches = true,
            face_count = result.face_count,
            suit_counts = result.suit_counts,
            max_suit_ratio = result.max_suit_ratio * 100,
            vouchers = result.vouchers,
            tags = result.tags
        }
    end
    
    return {matches = false}
end

function ImmolateIntegration.batch_test(start_seed, batch_size, config, max_seeds)
    if not ImmolateIntegration.available then
        return nil
    end
    
    local brainstorm_config = {
        erratic_deck = config.deck_type == "erratic",
        no_faces = config.no_faces,
        face_cards = {},
        suit_ratio = {},
        vouchers = {},
        tags = {}
    }
    
    if config.face_cards_enabled then
        brainstorm_config.face_cards.min = config.face_cards_min or 0
        brainstorm_config.face_cards.max = config.face_cards_max or 52
    end
    
    if config.suit_ratio_enabled then
        brainstorm_config.suit_ratio.min = (config.suit_ratio_min or 0) / 100
        brainstorm_config.suit_ratio.max = (config.suit_ratio_max or 100) / 100
        brainstorm_config.suit_ratio.target = config.suit_ratio_target
    end
    
    if config.voucher_enabled then
        for voucher_key, enabled in pairs(config.vouchers or {}) do
            if enabled then
                table.insert(brainstorm_config.vouchers, voucher_key)
            end
        end
    end
    
    if config.tag_enabled then
        for tag_key, enabled in pairs(config.tags or {}) do
            if enabled then
                table.insert(brainstorm_config.tags, tag_key)
            end
        end
    end
    
    local ffi_config = ImmolateIntegration.lib.configure_for_brainstorm(brainstorm_config)
    
    local results = {}
    local current_seed = start_seed
    local total_tested = 0
    
    while total_tested < max_seeds do
        local batch_results = ImmolateIntegration.lib.test_seeds(
            current_seed, 
            math.min(batch_size, max_seeds - total_tested), 
            ffi_config
        )
        
        for _, result in ipairs(batch_results) do
            table.insert(results, {
                seed = result.seed,
                face_count = result.face_count,
                suit_counts = result.suit_counts,
                max_suit_ratio = result.max_suit_ratio * 100,
                vouchers = result.vouchers,
                tags = result.tags
            })
        end
        
        total_tested = total_tested + batch_size
        current_seed = current_seed + batch_size
        
        if #results > 0 then
            break
        end
    end
    
    return results
end

function ImmolateIntegration.get_performance_estimate(config)
    if not ImmolateIntegration.available then
        return 100
    end
    
    if config.deck_type == "erratic" then
        if config.face_cards_enabled or config.suit_ratio_enabled then
            return 50000
        else
            return 100000
        end
    else
        return 200000
    end
end

function ImmolateIntegration.test_performance()
    if not ImmolateIntegration.available then
        print("[Brainstorm] Immolate library not available")
        return
    end
    
    print("[Brainstorm] Testing Immolate performance...")
    ImmolateIntegration.lib.test_performance()
end

return ImmolateIntegration