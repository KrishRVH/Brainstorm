-- CRITICAL FIX for Brainstorm.lua
-- Replace lines 935-999 with this code

  -- Detect which DLL version we have based on function signature
  local using_enhanced_dll = false
  
  -- Try calling with 8 parameters (enhanced DLL)
  local enhanced_success, enhanced_result = pcall(function()
    return immolate.brainstorm(
      seed_found,
      voucher_name,
      pack_name, 
      tag_name,
      tag2_name,  -- 8th parameter - enhanced DLL
      Brainstorm.config.ar_filters.soul_skip,
      Brainstorm.config.ar_filters.inst_observatory,
      Brainstorm.config.ar_filters.inst_perkeo
    )
  end)
  
  local result = nil
  
  if enhanced_success then
    -- Enhanced DLL worked!
    using_enhanced_dll = true
    result = enhanced_result
    if tag2_name ~= "" then
      print("[Brainstorm] Using enhanced DLL for dual tag search (fast mode)")
    end
  else
    -- Fall back to old DLL with 7 parameters
    print("[Brainstorm] Enhanced DLL not detected, using compatibility mode")
    
    if tag2_name ~= "" then
      -- Old DLL can't handle dual tags properly
      print("[Brainstorm] WARNING: Dual tag search will be slow with old DLL")
      print("[Brainstorm] Install Immolate_new.dll for 10-100x faster searches")
      
      -- For old DLL, just check for first tag
      local call_success, call_result = pcall(function()
        return immolate.brainstorm(
          seed_found,
          voucher_name,
          pack_name,
          tag_name,  -- Only check first tag
          Brainstorm.config.ar_filters.soul_skip,
          Brainstorm.config.ar_filters.inst_observatory,
          Brainstorm.config.ar_filters.inst_perkeo
        )
      end)
      
      if call_success then
        result = call_result
        -- Will need to validate second tag after restart
      else
        print("[Brainstorm] DLL call failed: " .. tostring(call_result))
      end
    else
      -- Single tag - works with old DLL
      local call_success, call_result = pcall(function()
        return immolate.brainstorm(
          seed_found,
          voucher_name,
          pack_name,
          tag_name,
          Brainstorm.config.ar_filters.soul_skip,
          Brainstorm.config.ar_filters.inst_observatory,
          Brainstorm.config.ar_filters.inst_perkeo
        )
      end)
      
      if call_success then
        result = call_result
      else
        print("[Brainstorm] DLL call failed: " .. tostring(call_result))
      end
    end
  end