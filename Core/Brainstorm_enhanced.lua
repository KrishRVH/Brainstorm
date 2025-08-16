-- Enhanced auto_reroll function that uses the new DLL
function Brainstorm.auto_reroll_enhanced()
  if not init_ffi() then
    print("[Brainstorm] FFI not initialized")
    return nil
  end
  
  local seed_found = ""
  for _ = 1, 8 do
    seed_found = seed_found
      .. string.char(
        ("%1234567890QWERTYUIOPASDFGHJKLZXCVBNM"):byte(
          love.math.random(1, 38),
          love.math.random(1, 38)
        )
      )
  end
  
  local success, immolate = pcall(ffi.load, Brainstorm.PATH .. "/Immolate.dll")
  if not success then
    print("[Brainstorm] Failed to load Immolate.dll: " .. tostring(immolate))
    return nil
  end
  
  -- Extract configuration
  local pack = ""
  if #Brainstorm.config.ar_filters.pack > 0 then
    pack = Brainstorm.config.ar_filters.pack[1]:match("^(.*)_") or ""
  end
  local pack_name = pack ~= ""
      and localize({ type = "name_text", set = "Other", key = pack })
    or ""
    
  local tag_name = localize({
    type = "name_text",
    set = "Tag",
    key = Brainstorm.config.ar_filters.tag_name,
  })
  
  local tag2_name = ""
  if Brainstorm.config.ar_filters.tag2_name and Brainstorm.config.ar_filters.tag2_name ~= "" then
    tag2_name = localize({
      type = "name_text",
      set = "Tag",
      key = Brainstorm.config.ar_filters.tag2_name,
    })
  end
  
  local voucher_name = localize({
    type = "name_text",
    set = "Voucher",
    key = Brainstorm.config.ar_filters.voucher_name,
  })
  
  -- Call the enhanced DLL function with both tags
  local call_success, call_result = pcall(function()
    return immolate.brainstorm(
      seed_found,
      voucher_name,
      pack_name,
      tag_name,
      tag2_name,  -- Now passing the second tag directly!
      Brainstorm.config.ar_filters.soul_skip,
      Brainstorm.config.ar_filters.inst_observatory,
      Brainstorm.config.ar_filters.inst_perkeo
    )
  end)
  
  if not call_success then
    print("[Brainstorm] DLL call failed: " .. tostring(call_result))
    return nil
  end
  
  if call_result then
    local result_str = ffi.string(call_result)
    
    -- Free the result memory
    if immolate.free_result then
      immolate.free_result(call_result)
    end
    
    -- With the new DLL, dual tags are already properly validated
    -- No need for additional checks after restart!
    return result_str
  end
  
  return nil
end

-- Optional: Function to test the new get_tags functionality
function Brainstorm.test_seed_tags(seed)
  if not init_ffi() then
    print("[Brainstorm] FFI not initialized")
    return nil
  end
  
  local success, immolate = pcall(ffi.load, Brainstorm.PATH .. "/Immolate.dll")
  if not success then
    print("[Brainstorm] Failed to load Immolate.dll")
    return nil
  end
  
  local call_success, call_result = pcall(function()
    return immolate.get_tags(seed)
  end)
  
  if call_success and call_result then
    local result_str = ffi.string(call_result)
    
    -- Free the result memory
    if immolate.free_result then
      immolate.free_result(call_result)
    end
    
    -- Parse the result (format: "small_tag|big_tag")
    local small_tag, big_tag = result_str:match("([^|]+)|([^|]+)")
    print(string.format("[Brainstorm] Seed %s has tags: Small=%s, Big=%s", 
      seed, small_tag, big_tag))
    
    return {small = small_tag, big = big_tag}
  end
  
  return nil
end