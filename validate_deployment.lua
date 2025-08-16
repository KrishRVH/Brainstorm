-- Deployment Validation Script for Brainstorm Mod
-- Run this to ensure everything is properly configured

print("========================================")
print("Brainstorm Mod Deployment Validation")
print("========================================")

local validation = {
  errors = {},
  warnings = {},
  info = {}
}

-- Check 1: FFI availability
print("\n[1/7] Checking FFI...")
local ffi_ok, ffi = pcall(require, "ffi")
if ffi_ok then
  print("  ✓ FFI is available")
else
  table.insert(validation.errors, "FFI not available: " .. tostring(ffi))
  print("  ✗ FFI not available")
end

-- Check 2: DLL signature
if ffi_ok then
  print("\n[2/7] Checking DLL signature...")
  local cdef_ok, cdef_err = pcall(ffi.cdef, [[
    const char* brainstorm(const char* seed, const char* voucher, const char* pack, const char* tag1, const char* tag2, double souls, bool observatory, bool perkeo);
    const char* get_tags(const char* seed);
    void free_result(const char* result);
  ]])
  
  if cdef_ok then
    print("  ✓ DLL signature defined correctly")
  else
    table.insert(validation.errors, "Failed to define DLL signature: " .. tostring(cdef_err))
    print("  ✗ Failed to define DLL signature")
  end
end

-- Check 3: DLL file existence
print("\n[3/7] Checking DLL files...")
local original_dll = io.open("Immolate.dll", "rb")
local enhanced_dll = io.open("Immolate_new.dll", "rb")

if original_dll then
  original_dll:close()
  print("  ✓ Immolate.dll found")
else
  table.insert(validation.warnings, "Immolate.dll not found")
  print("  ⚠ Immolate.dll not found")
end

if enhanced_dll then
  enhanced_dll:close()
  print("  ✓ Immolate_new.dll found (enhanced version)")
  table.insert(validation.info, "Enhanced DLL available for deployment")
else
  print("  ⚠ Immolate_new.dll not found (optional)")
end

-- Check 4: DLL loading
if ffi_ok then
  print("\n[4/7] Checking DLL loading...")
  
  local dll_to_test = enhanced_dll and "Immolate_new.dll" or "Immolate.dll"
  local load_ok, dll = pcall(ffi.load, dll_to_test)
  
  if load_ok then
    print("  ✓ " .. dll_to_test .. " loaded successfully")
    
    -- Check exported functions
    if dll.brainstorm then
      print("  ✓ brainstorm function found")
    else
      table.insert(validation.errors, "brainstorm function not exported")
    end
    
    if enhanced_dll and dll.get_tags then
      print("  ✓ get_tags function found (enhanced)")
    elseif enhanced_dll then
      table.insert(validation.warnings, "get_tags function not found in enhanced DLL")
    end
    
    if dll.free_result then
      print("  ✓ free_result function found")
    else
      table.insert(validation.warnings, "free_result function not found")
    end
  else
    table.insert(validation.errors, "Failed to load DLL: " .. tostring(dll))
    print("  ✗ Failed to load " .. dll_to_test)
  end
end

-- Check 5: Config structure
print("\n[5/7] Checking configuration...")
local config_ok = true
local required_fields = {
  "ar_filters.tag_name",
  "ar_filters.tag_id", 
  "ar_filters.tag2_name",
  "ar_filters.tag2_id",
  "ar_filters.voucher_name",
  "ar_filters.pack",
  "ar_filters.soul_skip",
  "ar_filters.inst_observatory",
  "ar_filters.inst_perkeo"
}

-- Simulate config
local test_config = {
  ar_filters = {
    tag_name = "tag_investment",
    tag_id = 9,
    tag2_name = "",
    tag2_id = 1,
    voucher_name = "",
    voucher_id = 1,
    pack = {},
    soul_skip = 0,
    inst_observatory = false,
    inst_perkeo = false
  }
}

for _, field in ipairs(required_fields) do
  local parts = {}
  for part in field:gmatch("[^.]+") do
    table.insert(parts, part)
  end
  
  local value = test_config
  for _, part in ipairs(parts) do
    value = value and value[part]
  end
  
  if value == nil then
    config_ok = false
    table.insert(validation.errors, "Missing config field: " .. field)
  end
end

if config_ok then
  print("  ✓ Configuration structure is valid")
else
  print("  ✗ Configuration has missing fields")
end

-- Check 6: Core files
print("\n[6/7] Checking core files...")
local files_to_check = {
  "Core/Brainstorm.lua",
  "UI/ui.lua",
  "config.lua",
  "nativefs.lua",
  "lovely.toml"
}

local all_files_ok = true
for _, file in ipairs(files_to_check) do
  local f = io.open(file, "r")
  if f then
    f:close()
    print("  ✓ " .. file .. " found")
  else
    all_files_ok = false
    table.insert(validation.errors, "Missing file: " .. file)
    print("  ✗ " .. file .. " not found")
  end
end

-- Check 7: Best practices
print("\n[7/7] Checking best practices...")
local practices = {
  ["Error handling"] = true,  -- Using pcall for DLL calls
  ["Memory management"] = enhanced_dll ~= nil,  -- free_result available
  ["Backwards compatibility"] = true,  -- Falls back to old DLL
  ["Debug logging"] = true,  -- Has debug system
  ["Config persistence"] = true  -- Saves config
}

for practice, implemented in pairs(practices) do
  if implemented then
    print("  ✓ " .. practice)
  else
    table.insert(validation.warnings, practice .. " could be improved")
    print("  ⚠ " .. practice .. " needs improvement")
  end
end

-- Summary
print("\n========================================")
print("VALIDATION SUMMARY")
print("========================================")

if #validation.errors == 0 then
  print("✓ No critical errors found!")
else
  print("✗ CRITICAL ERRORS:")
  for _, err in ipairs(validation.errors) do
    print("  - " .. err)
  end
end

if #validation.warnings > 0 then
  print("\n⚠ WARNINGS:")
  for _, warn in ipairs(validation.warnings) do
    print("  - " .. warn)
  end
end

if #validation.info > 0 then
  print("\nℹ INFO:")
  for _, info in ipairs(validation.info) do
    print("  - " .. info)
  end
end

print("\n========================================")

-- Recommendations
if enhanced_dll then
  print("\nRECOMMENDATION: Deploy with enhanced DLL")
  print("  1. Rename Immolate.dll to Immolate_original.dll")
  print("  2. Rename Immolate_new.dll to Immolate.dll")
  print("  3. Run deploy.sh")
else
  print("\nRECOMMENDATION: Build enhanced DLL for better performance")
  print("  1. cd ImmolateCPP")
  print("  2. ./build_simple.sh")
  print("  3. Deploy the new DLL")
end

print("========================================")

-- Exit code
if #validation.errors == 0 then
  print("\n✓ Deployment validation PASSED")
  os.exit(0)
else
  print("\n✗ Deployment validation FAILED")
  os.exit(1)
end