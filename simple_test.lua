#!/usr/bin/env lua

-- Simple test runner that actually works
print("Running Brainstorm Tests...")

-- Test 1: Check if files exist
local files_to_check = {
  "Core/Brainstorm.lua",
  "Core/logger.lua",
  "UI/ui.lua",
  "config.lua",
  "Immolate.dll",
}

print("\n=== File Existence Tests ===")
local nfs = require("nativefs")
for _, file in ipairs(files_to_check) do
  local exists = nfs.getInfo(file) ~= nil
  if exists then
    print("✓ " .. file .. " exists")
  else
    print("✗ " .. file .. " missing!")
  end
end

-- Test 2: Logger functionality
print("\n=== Logger Tests ===")
local logger_ok, logger = pcall(require, "Core.logger")
if logger_ok then
  print("✓ Logger module loads")
  if logger.for_module then
    print("✓ for_module function exists")
  else
    print("✗ for_module function missing")
  end
else
  print("✗ Logger module failed to load: " .. tostring(logger))
end

-- Test 3: Config structure
print("\n=== Config Tests ===")
local config_ok, config_content = pcall(nfs.read, "config.lua")
if config_ok and config_content then
  print("✓ Config file readable")
  -- Check for key settings
  if config_content:find("use_cuda") then
    print("✓ use_cuda setting present")
  end
  if config_content:find("debug_enabled") then
    print("✓ debug_enabled setting present")
  end
else
  print("✗ Config file not readable")
end

print("\n=== All Tests Complete ===")
