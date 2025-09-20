-- Emergency GPU Hard Reset Script
-- Run this from Balatro's Lua console or add to Brainstorm.lua

local ffi = require("ffi")

-- Declare the hard reset function
ffi.cdef[[
    bool brainstorm_gpu_hard_reset();
    int brainstorm_get_last_cuda_error();
    bool brainstorm_is_driver_ready();
    void brainstorm_gpu_disable_for_session();
]]

-- Load the DLL
local immolate = ffi.load("Immolate")

print("========================================")
print("  GPU HARD RESET ATTEMPT")
print("========================================")

-- Get current status
local last_error = immolate.brainstorm_get_last_cuda_error()
local is_ready = immolate.brainstorm_is_driver_ready()

print("Current status:")
print("  Last CUDA error: " .. tostring(last_error))
print("  Driver ready: " .. tostring(is_ready))
print("")
print("Attempting hard reset...")
print("This will unload and reload the CUDA driver...")

-- Attempt hard reset
local success = immolate.brainstorm_gpu_hard_reset()

if success then
    print("✓ HARD RESET SUCCESSFUL!")
    print("GPU should now be functional.")
else
    print("✗ HARD RESET FAILED!")
    print("SYSTEM REBOOT REQUIRED!")
    print("")
    print("The GPU driver is in a permanently corrupted state.")
    print("Please restart Windows to clear the GPU state.")
    
    -- Disable GPU for this session
    immolate.brainstorm_gpu_disable_for_session()
    print("GPU has been disabled for this session.")
end

print("========================================")

return success