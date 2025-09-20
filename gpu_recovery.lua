-- Enhanced GPU Recovery Script (per Distinguished Engineer's plan)
-- Tries soft reset first, then hard reset, with smoke test validation

local ffi = require("ffi")

-- Declare all recovery functions
ffi.cdef[[
    bool brainstorm_gpu_reset();        // Soft reset
    bool brainstorm_gpu_hard_reset();   // Hard reset with driver unload
    bool brainstorm_is_driver_ready();
    int  brainstorm_get_last_cuda_error();
    bool brainstorm_run_smoke();        // Smoke test kernel
    void brainstorm_gpu_disable_for_session();
]]

-- Load the DLL
local immolate = ffi.load("Immolate")

-- CUDA error code names for common errors
local cuda_errors = {
    [0] = "SUCCESS",
    [201] = "DEINITIALIZED",
    [700] = "ILLEGAL_ADDRESS",
    [702] = "LAUNCH_TIMEOUT",
    [709] = "CONTEXT_IS_DESTROYED",
    [716] = "INVALID_CONTEXT",
    [719] = "NOT_PERMITTED"
}

local function get_error_name(code)
    return cuda_errors[code] or string.format("ERROR_%d", code)
end

local function try_reset(kind)
    if kind == "soft" then
        print("Attempting SOFT reset...")
        if immolate.brainstorm_gpu_reset() then
            print("  Soft reset completed, running smoke test...")
            if immolate.brainstorm_run_smoke() then
                print("✓ Soft reset SUCCEEDED - GPU is functional")
                return true
            else
                print("  Smoke test failed after soft reset")
            end
        else
            print("  Soft reset failed")
        end
    else
        print("Attempting HARD reset (driver unload/reload)...")
        if immolate.brainstorm_gpu_hard_reset() then
            print("  Hard reset completed, running smoke test...")
            if immolate.brainstorm_run_smoke() then
                print("✓ Hard reset SUCCEEDED - GPU is functional")
                return true
            else
                print("  Smoke test failed after hard reset")
            end
        else
            print("  Hard reset failed")
        end
    end
    return false
end

print("========================================")
print("  GPU RECOVERY PROCEDURE")
print("========================================")
print("")

-- Get current status
local last_error = immolate.brainstorm_get_last_cuda_error()
local is_ready = immolate.brainstorm_is_driver_ready()

print("Initial status:")
print(string.format("  Driver ready: %s", tostring(is_ready)))
print(string.format("  Last CUDA error: %s", get_error_name(last_error)))
print("")

-- Try recovery ladder: soft → hard → disable
if try_reset("soft") then
    print("")
    print("========================================")
    print("  RECOVERY SUCCESSFUL!")
    print("========================================")
    print("GPU acceleration is now available.")
    return true
elseif try_reset("hard") then
    print("")
    print("========================================")
    print("  RECOVERY SUCCESSFUL!")
    print("========================================")
    print("GPU acceleration is now available.")
    return true
else
    print("")
    print("========================================")
    print("  RECOVERY FAILED - REBOOT REQUIRED")
    print("========================================")
    print("")
    print("The GPU driver is in a permanently corrupted state.")
    print("This typically happens after:")
    print("  - Illegal memory access in previous kernel")
    print("  - TDR (timeout) during long operations")
    print("  - Driver bug or hardware issue")
    print("")
    print("REQUIRED ACTION: Restart Windows")
    print("")
    print("GPU has been disabled for this session to prevent crashes.")
    
    -- Disable GPU for safety
    immolate.brainstorm_gpu_disable_for_session()
    
    return false
end