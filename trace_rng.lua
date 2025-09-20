-- Simple RNG trace activation script
-- Just require this file to activate tracing

print("[RNG TRACE] Starting RNG trace activation...")

-- Check if Brainstorm is loaded
if not Brainstorm then
    print("[RNG TRACE] ERROR: Brainstorm not loaded!")
    return false
end

-- Check if RNGTrace module is available
if not Brainstorm.RNGTrace then
    print("[RNG TRACE] ERROR: RNGTrace module not found!")
    return false
end

-- Activate the patches
print("[RNG TRACE] Installing patches...")
Brainstorm.RNGTrace.install_all_patches()

-- Generate test traces for known seeds
print("[RNG TRACE] Generating test traces for seeds: AAAAAAAA, 00000000, 7NTPKW6P, ZZZZZZZZ")
Brainstorm.RNGTrace.generate_test_traces()

print("[RNG TRACE] ✓ Complete! Check trace file at:")
print("  C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm\\rng_trace.jsonl")

return true