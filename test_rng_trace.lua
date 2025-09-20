-- Test script for RNG tracing
-- Run this in Balatro's console to generate test traces

-- Activate RNG tracing
if Brainstorm and Brainstorm.activate_rng_trace then
    print("Activating RNG trace...")
    if Brainstorm.activate_rng_trace() then
        print("RNG trace activated successfully")
        
        -- Generate test traces for known seeds
        print("Generating test traces...")
        if Brainstorm.generate_test_traces() then
            print("Test traces generated successfully")
            print("Check: C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm\\rng_trace.jsonl")
        else
            print("Failed to generate test traces")
        end
    else
        print("Failed to activate RNG trace")
    end
else
    print("Brainstorm or RNG trace not available")
end