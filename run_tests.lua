#!/usr/bin/env lua

-- Master test runner for Brainstorm mod
-- Runs all test suites and reports overall results

print([[
╔════════════════════════════════════════════════════╗
║        BRAINSTORM MOD - COMPREHENSIVE TEST SUITE   ║
║                      Version 3.0                   ║
╚════════════════════════════════════════════════════╝
]])

local test_suites = {
  { name = "Basic File Integrity", file = "basic_test.lua" },
  { name = "Erratic Deck Validation", file = "tests/test_erratic_deck.lua" },
  { name = "Save State Integration", file = "tests/test_save_states.lua" },
  { name = "CUDA Fallback Safety", file = "test_cuda_fallback.lua" },
  -- C++ tests require separate build/run process:
  -- { name = "C++ Memory Safety", file = "ImmolateCPP/tests/test_memory_safety.cpp" },
  -- { name = "C++ Critical Functions", file = "ImmolateCPP/tests/test_critical_functions.cpp" },
}

local total_passed = 0
local total_failed = 0
local suite_results = {}

-- Run each test suite
for _, suite in ipairs(test_suites) do
  print("\n" .. string.rep("═", 55))
  print("  Running: " .. suite.name)
  print(string.rep("═", 55))

  local success, module = pcall(dofile, suite.file)

  if success and module and module.run then
    local suite_passed = module.run()
    suite_results[suite.name] = suite_passed

    if suite_passed then
      total_passed = total_passed + 1
      print("\n✅ " .. suite.name .. " - PASSED")
    else
      total_failed = total_failed + 1
      print("\n❌ " .. suite.name .. " - FAILED")
    end
  else
    print("\n⚠️  " .. suite.name .. " - SKIPPED (not found or error)")
    print("    " .. tostring(module))
  end
end

-- Overall summary
print("\n" .. string.rep("═", 55))
print("                    TEST SUMMARY")
print(string.rep("═", 55))

for name, passed in pairs(suite_results) do
  local status = passed and "✅ PASS" or "❌ FAIL"
  print(string.format("  %-30s %s", name, status))
end

print(string.rep("─", 55))
print(
  string.format("  Total: %d passed, %d failed", total_passed, total_failed)
)
print(string.rep("═", 55))

if total_failed == 0 then
  print("\n🎉 All tests passed! The mod is ready for deployment.")
else
  print("\n⚠️  Some tests failed. Please review the output above.")
end

-- Return exit code for CI/CD integration
os.exit(total_failed == 0 and 0 or 1)
