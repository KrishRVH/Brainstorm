-- Logging integration helper
-- This file shows how logging should be integrated throughout the codebase

local examples = {
  -- Before: Simple print
  before_1 = [[print("[Brainstorm] ERROR: Could not find directory")]],
  after_1 = [[log:error("Could not find directory")]],

  -- Before: Print with concatenation
  before_2 = [[print("[Brainstorm] Found seed: " .. seed)]],
  after_2 = [[log:info("Found seed", {seed = seed})]],

  -- Before: Debug print with complex formatting
  before_3 = [[
    if Brainstorm.debug.enabled then
      print(string.format("[Brainstorm] Speed: %d seeds/sec", speed))
    end
  ]],
  after_3 = [[log:debug("Performance update", {seeds_per_sec = speed})]],

  -- Before: Conditional debug logging
  before_4 = [[
    if Brainstorm.debug.enabled and count % 100 == 0 then
      print("[Brainstorm] Checked " .. count .. " seeds")
    end
  ]],
  after_4 = [[
    log:log_throttled(
      logger.LEVELS.DEBUG,
      "seed_check",
      "Seed check progress",
      {count = count},
      5 -- Log every 5 seconds max
    )
  ]],

  -- Before: Error with details
  before_5 = [[
    print("[Brainstorm] Failed to load DLL: " .. tostring(err))
  ]],
  after_5 = [[
    log:error("Failed to load DLL", {
      path = dll_path,
      error = tostring(err),
      cwd = love.filesystem.getWorkingDirectory()
    })
  ]],

  -- Before: Performance timing
  before_6 = [[
    local start = os.clock()
    -- ... do work ...
    local elapsed = os.clock() - start
    print("Operation took " .. elapsed .. " seconds")
  ]],
  after_6 = [[
    log:start_timer("operation")
    -- ... do work ...
    log:end_timer("operation", "Operation completed")
  ]],

  -- Before: Success messages
  before_7 = [[
    print("[Brainstorm] SUCCESS! Found matching seed")
  ]],
  after_7 = [[
    log:info("Found matching seed", {
      seed = current_seed,
      attempts = attempts,
      duration = elapsed,
      filters = {
        tags = {tag1, tag2},
        voucher = voucher_name,
        pack = pack_name
      }
    })
  ]],
}

-- Logging best practices for Brainstorm

local best_practices = {
  -- 1. Use structured data instead of string concatenation
  good = [[log:info("Seed found", {seed = seed, attempts = 42})]],
  bad = [[log:info("Seed found: " .. seed .. " after " .. attempts)]],

  -- 2. Use appropriate log levels
  levels = {
    TRACE = "Very detailed debugging info (function entry/exit)",
    DEBUG = "Detailed debugging info (variable values, state changes)",
    INFO = "Important events (seed found, save completed)",
    WARN = "Potential issues (fallback used, slow performance)",
    ERROR = "Errors that don't stop execution (DLL load failed)",
    FATAL = "Errors that stop the mod (critical failure)",
  },

  -- 3. Use throttling for high-frequency events
  high_freq = [[
    -- Don't log every seed check
    log:log_throttled(
      logger.LEVELS.DEBUG,
      "seed_progress",
      "Seed check progress",
      {checked = count, rate = seeds_per_sec},
      1 -- Max once per second
    )
  ]],

  -- 4. Include context in errors
  error_context = [[
    log:error("Failed to validate deck", {
      face_cards = face_count,
      required = min_face_cards,
      suit_ratio = ratio,
      seed = current_seed,
      deck_type = "Erratic"
    })
  ]],

  -- 5. Use timers for performance monitoring
  performance = [[
    log:start_timer("dll_search")
    local result = immolate.brainstorm(...)
    local elapsed = log:end_timer("dll_search", "DLL search completed")
    
    if elapsed > 1.0 then
      log:warn("Slow DLL search", {duration = elapsed, seed = seed})
    end
  ]],

  -- 6. Log state transitions
  state_change = [[
    log:debug("Auto-reroll state change", {
      from = old_state,
      to = new_state,
      trigger = trigger_event
    })
  ]],

  -- 7. Use module-specific loggers
  modules = [[
    -- In UI module
    local log = logger.for_module("Brainstorm:UI")
    
    -- In DLL interface
    local log = logger.for_module("Brainstorm:DLL")
    
    -- In save system
    local log = logger.for_module("Brainstorm:SaveState")
  ]],

  -- 8. Log configuration changes
  config = [[
    log:info("Configuration updated", {
      setting = "use_cuda",
      old_value = old_val,
      new_value = new_val,
      source = "UI"
    })
  ]],

  -- 9. Log performance statistics periodically
  stats = [[
    if os.clock() - last_stats_time > 30 then
      local stats = {
        seeds_tested = total_seeds,
        seeds_per_sec = current_rate,
        matches_found = matches,
        success_rate = matches / total_seeds,
        gpu_enabled = Brainstorm.debug.gpu_enabled,
        memory_used = collectgarbage("count")
      }
      log:info("Performance statistics", stats)
      last_stats_time = os.clock()
    end
  ]],

  -- 10. Log critical paths for debugging
  critical = [[
    log:trace("Entering dual tag validation", {
      tag1 = tag1,
      tag2 = tag2,
      small_blind = small_blind_tag,
      big_blind = big_blind_tag
    })
    
    -- ... validation logic ...
    
    log:trace("Dual tag validation result", {
      matched = result,
      reason = rejection_reason
    })
  ]],
}

-- Function to update all logging in a file
local function update_logging_in_file(file_path)
  -- This would be run to update all print statements to use the logger
  local replacements = {
    -- Error messages
    {
      pattern = 'print%("?%[Brainstorm%] ERROR: ([^"]+)"?%)',
      replacement = 'log:error("%1")',
    },
    -- Info messages
    {
      pattern = 'print%("?%[Brainstorm%] ([^"]+)"?%)',
      replacement = 'log:info("%1")',
    },
    -- Debug messages
    {
      pattern = "if%s+Brainstorm%.debug%.enabled%s+then%s+print%((.-)%)",
      replacement = "log:debug(%1)",
    },
  }

  -- Read file
  local file = io.open(file_path, "r")
  if not file then
    return
  end
  local content = file:read("*all")
  file:close()

  -- Apply replacements
  for _, rule in ipairs(replacements) do
    content = content:gsub(rule.pattern, rule.replacement)
  end

  -- Write updated file
  file = io.open(file_path, "w")
  file:write(content)
  file:close()
end

return {
  examples = examples,
  best_practices = best_practices,
  update_logging_in_file = update_logging_in_file,
}
