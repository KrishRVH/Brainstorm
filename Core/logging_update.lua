-- Script to update all logging in Brainstorm to use the new logger
-- This shows the comprehensive logging strategy

local updates = {}

-- 1. Configuration logging
updates.config = {
  load = function()
    log:info(
      "Loading configuration",
      { path = Brainstorm.PATH .. "/config.lua" }
    )
    -- After loading
    log:debug("Configuration loaded", {
      debug_enabled = Brainstorm.config.debug_enabled,
      use_cuda = Brainstorm.config.use_cuda,
      filters = {
        face_cards = Brainstorm.config.ar_prefs.face_count,
        suit_ratio = Brainstorm.config.ar_prefs.suit_ratio_percent,
        tag1 = Brainstorm.config.ar_filters.tag_name,
        tag2 = Brainstorm.config.ar_filters.tag2_name,
      },
    })
  end,

  save = function()
    log:debug(
      "Saving configuration",
      { path = Brainstorm.PATH .. "/config.lua" }
    )
    -- After saving
    log:trace("Configuration saved successfully")
  end,
}

-- 2. DLL interface logging
updates.dll = {
  load = function()
    log:info("Loading DLL", { path = dll_path })
    log:start_timer("dll_load")
    -- ... load DLL ...
    log:end_timer("dll_load", "DLL loaded")
  end,

  search = function()
    log:trace("Starting DLL search", {
      seed = seed,
      voucher = voucher_name,
      pack = pack_name,
      tags = { tag1_name, tag2_name },
    })

    log:start_timer("dll_search")
    -- ... perform search ...
    local elapsed = log:end_timer("dll_search")

    if elapsed > 0.5 then
      log:warn("Slow DLL search", {
        duration = elapsed,
        seed = seed,
        reason = "Consider reducing filter complexity",
      })
    end
  end,

  compatibility = function()
    log:debug("DLL compatibility check", {
      enhanced_available = enhanced_success,
      fallback_to_original = not enhanced_success,
      parameters = enhanced_success and 8 or 7,
    })
  end,
}

-- 3. Auto-reroll state machine logging
updates.auto_reroll = {
  state_change = function(old_state, new_state, trigger)
    log:debug("Auto-reroll state transition", {
      from = old_state,
      to = new_state,
      trigger = trigger,
      frame = Brainstorm.ar_frames,
    })
  end,

  progress = function()
    -- Use throttled logging for high-frequency updates
    log:log_throttled(
      logger.LEVELS.DEBUG,
      "ar_progress",
      "Auto-reroll progress",
      {
        seeds_tested = Brainstorm.debug.seeds_tested,
        seeds_per_sec = seeds_per_sec,
        time_elapsed = elapsed,
        matches = Brainstorm.debug.seeds_found,
      },
      5 -- Max once per 5 seconds
    )
  end,

  found = function(seed, attempts, elapsed)
    log:info("SEED FOUND!", {
      seed = seed,
      attempts = attempts,
      duration = string.format("%.2fs", elapsed),
      rate = string.format("%.1f seeds/sec", attempts / elapsed),
      filters = {
        tags = {
          Brainstorm.config.ar_filters.tag_name,
          Brainstorm.config.ar_filters.tag2_name,
        },
        voucher = Brainstorm.config.ar_filters.voucher_name,
        pack = Brainstorm.config.ar_filters.pack,
        face_cards = Brainstorm.config.ar_prefs.face_count,
        suit_ratio = Brainstorm.config.ar_prefs.suit_ratio_percent,
      },
    })
  end,
}

-- 4. Deck validation logging
updates.deck_validation = {
  start = function(deck_data)
    log:trace("Starting deck validation", {
      min_face_cards = min_face_cards,
      suit_ratio = dominant_suit_ratio,
      deck_size = #deck_data,
    })
  end,

  rejection = function(reason, details)
    log:debug("Deck rejected", {
      reason = reason,
      details = details,
      seed = G.GAME.pseudorandom.seed,
    })
  end,

  success = function(stats)
    log:debug("Deck validated", {
      face_cards = stats.face_count,
      suit_ratio = stats.suit_ratio,
      aces = stats.ace_count,
      seed = G.GAME.pseudorandom.seed,
    })
  end,
}

-- 5. Save state logging
updates.save_state = {
  save = function(slot)
    log:info("Saving game state", { slot = slot })
    log:start_timer("save_state")
    -- ... perform save ...
    log:end_timer("save_state", "State saved")
  end,

  load = function(slot)
    log:info("Loading game state", { slot = slot })
    log:start_timer("load_state")
    -- ... perform load ...
    log:end_timer("load_state", "State loaded")
  end,

  error = function(operation, slot, error)
    log:error("Save state operation failed", {
      operation = operation,
      slot = slot,
      error = tostring(error),
      file = "save_state_" .. slot .. ".jkr",
    })
  end,
}

-- 6. Performance monitoring
updates.performance = {
  periodic_stats = function()
    local stats = {
      -- Memory
      memory_mb = collectgarbage("count") / 1024,

      -- Performance
      seeds_tested = Brainstorm.debug.seeds_tested,
      seeds_per_sec = current_rate,
      time_elapsed = elapsed,

      -- Success metrics
      matches_found = Brainstorm.debug.seeds_found,
      success_rate = Brainstorm.debug.seeds_found
        / Brainstorm.debug.seeds_tested,

      -- Rejection analysis
      rejections = {
        face_cards = Brainstorm.debug.rejection_reasons.face_cards,
        suit_ratio = Brainstorm.debug.rejection_reasons.suit_ratio,
        dll_filter = Brainstorm.debug.rejection_reasons.dll_filter,
      },

      -- System
      gpu_enabled = Brainstorm.debug.gpu_enabled,
      dll_version = immolate_dll and "enhanced" or "original",
    }

    log:info("Performance statistics", stats)

    -- Log hot paths if in debug mode
    if Brainstorm.debug.enabled then
      local hot_paths = logger.global:get_stats().hot_paths
      if #hot_paths > 0 then
        log:debug("Hot code paths", { paths = hot_paths })
      end
    end
  end,

  bottleneck_detection = function()
    if
      seeds_per_sec < 100 and not Brainstorm.config.ar_prefs.erratic_required
    then
      log:warn("Performance bottleneck detected", {
        seeds_per_sec = seeds_per_sec,
        possible_causes = {
          "Complex filter combination",
          "DLL not loaded",
          "Debug logging overhead",
        },
      })
    end
  end,
}

-- 7. Debug report with structured logging
updates.debug_report = function(success)
  local elapsed = os_clock() - Brainstorm.debug.start_time
  local seeds_per_sec = Brainstorm.debug.seeds_tested / elapsed

  -- Main report
  local report = {
    result = success and "SUCCESS" or "STOPPED",
    duration = string.format("%.2fs", elapsed),
    seeds_tested = Brainstorm.debug.seeds_tested,
    seeds_per_sec = string.format("%.1f", seeds_per_sec),
    matches_found = Brainstorm.debug.seeds_found,
  }

  -- Rejection analysis
  local total_rejections = 0
  for reason, count in pairs(Brainstorm.debug.rejection_reasons) do
    total_rejections = total_rejections + count
  end

  if total_rejections > 0 then
    report.rejections = {}
    for reason, count in pairs(Brainstorm.debug.rejection_reasons) do
      report.rejections[reason] = {
        count = count,
        percentage = string.format("%.1f%%", count / total_rejections * 100),
      }
    end
  end

  -- Distribution analysis
  report.distributions = {
    face_cards = {},
    suit_ratios = {},
  }

  for bucket, count in pairs(Brainstorm.debug.distributions.face_cards) do
    if count > 0 then
      report.distributions.face_cards[bucket] = {
        count = count,
        percentage = string.format(
          "%.1f%%",
          count / Brainstorm.debug.seeds_tested * 100
        ),
      }
    end
  end

  -- Recommendations
  if not success and elapsed > 30 then
    report.recommendations = {}

    if
      Brainstorm.debug.highest_face_count
      < Brainstorm.config.ar_prefs.face_count
    then
      table.insert(
        report.recommendations,
        string.format(
          "Lower face card requirement (highest found: %d)",
          Brainstorm.debug.highest_face_count
        )
      )
    end

    if
      Brainstorm.debug.highest_suit_ratio
      < Brainstorm.config.ar_prefs.suit_ratio_decimal
    then
      table.insert(
        report.recommendations,
        string.format(
          "Lower suit ratio requirement (highest found: %.1f%%)",
          Brainstorm.debug.highest_suit_ratio * 100
        )
      )
    end
  end

  -- Log the complete report
  log:info("=== Debug Report ===", report)

  -- Also write to file if in debug mode
  if Brainstorm.debug.enabled and logger.global.file_path then
    local report_file = Brainstorm.PATH
      .. "/debug_report_"
      .. os.date("%Y%m%d_%H%M%S")
      .. ".json"
    local file = io.open(report_file, "w")
    if file then
      -- Simple JSON serialization
      file:write(json_encode(report))
      file:close()
      log:debug("Debug report saved", { file = report_file })
    end
  end
end

-- 8. Error handling with context
updates.error_handling = {
  wrap_critical = function(operation, func)
    log:trace("Starting critical operation", { operation = operation })

    local success, result = pcall(func)

    if success then
      log:trace("Critical operation completed", { operation = operation })
      return result
    else
      log:error("Critical operation failed", {
        operation = operation,
        error = tostring(result),
        stack = debug.traceback(),
      })

      -- Log state for debugging
      log:debug("State at error", {
        ar_active = Brainstorm.ar_active,
        ar_frames = Brainstorm.ar_frames,
        seeds_tested = Brainstorm.debug.seeds_tested,
        config = Brainstorm.config,
      })

      return nil
    end
  end,
}

return updates
