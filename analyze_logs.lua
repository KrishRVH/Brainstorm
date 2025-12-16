#!/usr/bin/env lua

-- Log analysis tool for Brainstorm
-- Analyzes log files to identify patterns, errors, and performance issues

local function read_log_file(path)
  local file = io.open(path, "r")
  if not file then
    print("Error: Could not open log file: " .. path)
    return nil
  end

  local lines = {}
  for line in file:lines() do
    table.insert(lines, line)
  end
  file:close()

  return lines
end

local function parse_log_line(line)
  -- Parse format: [timestamp] [level] [module] [caller] message {data}
  local pattern =
    "%[([^%]]+)%]%s*%[(%w+)%]%s*%[([^%]]+)%]%s*%[?([^%]]*)%]?%s*(.+)"
  local timestamp, level, module, caller, message = line:match(pattern)

  if not timestamp then
    -- Try simpler format without caller
    pattern = "%[([^%]]+)%]%s*%[(%w+)%]%s*%[([^%]]+)%]%s*(.+)"
    timestamp, level, module, message = line:match(pattern)
    caller = ""
  end

  if not timestamp then
    return nil
  end

  -- Extract structured data if present
  local data = {}
  local data_str = message:match("{(.+)}$")
  if data_str then
    message = message:gsub("%s*{.+}$", "")
    -- Simple parsing of key=value pairs
    for key, value in data_str:gmatch("(%w+)=([^,]+)") do
      data[key] = value:gsub("^%s+", ""):gsub("%s+$", "")
    end
  end

  return {
    timestamp = timestamp,
    level = level,
    module = module,
    caller = caller,
    message = message,
    data = data,
  }
end

local function analyze_logs(log_path)
  local lines = read_log_file(log_path)
  if not lines then
    return
  end

  print(string.rep("=", 70))
  print("  BRAINSTORM LOG ANALYSIS")
  print(string.rep("=", 70))
  print("Log file: " .. log_path)
  print("Total lines: " .. #lines)
  print()

  local stats = {
    by_level = {},
    by_module = {},
    errors = {},
    warnings = {},
    performance = {},
    seeds_found = 0,
    total_seeds_tested = 0,
    hot_paths = {},
    time_range = { first = nil, last = nil },
  }

  -- Parse all lines
  local entries = {}
  for _, line in ipairs(lines) do
    local entry = parse_log_line(line)
    if entry then
      table.insert(entries, entry)

      -- Update stats
      stats.by_level[entry.level] = (stats.by_level[entry.level] or 0) + 1
      stats.by_module[entry.module] = (stats.by_module[entry.module] or 0) + 1

      -- Track time range
      if not stats.time_range.first then
        stats.time_range.first = entry.timestamp
      end
      stats.time_range.last = entry.timestamp

      -- Collect errors and warnings
      if entry.level == "ERROR" then
        table.insert(stats.errors, entry)
      elseif entry.level == "WARN" then
        table.insert(stats.warnings, entry)
      end

      -- Track performance metrics
      if entry.message:match("Performance statistics") then
        table.insert(stats.performance, entry)
        if entry.data.seeds_tested then
          local tested = tonumber(entry.data.seeds_tested)
          if tested and tested > stats.total_seeds_tested then
            stats.total_seeds_tested = tested
          end
        end
      end

      -- Count seeds found
      if entry.message:match("SEED FOUND") then
        stats.seeds_found = stats.seeds_found + 1
      end

      -- Track hot paths
      if entry.caller and entry.caller ~= "" then
        stats.hot_paths[entry.caller] = (stats.hot_paths[entry.caller] or 0) + 1
      end
    end
  end

  -- Display analysis
  print("TIME RANGE")
  print(string.rep("-", 70))
  print("First entry: " .. (stats.time_range.first or "N/A"))
  print("Last entry:  " .. (stats.time_range.last or "N/A"))
  print()

  print("LOG LEVELS")
  print(string.rep("-", 70))
  local level_order = { "TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL" }
  for _, level in ipairs(level_order) do
    local count = stats.by_level[level] or 0
    if count > 0 then
      local bar = string.rep("█", math.min(50, math.floor(count / 10)))
      print(string.format("%-6s: %6d  %s", level, count, bar))
    end
  end
  print()

  print("MODULES")
  print(string.rep("-", 70))
  local modules = {}
  for module, count in pairs(stats.by_module) do
    table.insert(modules, { module = module, count = count })
  end
  table.sort(modules, function(a, b)
    return a.count > b.count
  end)

  for i = 1, math.min(10, #modules) do
    local m = modules[i]
    print(string.format("%-20s: %6d", m.module, m.count))
  end
  print()

  if #stats.errors > 0 then
    print("ERRORS (" .. #stats.errors .. " found)")
    print(string.rep("-", 70))
    for i = 1, math.min(5, #stats.errors) do
      local err = stats.errors[i]
      print(string.format("[%s] %s", err.timestamp, err.message))
      if err.data.error then
        print("  Error: " .. err.data.error)
      end
    end
    print()
  end

  if #stats.warnings > 0 then
    print("WARNINGS (" .. #stats.warnings .. " found)")
    print(string.rep("-", 70))
    for i = 1, math.min(5, #stats.warnings) do
      local warn = stats.warnings[i]
      print(string.format("[%s] %s", warn.timestamp, warn.message))
    end
    print()
  end

  print("PERFORMANCE SUMMARY")
  print(string.rep("-", 70))
  if #stats.performance > 0 then
    local last_perf = stats.performance[#stats.performance]
    print("Seeds tested:    " .. stats.total_seeds_tested)
    print("Seeds found:     " .. stats.seeds_found)
    if last_perf.data.seeds_per_sec then
      print("Speed:           " .. last_perf.data.seeds_per_sec .. " seeds/sec")
    end
    if last_perf.data.gpu_enabled then
      print("GPU enabled:     " .. last_perf.data.gpu_enabled)
    end
    if last_perf.data.memory_mb then
      print("Memory used:     " .. last_perf.data.memory_mb .. " MB")
    end
  else
    print("No performance data found")
  end
  print()

  -- Hot paths analysis
  local hot_paths = {}
  for path, count in pairs(stats.hot_paths) do
    table.insert(hot_paths, { path = path, count = count })
  end
  table.sort(hot_paths, function(a, b)
    return a.count > b.count
  end)

  if #hot_paths > 0 then
    print("HOT CODE PATHS")
    print(string.rep("-", 70))
    for i = 1, math.min(10, #hot_paths) do
      local hp = hot_paths[i]
      print(string.format("%-40s: %6d calls", hp.path, hp.count))
    end
    print()
  end

  -- Pattern detection
  print("PATTERNS DETECTED")
  print(string.rep("-", 70))

  -- Check for performance degradation
  if #stats.performance >= 2 then
    local first_speed = tonumber(stats.performance[1].data.seeds_per_sec) or 0
    local last_speed = tonumber(
      stats.performance[#stats.performance].data.seeds_per_sec
    ) or 0

    if last_speed < first_speed * 0.5 then
      print(
        "⚠ Performance degradation detected: speed dropped by "
          .. string.format("%.1f%%", (1 - last_speed / first_speed) * 100)
      )
    end
  end

  -- Check for repeated errors
  local error_counts = {}
  for _, err in ipairs(stats.errors) do
    error_counts[err.message] = (error_counts[err.message] or 0) + 1
  end

  for msg, count in pairs(error_counts) do
    if count > 3 then
      print("⚠ Repeated error (" .. count .. "x): " .. msg)
    end
  end

  -- Check for memory leaks
  if #stats.performance >= 2 then
    local first_mem = tonumber(stats.performance[1].data.memory_mb) or 0
    local last_mem = tonumber(
      stats.performance[#stats.performance].data.memory_mb
    ) or 0

    if last_mem > first_mem * 2 then
      print(
        "⚠ Possible memory leak: memory increased from "
          .. string.format("%.1f MB to %.1f MB", first_mem, last_mem)
      )
    end
  end

  print()
  print(string.rep("=", 70))
  print("Analysis complete")
end

-- Main execution
local args = { ... }
local log_path = args[1] or "brainstorm.log"

if args[1] == "--help" then
  print("Usage: lua analyze_logs.lua [log_file]")
  print("  Analyzes Brainstorm log files for patterns and issues")
  print("  Default: brainstorm.log")
  os.exit(0)
end

analyze_logs(log_path)
