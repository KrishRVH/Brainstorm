-- Centralized logging system for Brainstorm
-- Provides structured logging with levels, timestamps, and file output

local Logger = {}
Logger.__index = Logger

-- Log levels
Logger.LEVELS = {
  TRACE = 1,
  DEBUG = 2,
  INFO = 3,
  WARN = 4,
  ERROR = 5,
  FATAL = 6,
}

-- Level names for output
Logger.LEVEL_NAMES = {
  [1] = "TRACE",
  [2] = "DEBUG",
  [3] = "INFO",
  [4] = "WARN",
  [5] = "ERROR",
  [6] = "FATAL",
}

-- ANSI color codes for console output
Logger.COLORS = {
  [1] = "\27[90m", -- TRACE: Gray
  [2] = "\27[36m", -- DEBUG: Cyan
  [3] = "\27[32m", -- INFO: Green
  [4] = "\27[33m", -- WARN: Yellow
  [5] = "\27[31m", -- ERROR: Red
  [6] = "\27[35m", -- FATAL: Magenta
  RESET = "\27[0m",
}

-- Create new logger instance
function Logger:new(name, config)
  local instance = setmetatable({}, self)
  instance.name = name or "Brainstorm"
  instance.config = config or {}

  -- Configuration
  instance.level = instance.config.level or Logger.LEVELS.INFO
  instance.file_path = instance.config.file_path
  instance.max_file_size = instance.config.max_file_size or (1024 * 1024) -- 1MB
  instance.enable_colors = instance.config.enable_colors ~= false
  instance.enable_timestamp = instance.config.enable_timestamp ~= false
  instance.enable_caller = instance.config.enable_caller -- Show file:line
  instance.buffer = {}
  instance.buffer_size = instance.config.buffer_size or 100

  -- Performance tracking
  instance.call_counts = {}
  instance.timing = {}

  -- Open log file if specified
  if instance.file_path then
    instance:rotate_log_if_needed()
  end

  return instance
end

-- Get timestamp string
function Logger:get_timestamp()
  if not self.enable_timestamp then
    return ""
  end
  return os.date("%Y-%m-%d %H:%M:%S")
end

-- Get caller information
function Logger:get_caller(level)
  if not self.enable_caller then
    return ""
  end

  local info = debug.getinfo(level or 3, "Sl")
  if info then
    local file = info.short_src:match("([^/\\]+)$") or info.short_src
    return string.format("%s:%d", file, info.currentline or 0)
  end
  return ""
end

-- Format log message
function Logger:format(level, message, data)
  local parts = {}

  -- Timestamp
  if self.enable_timestamp then
    table.insert(parts, "[" .. self:get_timestamp() .. "]")
  end

  -- Level
  local level_name = self.LEVEL_NAMES[level] or "?"
  if self.enable_colors and not self.file_path then
    table.insert(
      parts,
      self.COLORS[level] .. "[" .. level_name .. "]" .. self.COLORS.RESET
    )
  else
    table.insert(parts, "[" .. level_name .. "]")
  end

  -- Module name
  table.insert(parts, "[" .. self.name .. "]")

  -- Caller info
  local caller = self:get_caller(4)
  if caller ~= "" then
    table.insert(parts, "[" .. caller .. "]")
  end

  -- Message
  table.insert(parts, message)

  -- Structured data
  if data then
    local data_str
    if type(data) == "table" then
      -- Simple table serialization
      local items = {}
      for k, v in pairs(data) do
        table.insert(items, tostring(k) .. "=" .. tostring(v))
      end
      data_str = "{" .. table.concat(items, ", ") .. "}"
    else
      data_str = tostring(data)
    end
    table.insert(parts, data_str)
  end

  return table.concat(parts, " ")
end

-- Write to file
function Logger:write_to_file(message)
  if not self.file_path then
    return
  end

  local file = io.open(self.file_path, "a")
  if file then
    file:write(message .. "\n")
    file:close()
  end
end

-- Rotate log file if too large
function Logger:rotate_log_if_needed()
  if not self.file_path then
    return
  end

  local file = io.open(self.file_path, "r")
  if file then
    local size = file:seek("end")
    file:close()

    if size > self.max_file_size then
      -- Rotate log
      local backup = self.file_path .. "." .. os.date("%Y%m%d_%H%M%S")
      os.rename(self.file_path, backup)

      -- Keep only last 3 backups
      local pattern = self.file_path:gsub("([%.%-])", "%%%1") .. "%.%d+"
      local backups = {}
      for file in io.popen("ls " .. self.file_path .. ".* 2>/dev/null"):lines() do
        table.insert(backups, file)
      end

      if #backups > 3 then
        table.sort(backups)
        for i = 1, #backups - 3 do
          os.remove(backups[i])
        end
      end
    end
  end
end

-- Core logging function
function Logger:log(level, message, data)
  -- Check level threshold
  if level < self.level then
    return
  end

  -- Track call frequency (for debugging hot paths)
  local caller = self:get_caller(4)
  if caller ~= "" then
    self.call_counts[caller] = (self.call_counts[caller] or 0) + 1
  end

  -- Format message
  local formatted = self:format(level, message, data)

  -- Output to console
  print(formatted)

  -- Write to file
  self:write_to_file(formatted)

  -- Buffer for analysis
  table.insert(self.buffer, {
    timestamp = os.time(),
    level = level,
    message = message,
    data = data,
    caller = caller,
  })

  -- Trim buffer
  if #self.buffer > self.buffer_size then
    table.remove(self.buffer, 1)
  end
end

-- Convenience methods
function Logger:trace(message, data)
  self:log(self.LEVELS.TRACE, message, data)
end

function Logger:debug(message, data)
  self:log(self.LEVELS.DEBUG, message, data)
end

function Logger:info(message, data)
  self:log(self.LEVELS.INFO, message, data)
end

function Logger:warn(message, data)
  self:log(self.LEVELS.WARN, message, data)
end

function Logger:error(message, data)
  self:log(self.LEVELS.ERROR, message, data)
end

function Logger:fatal(message, data)
  self:log(self.LEVELS.FATAL, message, data)
end

-- Performance timing helpers
function Logger:start_timer(name)
  self.timing[name] = os.clock()
end

function Logger:end_timer(name, message)
  if self.timing[name] then
    local elapsed = os.clock() - self.timing[name]
    self:debug(
      message or name .. " completed",
      { duration = string.format("%.3fs", elapsed) }
    )
    self.timing[name] = nil
    return elapsed
  end
  return 0
end

-- Log with rate limiting (for high-frequency events)
function Logger:log_throttled(level, key, message, data, interval)
  interval = interval or 1 -- Default 1 second

  self._throttle = self._throttle or {}
  local now = os.time()

  if
    not self._throttle[key] or (now - self._throttle[key].time) >= interval
  then
    self._throttle[key] = self._throttle[key] or { count = 0 }

    -- Include suppressed count if any
    if self._throttle[key].count > 0 then
      message = message
        .. " (suppressed "
        .. self._throttle[key].count
        .. " similar messages)"
    end

    self:log(level, message, data)
    self._throttle[key].time = now
    self._throttle[key].count = 0
  else
    self._throttle[key].count = self._throttle[key].count + 1
  end
end

-- Structured context logging
function Logger:with_context(context)
  local child =
    Logger:new(self.name .. ":" .. (context.module or "?"), self.config)
  child.context = context
  child.parent = self
  return child
end

-- Get statistics about logging
function Logger:get_stats()
  local stats = {
    total_logs = 0,
    by_level = {},
    hot_paths = {},
  }

  -- Count by level
  for _, entry in ipairs(self.buffer) do
    stats.total_logs = stats.total_logs + 1
    local level_name = self.LEVEL_NAMES[entry.level]
    stats.by_level[level_name] = (stats.by_level[level_name] or 0) + 1
  end

  -- Find hot paths (most frequent callers)
  for caller, count in pairs(self.call_counts) do
    table.insert(stats.hot_paths, { caller = caller, count = count })
  end
  table.sort(stats.hot_paths, function(a, b)
    return a.count > b.count
  end)

  -- Keep only top 10
  while #stats.hot_paths > 10 do
    table.remove(stats.hot_paths)
  end

  return stats
end

-- Create global logger instance
local global_logger = Logger:new("Brainstorm", {
  level = Logger.LEVELS.DEBUG, -- Will be updated from config
  enable_colors = true,
  enable_timestamp = true,
  enable_caller = false, -- Enable in debug mode
})

-- Export module
return {
  Logger = Logger,
  global = global_logger,
  LEVELS = Logger.LEVELS,

  -- Create module-specific logger
  for_module = function(name)
    return global_logger -- For now, just return global logger
  end,

  -- Convenience functions using global logger
  trace = function(...)
    global_logger:trace(...)
  end,
  debug = function(...)
    global_logger:debug(...)
  end,
  info = function(...)
    global_logger:info(...)
  end,
  warn = function(...)
    global_logger:warn(...)
  end,
  error = function(...)
    global_logger:error(...)
  end,
  fatal = function(...)
    global_logger:fatal(...)
  end,

  -- Configuration helpers
  set_level = function(level)
    global_logger.level = level
  end,
  enable_file_logging = function(path)
    global_logger.file_path = path
    global_logger:rotate_log_if_needed()
  end,
  enable_caller_info = function(enabled)
    global_logger.enable_caller = enabled
  end,

  -- Create module-specific logger
  for_module = function(name)
    return Logger:new(name, global_logger.config)
  end,
}
