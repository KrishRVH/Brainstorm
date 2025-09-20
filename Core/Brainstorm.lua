-- Brainstorm Mod for Balatro
-- High-performance seed filtering and save state management
-- Author: Community Edition v3.0
-- License: MIT

-- Initialize Brainstorm table first
Brainstorm = {}

local lovely = require("lovely")
local nfs = require("nativefs")
local ffi = require("ffi")

-- Initialize logging system
local logger_ok, logger = pcall(require, "Core.logger")
local log

-- Initialize RNG tracing for debugging
local rng_trace_ok, RNGTrace = pcall(require, "Core.rng_trace")
if rng_trace_ok then
  -- Will be activated when needed for debugging
  Brainstorm.RNGTrace = RNGTrace
end

-- Initialize automatic RNG logger for shop generation
local rng_logger_ok, RNGLogger = pcall(require, "Core.rng_logger")
if rng_logger_ok then
  Brainstorm.RNGLogger = RNGLogger
  -- Automatically install hooks when mod loads
  RNGLogger.install()
  print("[Brainstorm] RNG Logger installed - will capture shop RNG values")
end

-- Load manual capture module
local manual_capture_ok, ManualCapture = pcall(require, "Core.manual_capture")
if manual_capture_ok then
  Brainstorm.ManualCapture = ManualCapture
  print("[Brainstorm] Manual capture module loaded - use capture_rng() in console")
end

-- Add console functions to Brainstorm table
Brainstorm.test_basic = function()
    print("===== BASIC TEST =====")
    print("This function works!")
    if G and G.GAME then
        print("Game is loaded: YES")
        if G.GAME.pseudorandom then
            print("Current seed: " .. tostring(G.GAME.pseudorandom.seed))
        end
    else
        print("Game is loaded: NO")
    end
    return true
end

Brainstorm.test_rng_simple = function()
    print("===== RNG TEST =====")
    if not pseudohash then
        print("ERROR: pseudohash not found")
        return false
    end
    
    local result = pseudohash("AAAAAAAA")
    print(string.format("pseudohash('AAAAAAAA') = %.17g", result))
    print("Expected: 0.43257138351543745")
    
    if pseudoseed and G and G.GAME and G.GAME.pseudorandom then
        local old_state = {}
        for k, v in pairs(G.GAME.pseudorandom) do
            old_state[k] = v
        end
        
        G.GAME.pseudorandom = {seed = "AAAAAAAA", hashed = result}
        local seed_result = pseudoseed("Voucher")
        print(string.format("pseudoseed('Voucher') = %.17g", seed_result))
        print("Expected: 0.46530388624939389")
        
        G.GAME.pseudorandom = old_state
    end
    
    return true
end

_G.trace_to_console = function()
    print("===== TRACING TO CONSOLE =====")
    
    if not G or not G.GAME or not G.GAME.pseudorandom then
        print("ERROR: Game not initialized")
        return false
    end
    
    local test_seeds = {"AAAAAAAA", "00000000", "7NTPKW6P"}
    
    for _, seed in ipairs(test_seeds) do
        print("\n--- Testing seed: " .. seed .. " ---")
        
        -- Save state
        local old_state = {}
        for k, v in pairs(G.GAME.pseudorandom) do
            old_state[k] = v
        end
        
        -- Set test seed
        local hashed = pseudohash(seed)
        G.GAME.pseudorandom = {seed = seed, hashed = hashed}
        
        print(string.format("  hashed = %.17g", hashed))
        
        -- Test RNG calls
        local v = pseudoseed("Voucher")
        print(string.format("  pseudoseed('Voucher') = %.17g", v))
        
        local p1 = pseudoseed("shop_pack1")
        print(string.format("  pseudoseed('shop_pack1') = %.17g", p1))
        
        local p2 = pseudoseed("shop_pack1")
        print(string.format("  pseudoseed('shop_pack1') #2 = %.17g", p2))
        
        local ts = pseudoseed("Tag_small")
        print(string.format("  pseudoseed('Tag_small') = %.17g", ts))
        
        local tb = pseudoseed("Tag_big")
        print(string.format("  pseudoseed('Tag_big') = %.17g", tb))
        
        -- Restore state
        G.GAME.pseudorandom = old_state
    end
    
    print("\n===== TRACE COMPLETE =====")
    return true
end
if logger_ok then
  log = logger.for_module("Brainstorm")
else
  -- Fallback if logger not available
  log = {
    trace = function(msg, data) end,
    debug = function(msg, data)
      if Brainstorm.debug and Brainstorm.debug.enabled then
        print("[DEBUG] " .. tostring(msg))
      end
    end,
    info = function(msg, data)
      print("[INFO] " .. tostring(msg))
    end,
    warn = function(msg, data)
      print("[WARN] " .. tostring(msg))
    end,
    error = function(msg, data)
      print("[ERROR] " .. tostring(msg))
    end,
    start_timer = function() end,
    end_timer = function() end,
    log_throttled = function() end,
  }
end

-- Brainstorm table already initialized at top of file

-- Mod version
Brainstorm.VERSION = "Brainstorm v3.0.0"

-- ============================================================================
-- DEBUG LOGGING SYSTEM
-- ============================================================================

Brainstorm.debug_log = function(module, format, ...)
  if not Brainstorm.config or not Brainstorm.config.debug_enabled then
    return
  end
  
  -- Capture varargs before pcall (can't use ... inside nested function)
  local args = {...}
  
  local ok, msg = pcall(function()
    local timestamp = os.date("%H:%M:%S")
    local message = string.format(format, unpack(args))
    local full = string.format("[%s] [%-12s] %s", timestamp, module, message)
    
    -- Console (game log)
    print(full)
    
    -- Primary path
    local path1 = (Brainstorm.PATH or ".") .. "/debug_full.log"
    local f1 = io.open(path1, "a")
    if f1 then 
      f1:write(full .. "\n")
      f1:close()
    end
    
    -- Secondary path (Windows Roaming)
    local appdata = os.getenv("APPDATA")
    if appdata then
      local path2 = appdata .. "\\Balatro\\Mods\\Brainstorm\\debug_full.log"
      local f2 = io.open(path2, "a")
      if f2 then 
        f2:write(full .. "\n")
        f2:close()
      end
    end
  end)
  
  if not ok then
    -- Last-ditch: console error
    print(string.format("[DEBUG_LOG_FAIL] %s", tostring(msg)))
  end
end

Brainstorm.debug_hex = function(module, label, data)
  if not Brainstorm.config or not Brainstorm.config.debug_enabled then
    return
  end
  
  Brainstorm.debug_log(module, "%s (%d bytes):", label, #data)
  local hex = ""
  for i = 1, #data do
    hex = hex .. string.format("%02x ", string.byte(data, i))
    if i % 16 == 0 then
      Brainstorm.debug_log(module, "  %s", hex)
      hex = ""
    end
  end
  if hex ~= "" then
    Brainstorm.debug_log(module, "  %s", hex)
  end
end

Brainstorm.debug_assert = function(module, name, expected, actual)
  if not Brainstorm.config or not Brainstorm.config.debug_enabled then
    return
  end
  
  if expected ~= actual then
    Brainstorm.debug_log(module, "ASSERTION FAILED: %s - expected %s, got %s",
                         name, tostring(expected), tostring(actual))
  else
    Brainstorm.debug_log(module, "Assertion passed: %s = %s", name, tostring(actual))
  end
end

Brainstorm.debug_timer = function(module, operation)
  if not Brainstorm.config or not Brainstorm.config.debug_enabled then
    return { stop = function() end }
  end
  
  local start_time = os.clock()
  Brainstorm.debug_log(module, "Starting: %s", operation)
  
  return {
    stop = function()
      local elapsed = (os.clock() - start_time) * 1000  -- Convert to ms
      Brainstorm.debug_log(module, "Completed: %s (took %.2f ms)", operation, elapsed)
    end
  }
end

-- Reserved for Steammodded compatibility
Brainstorm.SMODS = nil

Brainstorm.config = {
  enable = true,
  keybind_autoreroll = "r",
  keybinds = {
    options = "t",
    modifier = "lctrl",
    f_reroll = "r",
    a_reroll = "a",
    save_state = "z",
    load_state = "x",
  },
  ar_filters = {
    pack = {},
    pack_id = 1,
    voucher_name = "",
    voucher_id = 1,
    tag_name = "tag_charm",
    tag_id = 2,
    tag2_name = "",
    tag2_id = 1,
    soul_skip = 1,
    inst_observatory = false,
    inst_perkeo = false,
  },
  ar_prefs = {
    spf_id = 3,
    spf_int = 1000,
    face_count = 0,
    suit_ratio_id = 1,
    suit_ratio_percent = "Disabled",
    suit_ratio_decimal = 0,
  },
  debug_enabled = true,  -- ENABLE COMPREHENSIVE DEBUG LOGGING
}

-- Auto-reroll state management
-- Tracks the state of automatic seed rerolling
Brainstorm.ar_timer = 0 -- Time accumulator for reroll intervals
Brainstorm.ar_frames = 0 -- Frame counter for UI display timing
Brainstorm.ar_text = nil -- UI text element for "Rerolling..." message
Brainstorm.ar_active = false -- Whether auto-reroll is currently active

-- Debug statistics for performance monitoring and analysis
-- Helps users understand search difficulty and optimize filters
Brainstorm.debug = {
  enabled = false, -- Will be set from config
  seeds_tested = 0, -- Total seeds evaluated
  seeds_found = 0, -- Seeds matching DLL filters
  start_time = 0, -- When search started (os.clock)
  rejection_reasons = { -- Why seeds were rejected
    face_cards = 0, -- Not enough face cards
    suit_ratio = 0, -- Suit distribution too even
    dll_filter = 0, -- Failed DLL criteria
  },
  log_file = nil, -- Debug log file handle
  log_path = "C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm\\lua_debug.log",
  distributions = { -- Statistical distributions found
    face_cards = {}, -- Histogram of face card counts
    suit_ratios = {}, -- Histogram of suit ratios
  },
  last_report_time = 0, -- Last time we printed progress
  highest_suit_ratio = 0, -- Best ratio seen so far
  highest_face_count = 0, -- Most face cards seen
}

-- Static lookup tables for better performance
-- Avoids repeated string comparisons in hot loops
Brainstorm.RATIO_MAP = {
  ["Disabled"] = 0,
  ["50%"] = 0.5,
  ["60%"] = 0.6,
  ["70%"] = 0.7,
  ["75%"] = 0.75,
  ["80%"] = 0.80, -- 80% is mathematically impossible but kept for compatibility
}

-- Face card lookup for O(1) checking instead of multiple string comparisons
Brainstorm.FACE_CARDS = {
  ["Jack"] = true,
  ["Queen"] = true,
  ["King"] = true,
}

-- Map internal keys to DLL-expected names (must match stringToItem in DLL)
Brainstorm.TAG_TO_DLL = {
  ["tag_charm"] = "Charm Tag",
  ["tag_double"] = "Double Tag",
  ["tag_investment"] = "Investment Tag",
  ["tag_voucher"] = "Voucher Tag",
  ["tag_boss"] = "Boss Tag",
  ["tag_juggle"] = "Juggle Tag",
  ["tag_coupon"] = "Coupon Tag",
  ["tag_economy"] = "Economy Tag",
  ["tag_uncommon"] = "Uncommon Tag",
  ["tag_rare"] = "Rare Tag",
  ["tag_negative"] = "Negative Tag",
  ["tag_foil"] = "Foil Tag",
  ["tag_holo"] = "Holographic Tag",
  ["tag_poly"] = "Polychrome Tag",
  ["tag_buffoon"] = "Buffoon Tag",
  ["tag_handy"] = "Handy Tag",
  ["tag_garbage"] = "Garbage Tag",
  ["tag_ethereal"] = "Ethereal Tag",
  ["tag_standard"] = "Standard Tag",
  ["tag_top_up"] = "Top-up Tag",
  ["tag_d_six"] = "D6 Tag",
  ["tag_orbital"] = "Orbital Tag",
  ["tag_meteor"] = "Meteor Tag",
  ["tag_skip"] = "Speed Tag",
}

Brainstorm.VOUCHER_TO_DLL = {
  ["v_overstock_norm"] = "Overstock",
  ["v_clearance_sale"] = "Clearance Sale",
  ["v_hone"] = "Hone",
  ["v_reroll_surplus"] = "Reroll Surplus",
  ["v_crystal_ball"] = "Crystal Ball",
  ["v_telescope"] = "Telescope",
  ["v_grabber"] = "Grabber",
  ["v_wasteful"] = "Wasteful",
  ["v_tarot_merchant"] = "Tarot Merchant",
  ["v_planet_merchant"] = "Planet Merchant",
  ["v_seed_money"] = "Seed Money",
  ["v_blank"] = "Blank",
  ["v_magic_trick"] = "Magic Trick",
  ["v_hieroglyph"] = "Hieroglyph",
  ["v_directors_cut"] = "Directors Cut",
  ["v_retcon"] = "Retcon",
  ["v_paint_brush"] = "Paint Brush",
  ["v_overstock_plus"] = "Overstock Plus",
  ["v_liquidation"] = "Liquidation",
  ["v_glow_up"] = "Glow Up",
  ["v_reroll_glut"] = "Reroll Glut",
  ["v_omen_globe"] = "Omen Globe",
  ["v_observatory"] = "Observatory",
  ["v_nacho_tong"] = "Nacho Tong",
  ["v_recyclomancy"] = "Recyclomancy",
  ["v_money_tree"] = "Money Tree",
  ["v_antimatter"] = "Antimatter",
  ["v_illusion"] = "Illusion",
  ["v_petroglyph"] = "Petroglyph",
  ["v_curator"] = "Curator",
}

-- Mapping from pack keys to DLL-expected names
Brainstorm.PACK_TO_DLL = {
  ["p_arcana"] = "Arcana Pack",
  ["p_arcana_normal"] = "Arcana Pack",
  ["p_arcana_jumbo"] = "Jumbo Arcana Pack",
  ["p_arcana_mega"] = "Mega Arcana Pack",
  ["p_celestial"] = "Celestial Pack",
  ["p_celestial_normal"] = "Celestial Pack",
  ["p_celestial_jumbo"] = "Jumbo Celestial Pack",
  ["p_celestial_mega"] = "Mega Celestial Pack",
  ["p_standard"] = "Standard Pack",
  ["p_standard_normal"] = "Standard Pack",
  ["p_standard_jumbo"] = "Jumbo Standard Pack",
  ["p_standard_mega"] = "Mega Standard Pack",
  ["p_buffoon"] = "Buffoon Pack",
  ["p_buffoon_normal"] = "Buffoon Pack",
  ["p_buffoon_jumbo"] = "Jumbo Buffoon Pack",
  ["p_buffoon_mega"] = "Mega Buffoon Pack",
  ["p_spectral"] = "Spectral Pack",
  ["p_spectral_normal"] = "Spectral Pack",
  ["p_spectral_jumbo"] = "Jumbo Spectral Pack",
  ["p_spectral_mega"] = "Mega Spectral Pack",
}

-- Constants
Brainstorm.AR_INTERVAL = 0.01 -- Seconds between reroll attempts (100 Hz)

-- Performance optimization: Cache frequently used functions
-- This avoids table lookups in hot code paths
local string_format = string.format
local string_lower = string.lower
local math_floor = math.floor
local math_max = math.max
local math_min = math.min
local table_insert = table.insert
local table_sort = table.sort
local os_clock = os.clock
local pcall = pcall

-- Random seed generation constants
-- Seed generation factors using prime-based constants for good distribution
-- These values minimize correlation between cursor position and generated seeds:
--   X factor (~1/3): Spreads horizontal mouse movement across seed space
--   Y factor (~7/8): Ensures vertical movement contributes strongly
--   Time factor (~2/5): Incorporates temporal variation for entropy
local SEED_X_FACTOR = 0.33411983 -- Prime-derived for X axis distribution
local SEED_Y_FACTOR = 0.874146 -- Prime-derived for Y axis spread
local SEED_TIME_FACTOR = 0.412311010 -- Prime-derived for time entropy

-- Find the Brainstorm mod directory
-- Searches for a directory containing "brainstorm" (case-insensitive)
local function find_brainstorm_directory(directory)
  for _, item in ipairs(nfs.getDirectoryItems(directory)) do
    local itemPath = directory .. "/" .. item
    if
      nfs.getInfo(itemPath, "directory")
      and string_lower(item):find("brainstorm")
    then
      return itemPath
    end
  end
  return nil
end

local function file_exists(file_path)
  return nfs.getInfo(file_path) ~= nil
end

-- Load configuration from file with backward compatibility
-- Uses deep merge to preserve new fields when loading old configs
function Brainstorm.load_config()
  local config_path = Brainstorm.PATH .. "/config.lua"
  if not file_exists(config_path) then
    Brainstorm.write_config() -- Create default config
  else
    local config_file, err = nfs.read(config_path)
    if not config_file then
      log.error(
        "Failed to read config file",
        { error = err or "unknown error" }
      )
      return
    end
    -- STR_UNPACK is a Balatro function for deserializing Lua tables
    local success, loaded_config = pcall(STR_UNPACK, config_file)
    if success and loaded_config then
      -- Deep merge loaded config with defaults to handle new fields
      -- This ensures backward compatibility when new config options are added
      local function deep_merge(target, source)
        for key, value in pairs(source) do
          if type(value) == "table" and type(target[key]) == "table" then
            deep_merge(target[key], value) -- Recursively merge nested tables
          else
            target[key] = value -- Overwrite or add new value
          end
        end
      end

      deep_merge(Brainstorm.config, loaded_config)

      -- Ensure new fields have default values if missing
      Brainstorm.config.ar_prefs.face_count = Brainstorm.config.ar_prefs.face_count
        or 0
      Brainstorm.config.ar_prefs.suit_ratio_id = Brainstorm.config.ar_prefs.suit_ratio_id
        or 1
      Brainstorm.config.ar_prefs.suit_ratio_percent = Brainstorm.config.ar_prefs.suit_ratio_percent
        or "Disabled"

      -- Map suit ratio percentage to decimal value (use static table)
      Brainstorm.config.ar_prefs.suit_ratio_decimal = Brainstorm.RATIO_MAP[Brainstorm.config.ar_prefs.suit_ratio_percent]
        or 0
    end
  end
end

function Brainstorm.write_config()
  local config_path = Brainstorm.PATH .. "/config.lua"
  -- STR_PACK is a Balatro function for serializing Lua tables
  local success, packed = pcall(STR_PACK, Brainstorm.config)
  if success and packed then
    local write_success, err = nfs.write(config_path, packed)
    if not write_success then
      log.error(
        "Failed to write config file",
        { error = err or "unknown error" }
      )
    end
  end
end

function Brainstorm.init()
  print("[Brainstorm] Initializing...")

  Brainstorm.PATH = find_brainstorm_directory(lovely.mod_dir)
  if not Brainstorm.PATH then
    print("[Brainstorm] ERROR: Could not find Brainstorm directory")
    return false
  end

  print("[Brainstorm] Found mod at: " .. Brainstorm.PATH)

  Brainstorm.load_config()
  Brainstorm.debug.enabled = Brainstorm.config.debug_enabled or false

  -- Initialize logger with file path only if debug is enabled
  if logger_ok and logger.global and Brainstorm.debug.enabled then
    logger.global.file_path = Brainstorm.PATH .. "/brainstorm.log"
    logger.global:rotate_log_if_needed()
    -- Recreate the module logger with file path configured
    log = logger.for_module("Brainstorm")
    log:info(
      "Brainstorm initialized",
      { version = Brainstorm.VERSION, path = Brainstorm.PATH }
    )
  end

  -- Load UI with error handling
  local ui_path = Brainstorm.PATH .. "/UI/ui.lua"
  local ui_content = nfs.read(ui_path)
  if not ui_content then
    print("[Brainstorm] ERROR: Could not read UI file")
    return false
  end

  local ui_func, err = load(ui_content)
  if not ui_func then
    print("[Brainstorm] ERROR: Failed to load UI: " .. tostring(err))
    return false
  end

  local success, ui_err = pcall(ui_func)
  if not success then
    print("[Brainstorm] ERROR: Failed to initialize UI: " .. tostring(ui_err))
    return false
  end

  
  return true
end

-- RNG Trace functionality for debugging
function Brainstorm.activate_rng_trace()
  if Brainstorm.RNGTrace then
    Brainstorm.RNGTrace.install_all_patches()
    Brainstorm.save_state_alert("RNG Tracing Activated")
    return true
  end
  return false
end

function Brainstorm.generate_test_traces()
  if Brainstorm.RNGTrace then
    Brainstorm.RNGTrace.generate_test_traces()
    Brainstorm.save_state_alert("Test Traces Generated")
    return true
  end
  return false
end

-- Simple global functions for console access (lowercase for console compatibility)
function activate_rng_trace()
  return Brainstorm.activate_rng_trace()
end

function generate_test_traces()
  return Brainstorm.generate_test_traces()
end

-- Even simpler lowercase versions
function rng_trace()
  if Brainstorm and Brainstorm.RNGTrace then
    Brainstorm.RNGTrace.install_all_patches()
    print("[RNG] Patches installed")
    Brainstorm.RNGTrace.generate_test_traces()
    print("[RNG] Test traces generated")
    print("[RNG] Check: C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm\\rng_trace.jsonl")
    return true
  end
  print("[RNG] Failed - Brainstorm or RNGTrace not available")
  return false
end

-- Ultra simple test
function test_rng()
  print("Test RNG function called!")
  if G and G.GAME and G.GAME.pseudorandom then
    print("Current seed: " .. tostring(G.GAME.pseudorandom.seed))
  end
  return true
end

-- Get pool JSON for DLL (C++ expects specific format)
function Brainstorm.get_pool_json()
  -- C++ parser expects named contexts, not an array
  local pools = {
    voucher = {
      ctx_key = "Voucher",
      weighted = false,
      items = {"Overstock", "Clearance Sale", "Tarot Merchant", "Planet Merchant", "Hone", "Reroll Surplus", "Crystal Ball", "Telescope"}
    },
    pack1 = {
      ctx_key = "shop_pack1",
      weighted = false,
      items = {"Arcana Pack", "Celestial Pack", "Spectral Pack", "Standard Pack", "Buffoon Pack"}
    },
    pack2 = {
      ctx_key = "shop_pack2", 
      weighted = false,
      items = {"Arcana Pack", "Celestial Pack", "Spectral Pack", "Standard Pack", "Buffoon Pack"}
    },
    tag_small = {
      ctx_key = "Tag_small",
      weighted = false,
      items = {"Uncommon Tag", "Rare Tag", "Negative Tag", "Foil Tag", "Holographic Tag", "Polychrome Tag", "Investment Tag", "Voucher Tag", "Boss Tag", "Standard Tag", "Charm Tag", "Meteor Tag", "Buffoon Tag", "Handy Tag", "Garbage Tag", "Ethereal Tag", "Coupon Tag", "Double Tag", "Juggle Tag", "D6 Tag", "Top-up Tag", "Speed Tag", "Orbital Tag", "Economy Tag"}
    },
    tag_big = {
      ctx_key = "Tag_big",
      weighted = false,
      items = {"Uncommon Tag", "Rare Tag", "Negative Tag", "Foil Tag", "Holographic Tag", "Polychrome Tag", "Investment Tag", "Voucher Tag", "Boss Tag", "Standard Tag", "Charm Tag", "Meteor Tag", "Buffoon Tag", "Handy Tag", "Garbage Tag", "Ethereal Tag", "Coupon Tag", "Double Tag", "Juggle Tag", "D6 Tag", "Top-up Tag", "Speed Tag", "Orbital Tag", "Economy Tag"}
    }
  }
  
  -- Try json library first
  local ok, json = pcall(require, "json")
  if ok and json and json.encode then
    return json.encode(pools)
  end
  
  -- Fallback: manual JSON construction
  local function encode_array(arr)
    local str = "["
    for i, v in ipairs(arr) do
      if i > 1 then str = str .. "," end
      str = str .. '"' .. v .. '"'
    end
    return str .. "]"
  end
  
  local json_str = '{'
  json_str = json_str .. '"voucher":{"ctx_key":"Voucher","weighted":false,"items":' .. encode_array(pools.voucher.items) .. '},'
  json_str = json_str .. '"pack1":{"ctx_key":"shop_pack1","weighted":false,"items":' .. encode_array(pools.pack1.items) .. '},'
  json_str = json_str .. '"pack2":{"ctx_key":"shop_pack2","weighted":false,"items":' .. encode_array(pools.pack2.items) .. '},'
  json_str = json_str .. '"tag_small":{"ctx_key":"Tag_small","weighted":false,"items":' .. encode_array(pools.tag_small.items) .. '},'
  json_str = json_str .. '"tag_big":{"ctx_key":"Tag_big","weighted":false,"items":' .. encode_array(pools.tag_big.items) .. '}'
  json_str = json_str .. '}'
  
  return json_str
end

-- Function to update GPU pools with current game state
function Brainstorm.update_pools()
  if not immolate or not immolate.brainstorm_update_pools then
    return false
  end
  
  -- Collect current pool data from the game
  local pool_data = {
    contexts = {}
  }
  
  -- Helper to extract pool items
  local function get_pool_items(pool_key)
    local items = {}
    local weights = {}
    
    if G.P_CENTER_POOLS and G.P_CENTER_POOLS[pool_key] then
      for _, item in ipairs(G.P_CENTER_POOLS[pool_key]) do
        if type(item) == "table" then
          table.insert(items, item.key or "unknown")
          -- Check for weight in item
          if item.weight then
            table.insert(weights, item.weight)
          end
        elseif type(item) == "string" then
          table.insert(items, item)
        end
      end
    end
    
    return items, weights
  end
  
  -- Voucher pool
  local voucher_items, voucher_weights = get_pool_items("Voucher")
  pool_data.contexts.voucher = {
    ctx_key = "Voucher",
    items = voucher_items,
    weights = #voucher_weights > 0 and voucher_weights or nil
  }
  
  -- Pack pools (shop_pack1 for both pack1 and pack2)
  local pack_items, pack_weights = get_pool_items("Booster")
  pool_data.contexts.pack1 = {
    ctx_key = "shop_pack1", 
    items = pack_items,
    weights = #pack_weights > 0 and pack_weights or nil
  }
  pool_data.contexts.pack2 = {
    ctx_key = "shop_pack1",  -- Same context for second pack
    items = pack_items,
    weights = #pack_weights > 0 and pack_weights or nil
  }
  
  -- Tag pools
  local tag_items, tag_weights = get_pool_items("Tag")
  pool_data.contexts.tag_small = {
    ctx_key = "Tag_small",
    items = tag_items,
    weights = #tag_weights > 0 and tag_weights or nil
  }
  pool_data.contexts.tag_big = {
    ctx_key = "Tag_big",
    items = tag_items,
    weights = #tag_weights > 0 and tag_weights or nil
  }
  
  -- Convert to JSON
  local success, json = pcall(function()
    -- Simple JSON serialization
    local function to_json(t)
      if type(t) == "table" then
        local is_array = #t > 0
        local result = is_array and "[" or "{"
        local first = true
        
        if is_array then
          for i, v in ipairs(t) do
            if not first then result = result .. "," end
            result = result .. to_json(v)
            first = false
          end
        else
          for k, v in pairs(t) do
            if not first then result = result .. "," end
            result = result .. '"' .. tostring(k) .. '":' .. to_json(v)
            first = false
          end
        end
        
        return result .. (is_array and "]" or "}")
      elseif type(t) == "string" then
        return '"' .. t:gsub('"', '\\"') .. '"'
      elseif type(t) == "number" then
        return tostring(t)
      elseif type(t) == "boolean" then
        return tostring(t)
      elseif t == nil then
        return "null"
      else
        return '"' .. tostring(t) .. '"'
      end
    end
    
    return to_json(pool_data)
  end)
  
  if success and json then
    -- Send to DLL
    local ok = pcall(function()
      immolate.brainstorm_update_pools(json)
    end)
    
    if ok and log then
      log:info("Updated GPU pools", {
        voucher_count = #voucher_items,
        pack_count = #pack_items,
        tag_count = #tag_items
      })
    end
    
    return ok
  end
  
  return false
end

-- Save state functionality
local save_state_keys = { "1", "2", "3", "4", "5" }

function Brainstorm.save_state_alert(text)
  G.E_MANAGER:add_event(Event({
    trigger = "after",
    delay = 0.4,
    func = function()
      attention_text({
        text = text,
        scale = 0.7,
        hold = 3,
        major = G.STAGE == G.STAGES.RUN and G.play or G.title_top,
        backdrop_colour = G.C.SECONDARY_SET.Tarot,
        align = "cm",
        offset = { x = 0, y = -3.5 },
        silent = true,
      })
      G.E_MANAGER:add_event(Event({
        trigger = "after",
        delay = 0.06 * G.SETTINGS.GAMESPEED,
        blockable = false,
        blocking = false,
        func = function()
          play_sound("other1", 0.76, 0.4)
          return true
        end,
      }))
      return true
    end,
  }))
end

function Brainstorm.save_game_state(slot)
  if G.STAGE == G.STAGES.RUN then
    local save_path = G.SETTINGS.profile
      .. "/"
      .. "save_state_"
      .. slot
      .. ".jkr"
    local success, err = pcall(compress_and_save, save_path, G.ARGS.save_run)
    if success then
      Brainstorm.save_state_alert("Saved state to slot [" .. slot .. "]")
      return true
    else
      print("[Brainstorm] Failed to save state: " .. tostring(err))
      Brainstorm.save_state_alert("Failed to save state")
      return false
    end
  end
  return false
end

function Brainstorm.load_game_state(slot)
  local save_path = G.SETTINGS.profile .. "/" .. "save_state_" .. slot .. ".jkr"
  local success, saved_game = pcall(get_compressed, save_path)

  if success and saved_game then
    local unpack_success, saved_data = pcall(STR_UNPACK, saved_game)
    if unpack_success and saved_data then
      G:delete_run()
      G.SAVED_GAME = saved_data
      G:start_run({ savetext = G.SAVED_GAME })
      Brainstorm.save_state_alert("Loaded state from slot [" .. slot .. "]")
      return true
    else
      print("[Brainstorm] Failed to unpack save: " .. tostring(saved_data))
      Brainstorm.save_state_alert("Corrupted save in slot [" .. slot .. "]")
      return false
    end
  else
    Brainstorm.save_state_alert("No save in slot [" .. slot .. "]")
    return false
  end
end

local key_press_update_ref = Controller.key_press_update
function Controller:key_press_update(key, dt)
  key_press_update_ref(self, key, dt)

  -- Safety check: ensure Brainstorm is initialized
  if
    not Brainstorm
    or not Brainstorm.config
    or not Brainstorm.config.keybinds
  then
    return
  end

  local keybinds = Brainstorm.config.keybinds

  -- Save state functionality
  for _, slot in ipairs(save_state_keys) do
    if key == slot then
      -- Save state
      if love.keyboard.isDown(keybinds.save_state) then
        Brainstorm.save_game_state(slot)
      end
      -- Load state
      if love.keyboard.isDown(keybinds.load_state) then
        Brainstorm.load_game_state(slot)
      end
    end
  end

  -- Original reroll functionality
  if love.keyboard.isDown(keybinds.modifier) then
    if key == keybinds.f_reroll then
      Brainstorm.reroll()
    elseif key == keybinds.a_reroll then
      print("[Brainstorm] Ctrl+A pressed - toggling auto-reroll")
      -- Wrap in pcall to catch any errors
      local success, err = pcall(function()
        if Brainstorm.ar_active then
          Brainstorm.stop_auto_reroll(false)
        else
          Brainstorm.ar_active = true

          -- Print helpful message for dual tag searches
          if
            Brainstorm.config.ar_filters.tag2_name
            and Brainstorm.config.ar_filters.tag2_name ~= ""
          then
            -- Safely get tag display names
            local tag1_display = Brainstorm.config.ar_filters.tag_name
            local tag2_display = Brainstorm.config.ar_filters.tag2_name

            -- Try to localize if the function exists and is safe
            if localize and type(localize) == "function" then
              local ok1, localized1 = pcall(localize, {
                type = "name_text",
                set = "Tag",
                key = Brainstorm.config.ar_filters.tag_name,
              })
              if ok1 and localized1 then
                tag1_display = localized1
              end

              local ok2, localized2 = pcall(localize, {
                type = "name_text",
                set = "Tag",
                key = Brainstorm.config.ar_filters.tag2_name,
              })
              if ok2 and localized2 then
                tag2_display = localized2
              end
            end

            if
              Brainstorm.config.ar_filters.tag_name
              == Brainstorm.config.ar_filters.tag2_name
            then
              print(
                string.format(
                  "[Brainstorm] Searching for DOUBLE %s tags...",
                  tag1_display
                )
              )
              print(
                "[Brainstorm] This is extremely rare! May take 5-30 seconds depending on the tag."
              )
            else
              print(
                string.format(
                  "[Brainstorm] Searching for %s + %s tags...",
                  tag1_display,
                  tag2_display
                )
              )
              print(
                "[Brainstorm] Dual tag combinations can take 5-20 seconds to find."
              )
            end
            log.info(
              "Order doesn't matter - either tag can be in either blind position."
            )
          end
        end
      end) -- End of pcall

      if not success then
        log.error("ERROR in auto-reroll toggle", { error = err })
      end
    end
  end
end

-- Check if the current seed has both required tags in the first ante
-- Supports order-agnostic matching and same-tag-twice requirements
-- Returns: true if tags match requirements, false otherwise
function Brainstorm.check_dual_tags()
  local tag1 = Brainstorm.config.ar_filters.tag_name
  local tag2 = Brainstorm.config.ar_filters.tag2_name

  -- If no second tag specified, always pass
  if not tag2 or tag2 == "" then
    return true
  end

  -- Tags are stored in G.GAME.round_resets.blind_tags.Small and .Big
  if
    not G.GAME
    or not G.GAME.round_resets
    or not G.GAME.round_resets.blind_tags
  then
    return false
  end

  local small_blind_tag = G.GAME.round_resets.blind_tags.Small
  local big_blind_tag = G.GAME.round_resets.blind_tags.Big

  -- Track dual tag checks for performance metrics
  if Brainstorm.debug.enabled then
    Brainstorm.debug.dual_tag_checks = (Brainstorm.debug.dual_tag_checks or 0)
      + 1
  end

  -- Special case: Looking for the same tag in BOTH blind positions
  -- Example: Double Investment Tags (extremely rare ~0.1% chance)
  if tag1 == tag2 then
    local both_match = (small_blind_tag == tag1 and big_blind_tag == tag1)
    if both_match then
      if Brainstorm.debug.enabled then
        log:info("Dual tag success: Both blinds have " .. tag1)
        Brainstorm.debug.dual_tag_successes = (
          Brainstorm.debug.dual_tag_successes or 0
        ) + 1
      end
      return true
    end
    return false
  else
    -- Different tags: Check both are present in either order
    -- Example: Investment + Charm (order doesn't matter)
    local has_tag1 = (small_blind_tag == tag1 or big_blind_tag == tag1)
    local has_tag2 = (small_blind_tag == tag2 or big_blind_tag == tag2)

    if has_tag1 and has_tag2 then
      if Brainstorm.debug.enabled then
        log:info("Dual tag success: Both tags found (order-agnostic)")
        Brainstorm.debug.dual_tag_successes = (
          Brainstorm.debug.dual_tag_successes or 0
        ) + 1
      end
      return true
    end
    return false
  end
end

-- Analyze the current deck composition
-- Used for Erratic deck validation (face cards and suit ratios)
-- Returns: Table with deck statistics
function Brainstorm.analyze_deck()
  local deck_summary = {}
  local suit_count = { Hearts = 0, Diamonds = 0, Clubs = 0, Spades = 0 }
  local face_card_count = 0
  local numeric_card_count = 0
  local ace_count = 0
  local unique_card_count = 0
  local face_cards = Brainstorm.FACE_CARDS -- Cache lookup table

  -- Iterate through all cards in the deck
  for _, card in ipairs(G.playing_cards) do
    if card.base then
      local card_value = card.base.value
      local card_suit = card.base.suit
      local card_name = card_value .. " of " .. card_suit
      deck_summary[card_name] = (deck_summary[card_name] or 0) + 1
      suit_count[card_suit] = (suit_count[card_suit] or 0) + 1

      -- Categorizing cards (optimized with lookup table)
      if card_value == "Ace" then
        ace_count = ace_count + 1
      elseif face_cards[card_value] then
        face_card_count = face_card_count + 1
      else
        numeric_card_count = numeric_card_count + 1
      end
    end
  end

  -- Count unique cards
  for _ in pairs(deck_summary) do
    unique_card_count = unique_card_count + 1
  end

  -- Return the analysis result
  return {
    deck_summary = deck_summary,
    suit_count = suit_count,
    face_card_count = face_card_count,
    numeric_card_count = numeric_card_count,
    ace_count = ace_count,
    unique_card_count = unique_card_count,
  }
end

-- Validate if a deck meets the specified requirements
-- Used for Erratic deck filtering based on face cards and suit distribution
-- Parameters:
--   deck_data: Analysis from analyze_deck()
--   min_face_cards: Minimum number of face cards required (0-23)
--   min_aces: Minimum number of aces required (unused currently)
--   dominant_suit_ratio: Required ratio for top 2 suits (0.5 to 0.75)
-- Returns: true if deck is valid, false otherwise
function Brainstorm.is_valid_deck(
  deck_data,
  min_face_cards,
  min_aces,
  dominant_suit_ratio
)
  -- Ensure parameters are not nil by providing default values
  min_face_cards = min_face_cards or 0
  min_aces = min_aces or 0
  dominant_suit_ratio = dominant_suit_ratio or 0

  -- Extract counts from the deck analysis
  local total_cards = #G.playing_cards
  local face_card_count = deck_data.face_card_count or 0
  local ace_count = deck_data.ace_count or 0
  local suit_count = deck_data.suit_count or {}

  -- Track distribution for debugging
  if Brainstorm.debug.enabled then
    local fc_bucket = math_floor(face_card_count / 5) * 5
    Brainstorm.debug.distributions.face_cards[fc_bucket] = (
      Brainstorm.debug.distributions.face_cards[fc_bucket] or 0
    ) + 1

    -- Track highest face count
    if face_card_count > Brainstorm.debug.highest_face_count then
      Brainstorm.debug.highest_face_count = face_card_count
    end
  end

  -- Check Face Cards & Aces
  if face_card_count < min_face_cards then
    if Brainstorm.debug.enabled then
      Brainstorm.debug.rejection_reasons.face_cards = Brainstorm.debug.rejection_reasons.face_cards
        + 1
    end
    return false
  end
  if ace_count < min_aces then
    --print("Deck has", ace_count, "aces, need", min_aces)
    return false
  end

  -- Check suit distribution (only if enabled - 0 means disabled)
  if dominant_suit_ratio and dominant_suit_ratio > 0 then
    local sorted_suits = {}
    for suit, count in pairs(suit_count) do
      table_insert(sorted_suits, { suit = suit, count = count })
    end

    if #sorted_suits > 0 then
      table_sort(sorted_suits, function(a, b)
        return a.count > b.count
      end)

      -- Calculate the combined percentage of the top 2 suits
      -- This represents how "suited" the deck is
      -- Higher values mean more cards of the same suits
      local top_2_suit_count = sorted_suits[1].count
        + (sorted_suits[2] and sorted_suits[2].count or 0)
      local top_2_suit_percentage = top_2_suit_count / total_cards

      -- Track suit ratio distribution
      if Brainstorm.debug.enabled then
        local ratio_bucket = math_floor(top_2_suit_percentage * 10) * 10
        Brainstorm.debug.distributions.suit_ratios[ratio_bucket] = (
          Brainstorm.debug.distributions.suit_ratios[ratio_bucket] or 0
        ) + 1

        -- Track the highest ratio we've seen
        if
          not Brainstorm.debug.highest_suit_ratio
          or top_2_suit_percentage > Brainstorm.debug.highest_suit_ratio
        then
          Brainstorm.debug.highest_suit_ratio = top_2_suit_percentage
        end
      end

      if top_2_suit_percentage < dominant_suit_ratio then
        if Brainstorm.debug.enabled then
          Brainstorm.debug.rejection_reasons.suit_ratio = Brainstorm.debug.rejection_reasons.suit_ratio
            + 1
        end
        return false
      end
    end
  end

  return true
end

function Brainstorm.print_debug_report(success)
  local elapsed = os_clock() - Brainstorm.debug.start_time
  local seeds_per_sec = Brainstorm.debug.seeds_tested / elapsed

  print("========================================")
  print("[Brainstorm Debug Report]")
  print(
    string.format(
      "Result: %s",
      success and "SUCCESS - Seed Found!" or "STOPPED"
    )
  )
  print(string.format("Time elapsed: %.2f seconds", elapsed))
  print(string.format("Seeds tested: %d", Brainstorm.debug.seeds_tested))
  print(string.format("Seeds per second: %.1f", seeds_per_sec))
  print(
    string.format("Seeds found matching DLL: %d", Brainstorm.debug.seeds_found)
  )

  print("\nRejection Reasons:")
  local total_rejections = Brainstorm.debug.rejection_reasons.face_cards
    + Brainstorm.debug.rejection_reasons.suit_ratio
  if total_rejections > 0 then
    print(
      string.format(
        "  Face cards: %d (%.1f%%)",
        Brainstorm.debug.rejection_reasons.face_cards,
        Brainstorm.debug.rejection_reasons.face_cards / total_rejections * 100
      )
    )
    print(
      string.format(
        "  Suit ratio: %d (%.1f%%)",
        Brainstorm.debug.rejection_reasons.suit_ratio,
        Brainstorm.debug.rejection_reasons.suit_ratio / total_rejections * 100
      )
    )
  end

  print("\nFace Card Distribution:")
  for bucket = 0, 30, 5 do
    local count = Brainstorm.debug.distributions.face_cards[bucket] or 0
    if count > 0 then
      print(
        string.format(
          "  %d-%d: %d seeds (%.1f%%)",
          bucket,
          bucket + 4,
          count,
          count / Brainstorm.debug.seeds_tested * 100
        )
      )
    end
  end

  print("\nSuit Ratio Distribution:")
  local max_ratio = 0
  for bucket = 40, 90, 10 do
    local count = Brainstorm.debug.distributions.suit_ratios[bucket] or 0
    if count > 0 then
      print(
        string.format(
          "  %d%%-%d%%: %d seeds (%.1f%%)",
          bucket,
          bucket + 9,
          count,
          count / Brainstorm.debug.seeds_tested * 100
        )
      )
      if bucket > max_ratio then
        max_ratio = bucket
      end
    end
  end

  -- Show exact max values found
  print(
    string.format(
      "\nHighest suit ratio found: %.1f%% (in %d%%-%d%% bucket)",
      (Brainstorm.debug.highest_suit_ratio or 0) * 100,
      max_ratio,
      max_ratio + 9
    )
  )
  print(
    string.format(
      "Highest face count found: %d face cards",
      Brainstorm.debug.highest_face_count or 0
    )
  )
  print(
    string.format(
      "\nTarget suit ratio: %.0f%%",
      (Brainstorm.config.ar_prefs.suit_ratio_decimal or 0) * 100
    )
  )
  print(
    string.format(
      "Target face count: %d",
      Brainstorm.config.ar_prefs.face_count or 0
    )
  )

  -- Recommendation
  if Brainstorm.config.ar_prefs.suit_ratio_decimal >= 0.75 then
    print("\nWARNING: 75% suit ratio is the maximum achievable!")
    print("Consider using 70% or lower for faster results.")
  elseif Brainstorm.config.ar_prefs.suit_ratio_decimal > 0.7 then
    print("\nNOTE: Suit ratios above 70% are extremely rare.")
    print("Expect longer search times.")
  end

  print("========================================")
end

-- Performance metrics are now tracked internally and shown in debug report only

-- Perform a single manual reroll
-- Preserves stake and challenge settings
function Brainstorm.reroll()
  local G = G -- Cache global for slight performance gain

  -- Preserve game state for restart
  G.GAME.viewed_back = nil
  G.run_setup_seed = G.GAME.seeded
  G.challenge_tab = G.GAME and G.GAME.challenge and G.GAME.challenge_tab or nil
  G.forced_seed = G.GAME.seeded and G.GAME.pseudorandom.seed or nil

  local seed = G.run_setup_seed and G.setup_seed or G.forced_seed
  local stake = (
    G.GAME.stake
    or G.PROFILES[G.SETTINGS.profile].MEMORY.stake
    or 1
  ) or 1

  G:delete_run()
  G:start_run({ stake = stake, seed = seed, challenge = G.challenge_tab })
end

-- Hook into the game's update loop for auto-reroll functionality
-- This is called every frame when the game is running
local update_ref = Game.update
function Game:update(dt)
  -- Safely call original update
  if update_ref then
    update_ref(self, dt)
  end

  -- Safety check for Brainstorm
  if not Brainstorm then
    return
  end

  -- Handle auto-reroll if active
  if Brainstorm.ar_active then
    -- Wrap in pcall to catch errors
    local success, err = pcall(function()
      -- Initialize debug tracking on first frame
      if Brainstorm.debug.start_time == 0 then
        Brainstorm.debug.start_time = os_clock()
        Brainstorm.debug.last_report_time = os_clock()
      end

      -- Update performance metrics
      if Brainstorm.debug.enabled then
        Brainstorm.debug.last_report_time = os_clock()
      end

      Brainstorm.ar_frames = Brainstorm.ar_frames + 1
      Brainstorm.ar_timer = Brainstorm.ar_timer + dt

      if Brainstorm.ar_timer >= Brainstorm.AR_INTERVAL then
        Brainstorm.ar_timer = Brainstorm.ar_timer - Brainstorm.AR_INTERVAL

        -- Try multiple seeds based on spf_int setting (seeds per second)
        -- AR_INTERVAL is 0.01 seconds, so we run 100 times per second
        -- Divide spf_int by 100 to get seeds per interval
        local seeds_to_try =
          math_max(1, math_floor(Brainstorm.config.ar_prefs.spf_int / 100))
        local seed_found = nil

        -- Determine if we need to validate Erratic deck requirements
        -- Erratic decks have random card distributions that need checking
        local has_erratic_requirements = G.GAME.starting_params.erratic_suits_and_ranks
          and (
            Brainstorm.config.ar_prefs.face_count > 0
            or Brainstorm.config.ar_prefs.suit_ratio_decimal > 0
          )

        -- Check if we have other filters active (vouchers, tags, packs)
        local has_other_filters = (
          Brainstorm.config.ar_filters.voucher_name ~= ""
        )
          or (Brainstorm.config.ar_filters.tag_name ~= "")
          or (Brainstorm.config.ar_filters.tag2_name ~= "")
          or (#Brainstorm.config.ar_filters.pack > 0)

        if has_erratic_requirements then
          -- Performance optimization for Erratic deck searches
          -- Limit seeds per frame to prevent lag spikes
          local max_seeds = 5 -- Conservative default to maintain 60 FPS
          if Brainstorm.config.ar_prefs.spf_int <= 500 then
            max_seeds = seeds_to_try -- Use full speed for low settings
          elseif Brainstorm.config.ar_prefs.spf_int <= 1000 then
            max_seeds = math_min(seeds_to_try, 5) -- Cap at 5 for medium
          else
            max_seeds = math_min(seeds_to_try, 10) -- Cap at 10 for high
          end
          local erratic_seeds_to_try = max_seeds

          -- Track performance metrics
          if
            Brainstorm.debug.enabled and Brainstorm.debug.seeds_tested == 0
          then
            log:info("Starting Erratic deck search", {
              seeds_per_frame = erratic_seeds_to_try,
              target_speed = Brainstorm.config.ar_prefs.spf_int,
            })
          end

          for i = 1, erratic_seeds_to_try do
            local test_seed = nil

            if has_other_filters then
              -- Use DLL to find seeds with voucher/tag/pack requirements
              test_seed = Brainstorm.auto_reroll()
              if test_seed then
                Brainstorm.debug.seeds_found = Brainstorm.debug.seeds_found + 1
              end
            else
              -- No other filters, just generate random seeds
              test_seed = random_string(
                8,
                G.CONTROLLER.cursor_hover.T.x * SEED_X_FACTOR
                  + G.CONTROLLER.cursor_hover.T.y * SEED_Y_FACTOR
                  + SEED_TIME_FACTOR
                    * (G.CONTROLLER.cursor_hover.time + i * 0.001)
              )
            end

            if test_seed then
              local stake = G.GAME.stake
              local challenge = G.GAME
                and G.GAME.challenge
                and G.GAME.challenge_tab
              G:delete_run()
              G:start_run({
                stake = stake,
                seed = test_seed,
                challenge = challenge,
              })

              -- Check if this seed meets the Erratic deck requirements
              local deck_data = Brainstorm.analyze_deck()
              Brainstorm.debug.seeds_tested = Brainstorm.debug.seeds_tested + 1

              -- Also check dual tags if configured
              local tags_valid = Brainstorm.check_dual_tags()

              if
                tags_valid
                and Brainstorm.is_valid_deck(
                  deck_data,
                  Brainstorm.config.ar_prefs.face_count,
                  0,
                  Brainstorm.config.ar_prefs.suit_ratio_decimal
                )
              then
                -- Found a valid seed that meets all criteria
                if Brainstorm.debug.enabled then
                  log:info("Seed found!", {
                    seed = test_seed,
                    seeds_tested = Brainstorm.debug.seeds_tested,
                  })
                end
                Brainstorm.stop_auto_reroll(true)
                G.GAME.used_filter = true
                G.GAME.seeded = false
                break
              end
            end
          end
        else
          -- No Erratic requirements, use DLL normally
          -- DLL already searches many seeds internally (100K-1M), so only call once
          seed_found = Brainstorm.auto_reroll()
          if seed_found then
            -- Found a seed that matches DLL criteria
            local stake = G.GAME.stake
            local challenge = G.GAME
              and G.GAME.challenge
              and G.GAME.challenge_tab
            G:delete_run()
            G:start_run({
              stake = stake,
              seed = seed_found,
              challenge = challenge,
            })

            -- Check dual tags if configured
            local tags_valid = Brainstorm.check_dual_tags()

            if tags_valid then
              if Brainstorm.debug.enabled then
                log:info("Seed found!", {
                  seed = seed_found,
                  seeds_tested = Brainstorm.debug.seeds_tested + 1,
                })
              end
              G.GAME.used_filter = true
              G.GAME.seeded = false
              Brainstorm.debug.seeds_tested = Brainstorm.debug.seeds_tested + 1
              Brainstorm.debug.seeds_found = Brainstorm.debug.seeds_found + 1
              Brainstorm.stop_auto_reroll(true)
            else
              -- Tags don't match, continue searching
              Brainstorm.debug.seeds_tested = Brainstorm.debug.seeds_tested + 1
            end
          end
        end
      end
      if
        Brainstorm.ar_frames == 60
        and not Brainstorm.ar_text
        and Brainstorm.ar_active
      then
        Brainstorm.ar_text = Brainstorm.attention_text({
          scale = 1.4,
          text = "Rerolling...",
          align = "cm",
          offset = { x = 0, y = -3.5 },
          major = G.STAGE == G.STAGES.RUN and G.play or G.title_top,
        })
      end
    end) -- End of pcall

    if not success then
      print("[Brainstorm] ERROR in auto-reroll update: " .. tostring(err))
      Brainstorm.ar_active = false -- Disable auto-reroll on error
    end
  end
end

-- Foreign Function Interface (FFI) for native DLL integration
-- The DLL provides high-performance seed filtering without game restarts
local ffi_loaded = false
local immolate_dll = nil -- Cache the DLL handle to avoid repeated loading

-- Initialize FFI definitions for DLL functions
local function init_ffi()
  if not ffi_loaded then
    local success, err = pcall(
      ffi.cdef,
      [[
      // Enhanced brainstorm function with dual tag support
      const char* brainstorm(const char* seed, const char* voucher, const char* pack, const char* tag1, const char* tag2, double souls, bool observatory, bool perkeo);
      // Get both tags for a specific seed
      const char* get_tags(const char* seed);
      // Free memory allocated by DLL (prevents memory leaks)
      void free_result(const char* result);
      // GPU/CUDA functions
      int get_acceleration_type();  // 0=CPU, 1=GPU
      const char* get_hardware_info();
      // Dynamic pool updates
      void brainstorm_update_pools(const char* json_utf8);
      void set_use_cuda(bool enable);
    ]]
    )
    if not success then
      print("[Brainstorm] Failed to initialize FFI: " .. tostring(err))
      return false
    end
    ffi_loaded = true
  end
  return true
end

-- Stop auto-reroll and clean up resources
-- Parameters:
--   success: Whether a matching seed was found
function Brainstorm.stop_auto_reroll(success)
  Brainstorm.ar_active = false
  Brainstorm.ar_frames = 0

  -- Remove UI text if present
  if Brainstorm.ar_text then
    if Brainstorm.ar_text.AT then
      Brainstorm.remove_attention_text(Brainstorm.ar_text)
    end
    Brainstorm.ar_text = nil
  end

  -- Print final debug report
  if Brainstorm.debug.enabled and Brainstorm.debug.seeds_tested > 0 then
    Brainstorm.print_debug_report(success)
  end

  -- Reset debug stats
  Brainstorm.debug.seeds_tested = 0
  Brainstorm.debug.seeds_found = 0
  Brainstorm.debug.start_time = 0
  Brainstorm.debug.highest_suit_ratio = 0
  Brainstorm.debug.highest_face_count = 0
  Brainstorm.debug.rejection_reasons = {
    face_cards = 0,
    suit_ratio = 0,
    dll_filter = 0,
  }
  Brainstorm.debug.distributions = {
    face_cards = {},
    suit_ratios = {},
  }
end

-- Generate and test seeds using the native DLL
-- Returns a matching seed or nil if none found
function Brainstorm.auto_reroll()
  -- Generate a pseudo-random seed based on cursor position and time
  local seed_found = random_string(
    8,
    G.CONTROLLER.cursor_hover.T.x * SEED_X_FACTOR
      + G.CONTROLLER.cursor_hover.T.y * SEED_Y_FACTOR
      + SEED_TIME_FACTOR * G.CONTROLLER.cursor_hover.time
  )

  -- Load native DLL with error handling (cache DLL handle)
  if not init_ffi() then
    print("[Brainstorm] FFI initialization failed")
    return nil
  end

  -- Use cached DLL handle if available
  local immolate = immolate_dll
  if not immolate then
    local dll_path = Brainstorm.PATH .. "/Immolate.dll"

    -- Check if DLL exists first
    local dll_file = io.open(dll_path, "rb")
    if not dll_file then
      if Brainstorm.debug.enabled then
        log:error("DLL not found", { path = dll_path })
      end
      return nil
    end
    dll_file:close()

    local success
    success, immolate = pcall(ffi.load, dll_path)
    if not success then
      if Brainstorm.debug.enabled then
        log:error("Failed to load Immolate.dll", { error = tostring(immolate) })
      end
      return nil
    end
    immolate_dll = immolate -- Cache for future use
    if Brainstorm.debug.enabled then
      log:info("DLL loaded successfully")
    end
    
    -- Update GPU pools with current game state
    if G and G.P_CENTER_POOLS then
      local pools_updated = Brainstorm.update_pools()
      if Brainstorm.debug.enabled then
        log:info("GPU pools update", { success = pools_updated })
      end
    end

    -- Configure GPU/CUDA support based on config
    if immolate_dll.set_use_cuda then
      local use_cuda = Brainstorm.config.use_cuda ~= false -- Default to true
      pcall(immolate_dll.set_use_cuda, use_cuda)

      -- Get hardware info for initialization
      if immolate_dll.get_hardware_info then
        local info_ptr = immolate_dll.get_hardware_info()
        if info_ptr ~= nil then
          local hardware_info = ffi.string(info_ptr)
          if Brainstorm.debug.enabled then
            log:info("Hardware detected", { info = hardware_info })
          end

          -- Check actual acceleration type
          if immolate_dll.get_acceleration_type then
            local accel_type = immolate_dll.get_acceleration_type()
            if accel_type == 1 then
              if Brainstorm.debug.enabled then
                log:info("GPU acceleration enabled")
              end
              Brainstorm.debug.gpu_enabled = true
            else
              if Brainstorm.debug.enabled then
                log:info("Using CPU acceleration")
              end
              Brainstorm.debug.gpu_enabled = false
            end
          end
        end
      end
    end
  end
  -- Extract pack name from configuration and map to DLL format
  local pack_name = ""
  if #Brainstorm.config.ar_filters.pack > 0 then
    local pack_key = Brainstorm.config.ar_filters.pack[1]
    -- Extract base pack name (e.g., "p_arcana_normal" from "p_arcana_normal_1")
    local base_key = pack_key:match("^([^_]+_[^_]+_[^_]+)") or pack_key:match("^([^_]+_[^_]+)") or pack_key
    -- Map to DLL-expected name
    pack_name = Brainstorm.PACK_TO_DLL[base_key] or ""
  end
  
  -- Map internal keys to DLL-expected names
  local tag_name = ""
  if
    Brainstorm.config.ar_filters.tag_name
    and Brainstorm.config.ar_filters.tag_name ~= ""
  then
    -- Use our mapping instead of localize() to ensure DLL compatibility
    tag_name = Brainstorm.TAG_TO_DLL[Brainstorm.config.ar_filters.tag_name] or ""
  end
  
  local tag2_name = ""
  if
    Brainstorm.config.ar_filters.tag2_name
    and Brainstorm.config.ar_filters.tag2_name ~= ""
  then
    -- Use our mapping instead of localize() to ensure DLL compatibility
    tag2_name = Brainstorm.TAG_TO_DLL[Brainstorm.config.ar_filters.tag2_name] or ""
  end
  
  local voucher_name = ""
  if
    Brainstorm.config.ar_filters.voucher_name
    and Brainstorm.config.ar_filters.voucher_name ~= ""
  then
    -- Use our mapping instead of localize() to ensure DLL compatibility
    voucher_name = Brainstorm.VOUCHER_TO_DLL[Brainstorm.config.ar_filters.voucher_name] or ""
  end

  -- Smart compatibility layer: Detect DLL version and use appropriate call
  -- Enhanced DLL (8 params) supports dual tags internally for 10-100x speedup
  -- Original DLL (7 params) requires external validation (slower)
  local result = nil

  -- Track DLL usage for performance metrics
  if Brainstorm.debug.enabled then
    Brainstorm.debug.dll_calls = (Brainstorm.debug.dll_calls or 0) + 1
  end

  -- Debug logging to verify parameter values
  if Brainstorm.debug.enabled then
    log:info("DLL parameters:", {
      seed = seed_found,
      voucher = voucher_name,
      pack = pack_name,
      tag1 = tag_name,
      tag2 = tag2_name,
      souls = Brainstorm.config.ar_filters.soul_skip or 0,
      observatory = Brainstorm.config.ar_filters.inst_observatory or false,
      perkeo = Brainstorm.config.ar_filters.inst_perkeo or false
    })
  end
  
  -- Comprehensive debug logging for DLL call
  local timer = Brainstorm.debug_timer("DLL", "GPU search call")
  
  Brainstorm.debug_log("DLL", "=== CALLING DLL ===")
  Brainstorm.debug_log("DLL", "Parameters being sent:")
  Brainstorm.debug_log("DLL", "  seed = '%s'", seed_found or "nil")
  Brainstorm.debug_log("DLL", "  voucher = '%s'", voucher_name or "(empty)")
  Brainstorm.debug_log("DLL", "  pack = '%s'", pack_name or "(empty)")
  Brainstorm.debug_log("DLL", "  tag1 = '%s'", tag_name or "(empty)")
  Brainstorm.debug_log("DLL", "  tag2 = '%s'", tag2_name or "(empty)")
  Brainstorm.debug_log("DLL", "  souls = %s", tostring(Brainstorm.config.ar_filters.soul_skip or 0))
  Brainstorm.debug_log("DLL", "  observatory = %s", tostring(Brainstorm.config.ar_filters.inst_observatory or false))
  Brainstorm.debug_log("DLL", "  perkeo = %s", tostring(Brainstorm.config.ar_filters.inst_perkeo or false))
  
  -- Seeds are alphanumeric (A-Z and 0-9) - no validation needed
  
  -- Pipeline visibility: log flow steps
  Brainstorm.debug_log("FLOW", "=== PIPELINE START ===")
  Brainstorm.debug_log("FLOW", "Step 1: Updating pools before search")
  
  -- Update pools before searching (critical!)
  if immolate.brainstorm_update_pools then
    local pools_json = Brainstorm.get_pool_json()
    if pools_json then
      Brainstorm.debug_log("DLL", "Sending pools JSON to DLL: %s", pools_json:sub(1, 200))
      local ok, err = pcall(immolate.brainstorm_update_pools, pools_json)
      if not ok then
        Brainstorm.debug_log("DLL", "ERROR: Failed to update pools: %s", tostring(err))
      else
        Brainstorm.debug_log("DLL", "Pools updated successfully")
      end
    else
      Brainstorm.debug_log("DLL", "ERROR: Failed to generate pools JSON")
    end
  else
    Brainstorm.debug_log("DLL", "ERROR: brainstorm_update_pools not found in DLL")
  end
  
  Brainstorm.debug_log("FLOW", "Step 2: Calling brainstorm(seed=%s, voucher=%s, pack=%s, tag1=%s, tag2=%s)", 
                       tostring(seed_found), tostring(voucher_name), tostring(pack_name), 
                       tostring(tag_name), tostring(tag2_name))
  
  -- Always use 8-parameter version (our DLL expects this)
  -- Use pcall to catch errors but always use 8 params
  local call_success, call_result = pcall(function()
    return immolate.brainstorm(
      seed_found,
      voucher_name or "",
      pack_name or "",
      tag_name or "",
      tag2_name or "",
      Brainstorm.config.ar_filters.soul_skip or 0,
      Brainstorm.config.ar_filters.inst_observatory or false,
      Brainstorm.config.ar_filters.inst_perkeo or false
    )
  end)
  
  timer.stop()
  
  local result = nil
  if call_success then
    result = call_result
    Brainstorm.debug_log("DLL", "DLL call successful, result type: %s", type(result))
    if result then
      Brainstorm.debug_log("DLL", "Result pointer: %s", tostring(result))
    end
  else
    Brainstorm.debug_log("DLL", "ERROR: DLL call failed - %s", tostring(call_result))
    return nil
  end

  -- Check for null pointer explicitly
  seed_found = nil
  if result ~= nil and result ~= ffi.NULL then
    seed_found = ffi.string(result)
    Brainstorm.debug_log("DLL", "=== DLL RETURNED: %s ===", seed_found)
    
    -- Safety gate: only accept valid 8-char [0-9A-Z] seeds
    local is_valid = (#seed_found == 8) and (seed_found:match("^[0-9A-Z]+$") ~= nil)
    if is_valid then
      Brainstorm.debug_log("DLL", "Valid seed found: %s", seed_found)
    elseif seed_found == "RETRY" then
      Brainstorm.debug_log("DLL", "DLL returned RETRY - no match or error")
      seed_found = nil
    else
      Brainstorm.debug_log("DLL", "WARNING: Invalid seed format: '%s' (len=%d)", seed_found, #seed_found)
      seed_found = nil
    end
  else
    Brainstorm.debug_log("DLL", "No result from DLL (null pointer)")
  end

  -- Free memory if enhanced DLL (always try to free for both versions)
  if result ~= nil and result ~= ffi.NULL then
    if immolate.free_result then
      local free_success = pcall(immolate.free_result, result)
      if not free_success and Brainstorm.debug.enabled then
        log:warn("Failed to free DLL result memory")
      end
    end
  end
  
  Brainstorm.debug_log("FLOW", "=== PIPELINE END (returning %s) ===", tostring(seed_found))

  -- Return the seed without restarting the game
  -- The caller will decide whether to restart based on deck validation
  return seed_found
end

-- Hook to color-code filtered seeds in the UI
-- Filtered seeds appear in blue instead of red
local cursr = create_UIBox_round_scores_row
function create_UIBox_round_scores_row(score, text_colour)
  local ret = cursr(score, text_colour)
  -- Color logic: Red for manual seeds, Blue for filtered seeds, Black otherwise
  ret.nodes[2].nodes[1].config.colour = (score == "seed" and G.GAME.seeded)
      and G.C.RED
    or (score == "seed" and G.GAME.used_filter) and G.C.BLUE
    or G.C.BLACK
  return ret
end

-- TODO: Rework attention text.
function Brainstorm.attention_text(args)
  args = args or {}
  args.text = args.text or "test"
  args.scale = args.scale or 1
  args.colour = copy_table(args.colour or G.C.WHITE)
  args.hold = (args.hold or 0) + 0.1 * G.SPEEDFACTOR
  args.pos = args.pos or { x = 0, y = 0 }
  args.align = args.align or "cm"
  args.emboss = args.emboss or nil

  args.fade = 1

  if args.cover then
    args.cover_colour = copy_table(args.cover_colour or G.C.RED)
    args.cover_colour_l = copy_table(lighten(args.cover_colour, 0.2))
    args.cover_colour_d = copy_table(darken(args.cover_colour, 0.2))
  else
    args.cover_colour = copy_table(G.C.CLEAR)
  end

  args.uibox_config = {
    align = args.align or "cm",
    offset = args.offset or { x = 0, y = 0 },
    major = args.cover or args.major or nil,
  }

  G.E_MANAGER:add_event(Event({
    trigger = "after",
    delay = 0,
    blockable = false,
    blocking = false,
    func = function()
      args.AT = UIBox({
        T = { args.pos.x, args.pos.y, 0, 0 },
        definition = {
          n = G.UIT.ROOT,
          config = {
            align = args.cover_align or "cm",
            minw = (args.cover and args.cover.T.w or 0.001)
              + (args.cover_padding or 0),
            minh = (args.cover and args.cover.T.h or 0.001)
              + (args.cover_padding or 0),
            padding = 0.03,
            r = 0.1,
            emboss = args.emboss,
            colour = args.cover_colour,
          },
          nodes = {
            {
              n = G.UIT.O,
              config = {
                draw_layer = 1,
                object = DynaText({
                  scale = args.scale,
                  string = args.text,
                  maxw = args.maxw,
                  colours = { args.colour },
                  float = true,
                  shadow = true,
                  silent = not args.noisy,
                  pop_in = 0,
                  pop_in_rate = 6,
                  rotate = args.rotate or nil,
                }),
              },
            },
          },
        },
        config = args.uibox_config,
      })
      args.AT.attention_text = true

      args.text = args.AT.UIRoot.children[1].config.object
      args.text:pulse(0.5)

      if args.cover then
        Particles(args.pos.x, args.pos.y, 0, 0, {
          timer_type = "TOTAL",
          timer = 0.01,
          pulse_max = 15,
          max = 0,
          scale = 0.3,
          vel_variation = 0.2,
          padding = 0.1,
          fill = true,
          lifespan = 0.5,
          speed = 2.5,
          attach = args.AT.UIRoot,
          colours = {
            args.cover_colour,
            args.cover_colour_l,
            args.cover_colour_d,
          },
        })
      end
      if args.backdrop_colour then
        args.backdrop_colour = copy_table(args.backdrop_colour)
        Particles(args.pos.x, args.pos.y, 0, 0, {
          timer_type = "TOTAL",
          timer = 5,
          scale = 2.4 * (args.backdrop_scale or 1),
          lifespan = 5,
          speed = 0,
          attach = args.AT,
          colours = { args.backdrop_colour },
        })
      end
      return true
    end,
  }))
  return args
end

function Brainstorm.remove_attention_text(args)
  if not args or not args.AT then
    return -- Nothing to remove
  end

  G.E_MANAGER:add_event(Event({
    trigger = "after",
    delay = 0,
    blockable = false,
    blocking = false,
    func = function()
      if not args.start_time then
        args.start_time = G.TIMERS.TOTAL
        if args.text and args.text.pop_out then
          args.text:pop_out(2)
        end
      else
        args.fade = math.max(0, 1 - 3 * (G.TIMERS.TOTAL - args.start_time))
        if args.cover_colour then
          args.cover_colour[4] = math.min(args.cover_colour[4], 2 * args.fade)
        end
        if args.cover_colour_l then
          args.cover_colour_l[4] = math.min(args.cover_colour_l[4], args.fade)
        end
        if args.cover_colour_d then
          args.cover_colour_d[4] = math.min(args.cover_colour_d[4], args.fade)
        end
        if args.backdrop_colour then
          args.backdrop_colour[4] = math.min(args.backdrop_colour[4], args.fade)
        end
        if args.colour then
          args.colour[4] = math.min(args.colour[4], args.fade)
        end
        if args.fade <= 0 and args.AT then
          args.AT:remove()
          return true
        end
      end
    end,
  }))
end
