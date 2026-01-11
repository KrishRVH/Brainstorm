-- Brainstorm Mod for Balatro
-- High-performance seed filtering and save state management
-- Created by OceanRamen. Immolate DLL by MathIsFun0.
-- License: MIT
-- Entry point loaded by lovely.toml; initializes config, UI, and game hooks.

local lovely = require("lovely")
local nfs = require("nativefs")
local ffi = require("ffi")

Brainstorm = {}

-- Mod version
Brainstorm.VERSION = "Brainstorm v3.0.0"

-- Reserved for Steammodded compatibility
Brainstorm.SMODS = nil

Brainstorm.config = {
  enable = true,
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
    joker_name = "",
    joker_search = "",
    joker_id = 1,
    joker_location = "any",
    joker_location_id = 1,
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
}

-- Auto-reroll state management
-- Tracks the state of automatic seed rerolling
Brainstorm.ar_timer = 0 -- Time accumulator for reroll intervals
Brainstorm.ar_frames = 0 -- Frame counter for UI display timing
Brainstorm.ar_text = nil -- UI text element for "Rerolling..." message
Brainstorm.ar_active = false -- Whether auto-reroll is currently active
Brainstorm.ar_last_error = nil -- Last auto-reroll error shown to the user
Brainstorm.ar_seeds_scanned = 0 -- Approximate total seeds scanned this session
Brainstorm.ar_status_text = "Rerolling..." -- Live status text for auto-reroll

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

-- Constants
Brainstorm.AR_INTERVAL = 0.01 -- Seconds between reroll attempts (100 Hz)

-- Performance optimization: Cache frequently used functions
-- This avoids table lookups in hot code paths
local string_lower = string.lower
local pcall = pcall
local get_time = love.timer.getTime

-- Random seed generation constants
-- Seed generation factors using prime-based constants for good distribution
-- These values minimize correlation between cursor position and generated seeds:
--   X factor (~1/3): Spreads horizontal mouse movement across seed space
--   Y factor (~7/8): Ensures vertical movement contributes strongly
--   Time factor (~2/5): Incorporates temporal variation for entropy
local SEED_X_FACTOR = 0.33411983 -- Prime-derived for X axis distribution
local SEED_Y_FACTOR = 0.874146 -- Prime-derived for Y axis spread
local SEED_TIME_FACTOR = 0.412311010 -- Prime-derived for time entropy

local function build_seed_start()
  local seed_input = get_time() * SEED_TIME_FACTOR
  local hover = G.CONTROLLER and G.CONTROLLER.cursor_hover
  if hover and hover.T then
    seed_input = seed_input
      + hover.T.x * SEED_X_FACTOR
      + hover.T.y * SEED_Y_FACTOR
      + SEED_TIME_FACTOR * hover.time
  end
  if type(random_string) ~= "function" then
    return nil
  end
  return random_string(8, seed_input)
end

local function as_string(value)
  if type(value) == "string" then
    return value
  end
  if value == nil then
    return ""
  end
  return tostring(value)
end

local function as_number(value, default)
  local num = tonumber(value)
  if num == nil then
    return default
  end
  return num
end

local function as_int(value, default)
  local num = tonumber(value)
  if num == nil then
    return default
  end
  if num < 0 then
    return 0
  end
  return math.floor(num)
end

local function as_bool(value)
  return value and true or false
end

local function format_count(value)
  if type(number_format) == "function" then
    return number_format(value or 0)
  end
  return tostring(value or 0)
end

local function update_auto_reroll_status()
  local scanned = Brainstorm.ar_seeds_scanned or 0
  Brainstorm.ar_status_text = "Rerolling... scanned "
    .. format_count(scanned)
    .. " seeds"
end

local function init_log_path()
  if Brainstorm.LOG_PATH then
    return
  end
  if Brainstorm.PATH then
    Brainstorm.LOG_PATH = Brainstorm.PATH .. "/brainstorm.log"
  elseif lovely and lovely.mod_dir then
    Brainstorm.LOG_PATH = lovely.mod_dir .. "/Brainstorm/brainstorm.log"
  else
    Brainstorm.LOG_PATH = "brainstorm.log"
  end
  if not nfs.getInfo(Brainstorm.LOG_PATH) then
    pcall(nfs.write, Brainstorm.LOG_PATH, "")
  end
end

local function log_lua(message)
  -- Logging disabled.
  -- if not Brainstorm.LOG_PATH then
  --   init_log_path()
  -- end
  -- if not Brainstorm.LOG_PATH then
  --   return
  -- end
  -- local timestamp = os.date("%Y-%m-%d %H:%M:%S")
  -- local line = timestamp .. " [Lua] " .. message .. "\n"
  -- pcall(nfs.append, Brainstorm.LOG_PATH, line)
end

Brainstorm.log_lua = log_lua

local function show_auto_reroll_text()
  if Brainstorm.ar_text then
    return
  end
  if not Brainstorm.ar_status_text or Brainstorm.ar_status_text == "" then
    update_auto_reroll_status()
  end
  -- log_lua("auto_reroll UI text shown")
  Brainstorm.ar_text = Brainstorm.attention_text({
    scale = 1.4,
    text = { { ref_table = Brainstorm, ref_value = "ar_status_text" } },
    align = "cm",
    offset = { x = 0, y = -3.5 },
    major = G.STAGE == G.STAGES.RUN and G.play or G.title_top,
  })
end

local function report_auto_reroll_error(message)
  if not message or message == Brainstorm.ar_last_error then
    return
  end
  Brainstorm.ar_last_error = message
  -- log_lua("auto_reroll error: " .. message)
  Brainstorm.save_state_alert(message)
end

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
    local config_file = nfs.read(config_path)
    if not config_file then
      return
    end
    -- STR_UNPACK is a Balatro function for deserializing Lua tables
    local success, loaded_config = pcall(STR_UNPACK, config_file)
    if success and loaded_config then
      -- Deep merge loaded config with defaults to handle new fields
      -- This ensures backward compatibility when new config options are added
      local function deep_merge(target, source, allow_new_keys)
        for key, value in pairs(source) do
          if allow_new_keys or target[key] ~= nil then
            if type(value) == "table" and type(target[key]) == "table" then
              deep_merge(target[key], value, true) -- Allow nested keys for tables/arrays
            else
              target[key] = value -- Overwrite known value
            end
          end
        end
      end

      deep_merge(Brainstorm.config, loaded_config, false)

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
    local write_success = nfs.write(config_path, packed)
    if not write_success then
      return
    end
  end
end

function Brainstorm.init()
  Brainstorm.PATH = find_brainstorm_directory(lovely.mod_dir)
  if not Brainstorm.PATH then
    return false
  end
  -- init_log_path()
  -- log_lua("init start")
  -- log_lua("mod path=" .. Brainstorm.PATH .. " log path=" .. Brainstorm.LOG_PATH)

  Brainstorm.load_config()
  -- log_lua("config loaded")

  -- Load UI with error handling
  local ui_path = Brainstorm.PATH .. "/UI.lua"
  local ui_content = nfs.read(ui_path)
  if not ui_content then
    -- log_lua("init failed: UI.lua missing")
    return false
  end

  local ui_func = load(ui_content)
  if not ui_func then
    -- log_lua("init failed: UI.lua load error")
    return false
  end

  local success, ui_err = pcall(ui_func)
  if not success then
    -- log_lua("init failed: UI.lua error " .. tostring(ui_err))
    return false
  end
  -- log_lua("UI loaded")

  return true
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
    local success = pcall(compress_and_save, save_path, G.ARGS.save_run)
    if success then
      Brainstorm.save_state_alert("Saved state to slot [" .. slot .. "]")
      return true
    else
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
      -- log_lua("keybind: manual reroll")
      Brainstorm.reroll()
    elseif key == keybinds.a_reroll then
      local success = pcall(function()
        if Brainstorm.ar_active then
          -- log_lua("auto_reroll toggle: stopping")
          Brainstorm.stop_auto_reroll()
        else
          -- log_lua("auto_reroll toggle: starting")
          Brainstorm.ar_active = true
          Brainstorm.ar_last_error = nil
          Brainstorm.ar_seeds_scanned = 0
          update_auto_reroll_status()
          show_auto_reroll_text()
        end
      end)
      if not success then
        -- log_lua("auto_reroll toggle failed")
        Brainstorm.stop_auto_reroll()
      end
    end
  end
end

-- Perform a single manual reroll
-- Preserves stake and challenge settings
function Brainstorm.reroll()
  -- log_lua("manual reroll triggered")
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
    local success = pcall(function()
      Brainstorm.ar_frames = Brainstorm.ar_frames + 1
      Brainstorm.ar_timer = Brainstorm.ar_timer + dt

      if Brainstorm.ar_timer >= Brainstorm.AR_INTERVAL then
        Brainstorm.ar_timer = Brainstorm.ar_timer - Brainstorm.AR_INTERVAL
        local seed_found, err = Brainstorm.auto_reroll()
        if seed_found == false then
          Brainstorm.stop_auto_reroll()
          report_auto_reroll_error(err)
        elseif seed_found then
          local stake = G.GAME.stake
          local challenge = G.GAME and G.GAME.challenge and G.GAME.challenge_tab
          G:delete_run()
          G:start_run({
            stake = stake,
            seed = seed_found,
            challenge = challenge,
          })
          G.GAME.used_filter = true
          G.GAME.seeded = false
          Brainstorm.stop_auto_reroll()
        end
      end
      if Brainstorm.ar_active then
        show_auto_reroll_text()
      end
    end)
    if not success then
      -- log_lua("auto_reroll update failed")
      Brainstorm.stop_auto_reroll()
    end
  end
end

-- Foreign Function Interface (FFI) for native DLL integration
-- The DLL provides high-performance seed filtering without game restarts
local ffi_loaded = false
local native_handle = nil
local DLL_NAME = "Immolate.dll"

-- Initialize FFI definitions for DLL functions
local function init_ffi()
  if not ffi_loaded then
    local success = pcall(
      ffi.cdef,
      [[
      // Brainstorm seed search
      const char* brainstorm_search(const char* seed_start, const char* voucher_key, const char* pack_key, const char* tag1_key, const char* tag2_key, const char* joker_name, const char* joker_location, double souls, bool observatory, bool perkeo, const char* deck_key, bool erratic, bool no_faces, int min_face_cards, double suit_ratio, long long num_seeds, int threads);
      // Set native log path (optional)
      void immolate_set_log_path(const char* path);
      // Free memory allocated by DLL (prevents memory leaks)
      void free_result(const char* result);
    ]]
    )
    if not success then
      -- log_lua("ffi cdef failed")
      return false
    end
    ffi_loaded = true
    -- log_lua("ffi cdef loaded")
  end
  return true
end

local function load_native()
  if native_handle then
    return native_handle
  end

  local dll_path = Brainstorm.PATH .. "/" .. DLL_NAME
  local dll_file = io.open(dll_path, "rb")
  if not dll_file then
    -- log_lua("native load failed: missing " .. dll_path)
    return nil
  end
  dll_file:close()

  local success, handle = pcall(ffi.load, dll_path)
  if not success then
    -- log_lua("native load failed: " .. tostring(handle))
    return nil
  end

  native_handle = handle
  -- log_lua("native loaded: " .. dll_path)
  -- if native_handle.immolate_set_log_path and Brainstorm.LOG_PATH then
  --   pcall(native_handle.immolate_set_log_path, Brainstorm.LOG_PATH)
  -- end
  return handle
end

-- Stop auto-reroll and clean up resources
function Brainstorm.stop_auto_reroll()
  -- log_lua("auto_reroll stop")
  Brainstorm.ar_active = false
  Brainstorm.ar_frames = 0

  -- Remove UI text if present
  if Brainstorm.ar_text then
    if Brainstorm.ar_text.AT then
      Brainstorm.remove_attention_text(Brainstorm.ar_text)
    end
    Brainstorm.ar_text = nil
  end
end

-- Generate and test seeds using the native DLL
-- Returns a matching seed or nil if none found
function Brainstorm.auto_reroll()
  -- log_lua("auto_reroll begin")
  local seed_start = build_seed_start()
  if not seed_start then
    -- log_lua("auto_reroll failed: seed generator unavailable")
    return false, "Auto-reroll stopped (seed generator unavailable)"
  end
  -- log_lua("seed_start=" .. seed_start)

  if not init_ffi() then
    -- log_lua("auto_reroll failed: ffi init")
    return false, "Auto-reroll stopped (FFI init failed)"
  end

  local immolate = load_native()
  if not immolate then
    -- log_lua("auto_reroll failed: native load")
    return false, "Auto-reroll stopped (Immolate.dll missing)"
  end

  local pack_key = ""
  local pack_filter = Brainstorm.config.ar_filters.pack
  if type(pack_filter) == "table" and #pack_filter > 0 then
    pack_key = pack_filter[1]
  elseif type(pack_filter) == "string" then
    pack_key = pack_filter
  end

  local deck_key = ""
  if G.GAME then
    local back_key = G.GAME.selected_back_key
    if type(back_key) == "string" then
      deck_key = back_key
    elseif type(back_key) == "table" and type(back_key.key) == "string" then
      deck_key = back_key.key
    elseif
      G.GAME.selected_back
      and G.GAME.selected_back.effect
      and G.GAME.selected_back.effect.center
      and type(G.GAME.selected_back.effect.center.key) == "string"
    then
      deck_key = G.GAME.selected_back.effect.center.key
    end
  end

  local erratic = G.GAME
      and G.GAME.starting_params
      and G.GAME.starting_params.erratic_suits_and_ranks
    or false
  local no_faces = G.GAME
      and G.GAME.starting_params
      and G.GAME.starting_params.no_faces
    or false

  local min_face_cards = as_int(Brainstorm.config.ar_prefs.face_count, 0)
  local suit_ratio = as_number(Brainstorm.config.ar_prefs.suit_ratio_decimal, 0)
  local seed_budget = as_int(Brainstorm.config.ar_prefs.spf_int, 0)

  -- log_lua(
  --   "filters voucher="
  --     .. as_string(Brainstorm.config.ar_filters.voucher_name)
  --     .. " pack="
  --     .. as_string(pack_key)
  --     .. " tag1="
  --     .. as_string(Brainstorm.config.ar_filters.tag_name)
  --     .. " tag2="
  --     .. as_string(Brainstorm.config.ar_filters.tag2_name)
  --     .. " joker="
  --     .. as_string(Brainstorm.config.ar_filters.joker_name)
  --     .. " joker_location="
  --     .. as_string(Brainstorm.config.ar_filters.joker_location)
  --     .. " souls="
  --     .. as_string(Brainstorm.config.ar_filters.soul_skip)
  --     .. " observatory="
  --     .. tostring(Brainstorm.config.ar_filters.inst_observatory)
  --     .. " perkeo="
  --     .. tostring(Brainstorm.config.ar_filters.inst_perkeo)
  --     .. " deck="
  --     .. as_string(deck_key)
  --     .. " erratic="
  --     .. tostring(erratic)
  --     .. " no_faces="
  --     .. tostring(no_faces)
  --     .. " min_face_cards="
  --     .. tostring(min_face_cards)
  --     .. " suit_ratio="
  --     .. tostring(suit_ratio)
  --     .. " seed_budget="
  --     .. tostring(seed_budget)
  -- )

  local search_fn = immolate.brainstorm_search
  if not search_fn then
    -- log_lua("auto_reroll failed: missing brainstorm_search")
    return false, "Auto-reroll stopped (brainstorm_search missing in DLL)"
  end

  -- log_lua("calling brainstorm_search")
  local call_success, result = pcall(
    search_fn,
    seed_start,
    as_string(Brainstorm.config.ar_filters.voucher_name),
    as_string(pack_key),
    as_string(Brainstorm.config.ar_filters.tag_name),
    as_string(Brainstorm.config.ar_filters.tag2_name),
    as_string(Brainstorm.config.ar_filters.joker_name),
    as_string(Brainstorm.config.ar_filters.joker_location),
    as_number(Brainstorm.config.ar_filters.soul_skip, 0),
    as_bool(Brainstorm.config.ar_filters.inst_observatory),
    as_bool(Brainstorm.config.ar_filters.inst_perkeo),
    as_string(deck_key),
    as_bool(erratic),
    as_bool(no_faces),
    min_face_cards,
    suit_ratio,
    seed_budget,
    0
  )
  if not call_success then
    -- log_lua("brainstorm_search failed: " .. tostring(result))
    return false,
      "Auto-reroll stopped (native call failed: " .. tostring(result) .. ")"
  end

  if result == nil or result == ffi.NULL then
    Brainstorm.ar_seeds_scanned = (Brainstorm.ar_seeds_scanned or 0)
      + math.max(0, seed_budget)
    update_auto_reroll_status()
    -- log_lua("brainstorm_search result=null")
    return nil
  end

  local seed_found = ffi.string(result)
  -- log_lua("brainstorm_search result=" .. tostring(seed_found))

  if immolate.free_result then
    pcall(immolate.free_result, result)
    -- log_lua("brainstorm_search result freed")
  end

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
