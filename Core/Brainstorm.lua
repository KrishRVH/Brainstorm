local lovely = require("lovely")
local nfs = require("nativefs")

Brainstorm = {}

-- Mod version
Brainstorm.VERSION = "Brainstorm v2.2.0-alpha"

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
  },
  ar_filters = {
    pack = {},
    pack_id = 1,
    voucher_name = "",
    voucher_id = 1,
    tag_name = "tag_charm",
    tag_id = 2,
    soul_skip = 1,
    inst_observatory = false,
    inst_perkeo = false,
  },
  ar_prefs = {
    spf_id = 3,
    spf_int = 1000,
    face_count = 0,
    suit_ratio_id = 1,
    suit_ratio_percent = "50%",
    suit_ratio_decimal = 0.5,
  },
}

-- Auto-reroll state
Brainstorm.ar_timer = 0
Brainstorm.ar_frames = 0
Brainstorm.ar_text = nil
Brainstorm.ar_active = false

-- Constants
Brainstorm.AR_INTERVAL = 0.01 -- Seconds between reroll attempts

-- Cache frequently used functions for performance
local string_format = string.format
local string_lower = string.lower

-- Random seed generation constants
local SEED_X_FACTOR = 0.33411983
local SEED_Y_FACTOR = 0.874146
local SEED_TIME_FACTOR = 0.412311010

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

function Brainstorm.load_config()
  local config_path = Brainstorm.PATH .. "/config.lua"
  if not file_exists(config_path) then
    Brainstorm.write_config()
  else
    local config_file, err = nfs.read(config_path)
    if not config_file then
      error("Failed to read config file: " .. (err or "unknown error"))
    end
    -- STR_UNPACK is a Balatro function for deserializing Lua tables
    local success, loaded_config = pcall(STR_UNPACK, config_file)
    if success and loaded_config then
      -- Deep merge loaded config with defaults to handle new fields
      local function deep_merge(target, source)
        for key, value in pairs(source) do
          if type(value) == "table" and type(target[key]) == "table" then
            deep_merge(target[key], value)
          else
            target[key] = value
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
        or "50%"
      Brainstorm.config.ar_prefs.suit_ratio_decimal = Brainstorm.config.ar_prefs.suit_ratio_decimal
        or 0.5
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
      error("Failed to write config file: " .. (err or "unknown error"))
    end
  end
end

function Brainstorm.init()
  Brainstorm.PATH = find_brainstorm_directory(lovely.mod_dir)
  Brainstorm.load_config()
  assert(load(nfs.read(Brainstorm.PATH .. "/UI/ui.lua")))()
end

local key_press_update_ref = Controller.key_press_update
function Controller:key_press_update(key, dt)
  key_press_update_ref(self, key, dt)
  local keybinds = Brainstorm.config.keybinds
  if love.keyboard.isDown(keybinds.modifier) then
    if key == keybinds.f_reroll then
      Brainstorm.reroll()
    elseif key == keybinds.a_reroll then
      Brainstorm.ar_active = not Brainstorm.ar_active
    end
  end
end

function Brainstorm.analyze_deck()
  local deck_summary = {}
  local suit_count = { Hearts = 0, Diamonds = 0, Clubs = 0, Spades = 0 }
  local face_card_count = 0
  local numeric_card_count = 0
  local ace_count = 0
  local unique_card_count = 0

  for _, card in ipairs(G.playing_cards) do
    if card.base then
      local card_name = card.base.value .. " of " .. card.base.suit
      deck_summary[card_name] = (deck_summary[card_name] or 0) + 1
      suit_count[card.base.suit] = (suit_count[card.base.suit] or 0) + 1

      -- Categorizing cards
      if card.base.value == "Ace" then
        ace_count = ace_count + 1
      elseif
        card.base.value == "Jack"
        or card.base.value == "Queen"
        or card.base.value == "King"
      then
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
  local suit_count = deck_data.suit_count or 0

  -- Check Face Cards & Aces
  if face_card_count < min_face_cards then
    --print("Not enough face cards:", face_card_count, "Required:", min_face_cards)
    return false
  end
  if ace_count < min_aces then
    --print("Not enough aces:", ace_count, "Required:", min_aces)
    return false
  end

  -- Check suit distribution
  local sorted_suits = {}
  for suit, count in pairs(suit_count) do
    table.insert(sorted_suits, { suit = suit, count = count })
  end
  table.sort(sorted_suits, function(a, b)
    return a.count > b.count
  end)

  -- Sum the top 2 suit counts
  local top_2_suit_count = sorted_suits[1].count
    + (sorted_suits[2] and sorted_suits[2].count or 0)
  local top_2_suit_percentage = top_2_suit_count / total_cards

  if top_2_suit_percentage < dominant_suit_ratio then
    --print("Suit distribution is too spread out.")
    return false
  end

  return true
end

function Brainstorm.reroll()
  local G = G -- Cache global G for performance
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

local update_ref = Game.update
function Game:update(dt)
  update_ref(self, dt)

  if Brainstorm.ar_active then
    Brainstorm.ar_frames = Brainstorm.ar_frames + 1
    Brainstorm.ar_timer = Brainstorm.ar_timer + dt

    if Brainstorm.ar_timer >= Brainstorm.AR_INTERVAL then
      Brainstorm.ar_timer = Brainstorm.ar_timer - Brainstorm.AR_INTERVAL

      -- Try multiple seeds based on spf_int setting (seeds per second)
      -- AR_INTERVAL is 0.01 seconds, so we run 100 times per second
      -- Divide spf_int by 100 to get seeds per interval
      local seeds_to_try =
        math.max(1, math.floor(Brainstorm.config.ar_prefs.spf_int / 100))
      local seed_found = nil

      for i = 1, seeds_to_try do
        seed_found = Brainstorm.auto_reroll()
        if seed_found then
          if G.GAME.starting_params.erratic_suits_and_ranks then
            local deck_data = Brainstorm.analyze_deck()
            if
              Brainstorm.is_valid_deck(
                deck_data,
                Brainstorm.config.ar_prefs.face_count,
                0,
                Brainstorm.config.ar_prefs.suit_ratio_decimal
              )
            then
              Brainstorm.stop_auto_reroll()
              break
            end
            -- Note: For Erratic decks, this may run indefinitely if no seed
            -- matches both the filter criteria AND deck requirements
          else
            Brainstorm.stop_auto_reroll()
            break
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
  end
end

local ffi = require("ffi")
local lovely = require("lovely")

-- FFI definition for native DLL
local ffi_loaded = false
local function init_ffi()
  if not ffi_loaded then
    local success, err = pcall(
      ffi.cdef,
      [[
      const char* brainstorm(const char* seed, const char* voucher, const char* pack, const char* tag, double souls, bool observatory, bool perkeo);
    ]]
    )
    if not success then
      error("Failed to initialize FFI: " .. tostring(err))
    end
    ffi_loaded = true
  end
end

function Brainstorm.stop_auto_reroll()
  Brainstorm.ar_active = false
  Brainstorm.ar_frames = 0
  if Brainstorm.ar_text then
    if Brainstorm.ar_text.AT then
      Brainstorm.remove_attention_text(Brainstorm.ar_text)
    end
    Brainstorm.ar_text = nil
  end
end

function Brainstorm.auto_reroll()
  local seed_found = random_string(
    8,
    G.CONTROLLER.cursor_hover.T.x * SEED_X_FACTOR
      + G.CONTROLLER.cursor_hover.T.y * SEED_Y_FACTOR
      + SEED_TIME_FACTOR * G.CONTROLLER.cursor_hover.time
  )

  -- Load native DLL with error handling
  init_ffi()
  local success, immolate = pcall(ffi.load, Brainstorm.PATH .. "/Immolate.dll")
  if not success then
    error("Failed to load Immolate.dll: " .. tostring(immolate))
    return nil
  end
  -- Extract pack name from configuration
  local pack = ""
  if #Brainstorm.config.ar_filters.pack > 0 then
    pack = Brainstorm.config.ar_filters.pack[1]:match("^(.*)_") or ""
  end
  local pack_name = pack ~= ""
      and localize({ type = "name_text", set = "Other", key = pack })
    or ""
  local tag_name = localize({
    type = "name_text",
    set = "Tag",
    key = Brainstorm.config.ar_filters.tag_name,
  })
  local voucher_name = localize({
    type = "name_text",
    set = "Voucher",
    key = Brainstorm.config.ar_filters.voucher_name,
  })
  -- Call native function with error handling
  local call_success, result = pcall(function()
    return immolate.brainstorm(
      seed_found,
      voucher_name,
      pack_name,
      tag_name,
      Brainstorm.config.ar_filters.soul_skip,
      Brainstorm.config.ar_filters.inst_observatory,
      Brainstorm.config.ar_filters.inst_perkeo
    )
  end)

  if not call_success then
    print("Error calling native brainstorm function: " .. tostring(result))
    return nil
  end

  seed_found = result and ffi.string(result) or nil
  if seed_found then
    local stake = G.GAME.stake
    G:delete_run()
    G:start_run({
      stake = stake,
      seed = seed_found,
      challenge = G.GAME and G.GAME.challenge and G.GAME.challenge_tab,
    })
    G.GAME.used_filter = true
    G.GAME.filter_info = {
      filter_params = {
        seed_found,
        voucher_name,
        pack_name,
        tag_name,
        Brainstorm.config.ar_filters.soul_skip,
        Brainstorm.config.ar_filters.inst_observatory,
        Brainstorm.config.ar_filters.inst_perkeo,
      },
    }
    G.GAME.seeded = false
  end
  return seed_found
end

local cursr = create_UIBox_round_scores_row
function create_UIBox_round_scores_row(score, text_colour)
  local ret = cursr(score, text_colour)
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
                  args.scale,
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
        --args.AT:align_to_attach()
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
