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
  debug_enabled = true,
}

-- Auto-reroll state
Brainstorm.ar_timer = 0
Brainstorm.ar_frames = 0
Brainstorm.ar_text = nil
Brainstorm.ar_active = false

-- Debug statistics
Brainstorm.debug = {
  enabled = false, -- Will be set from config
  seeds_tested = 0,
  seeds_found = 0,
  start_time = 0,
  rejection_reasons = {
    face_cards = 0,
    suit_ratio = 0,
    dll_filter = 0,
  },
  distributions = {
    face_cards = {},
    suit_ratios = {},
  },
  last_report_time = 0,
  highest_suit_ratio = 0,
  highest_face_count = 0,
}

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
      print(
        "[Brainstorm] Failed to read config file: " .. (err or "unknown error")
      )
      return
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
        or "Disabled"

      -- Map suit ratio percentage to decimal value
      local ratio_map = {
        ["Disabled"] = 0,
        ["50%"] = 0.5,
        ["60%"] = 0.6,
        ["70%"] = 0.7,
        ["75%"] = 0.75,
        ["80%"] = 0.80,
      }
      Brainstorm.config.ar_prefs.suit_ratio_decimal = ratio_map[Brainstorm.config.ar_prefs.suit_ratio_percent]
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
      print(
        "[Brainstorm] Failed to write config file: " .. (err or "unknown error")
      )
    end
  end
end

function Brainstorm.init()
  Brainstorm.PATH = find_brainstorm_directory(lovely.mod_dir)
  Brainstorm.load_config()
  Brainstorm.debug.enabled = Brainstorm.config.debug_enabled or false
  assert(load(nfs.read(Brainstorm.PATH .. "/UI/ui.lua")))()
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
    compress_and_save(save_path, G.ARGS.save_run)
    Brainstorm.save_state_alert("Saved state to slot [" .. slot .. "]")
    return true
  end
  return false
end

function Brainstorm.load_game_state(slot)
  local save_path = G.SETTINGS.profile .. "/" .. "save_state_" .. slot .. ".jkr"
  local saved_game = get_compressed(save_path)

  if saved_game then
    G:delete_run()
    G.SAVED_GAME = STR_UNPACK(saved_game)
    G:start_run({ savetext = G.SAVED_GAME })
    Brainstorm.save_state_alert("Loaded state from slot [" .. slot .. "]")
    return true
  else
    Brainstorm.save_state_alert("No save in slot [" .. slot .. "]")
    return false
  end
end

local key_press_update_ref = Controller.key_press_update
function Controller:key_press_update(key, dt)
  key_press_update_ref(self, key, dt)
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
      if Brainstorm.ar_active then
        Brainstorm.stop_auto_reroll(false)
      else
        Brainstorm.ar_active = true
        
        -- Print helpful message for dual tag searches
        if Brainstorm.config.ar_filters.tag2_name and Brainstorm.config.ar_filters.tag2_name ~= "" then
          local tag1_display = localize({type = "name_text", set = "Tag", key = Brainstorm.config.ar_filters.tag_name})
          local tag2_display = localize({type = "name_text", set = "Tag", key = Brainstorm.config.ar_filters.tag2_name})
          
          if Brainstorm.config.ar_filters.tag_name == Brainstorm.config.ar_filters.tag2_name then
            print(string.format("[Brainstorm] Searching for DOUBLE %s tags...", tag1_display))
            print("[Brainstorm] This is extremely rare! May take 5-30 seconds depending on the tag.")
          else
            print(string.format("[Brainstorm] Searching for %s + %s tags...", tag1_display, tag2_display))
            print("[Brainstorm] Dual tag combinations can take 5-20 seconds to find.")
          end
          print("[Brainstorm] Order doesn't matter - either tag can be in either blind position.")
        end
      end
    end
  end
end

function Brainstorm.check_dual_tags()
  -- Check if both specified tags are present in the starting blind options
  local tag1 = Brainstorm.config.ar_filters.tag_name
  local tag2 = Brainstorm.config.ar_filters.tag2_name

  if not tag2 or tag2 == "" then
    return true -- No second tag to check
  end

  -- Tags are stored in G.GAME.round_resets.blind_tags.Small and .Big
  if not G.GAME or not G.GAME.round_resets or not G.GAME.round_resets.blind_tags then
    return false
  end

  local small_blind_tag = G.GAME.round_resets.blind_tags.Small
  local big_blind_tag = G.GAME.round_resets.blind_tags.Big

  -- Only log every 10th check to reduce spam
  if Brainstorm.debug.enabled and (Brainstorm.debug.dual_tag_checks or 0) % 10 == 0 then
    print(string.format("[Brainstorm] Checking tags - Small: %s, Big: %s (looking for %s + %s)", 
      tostring(small_blind_tag), tostring(big_blind_tag), tag1, tag2))
  end

  -- Track dual tag checks
  if Brainstorm.debug.enabled then
    Brainstorm.debug.dual_tag_checks = (Brainstorm.debug.dual_tag_checks or 0) + 1
  end

  -- Special case: if looking for the same tag twice, need BOTH positions to have it
  if tag1 == tag2 then
    local both_match = (small_blind_tag == tag1 and big_blind_tag == tag1)
    if both_match then
      print("[Brainstorm] SUCCESS! Both blinds have " .. tag1)
      if Brainstorm.debug.enabled then
        Brainstorm.debug.dual_tag_successes = (Brainstorm.debug.dual_tag_successes or 0) + 1
      end
      return true
    else
      if Brainstorm.debug.enabled and Brainstorm.debug.dual_tag_checks % 100 == 0 then
        print(string.format("[Brainstorm] Progress: %d seeds checked for dual %s tags", 
          Brainstorm.debug.dual_tag_checks, tag1))
      end
      return false
    end
  else
    -- Different tags: check if both are present (order doesn't matter)
    local has_tag1 = (small_blind_tag == tag1 or big_blind_tag == tag1)
    local has_tag2 = (small_blind_tag == tag2 or big_blind_tag == tag2)

    if has_tag1 and has_tag2 then
      print("[Brainstorm] SUCCESS! Both tags found (order-agnostic)")
      if Brainstorm.debug.enabled then
        Brainstorm.debug.dual_tag_successes = (Brainstorm.debug.dual_tag_successes or 0) + 1
      end
      return true
    else
      if Brainstorm.debug.enabled and Brainstorm.debug.dual_tag_checks % 100 == 0 then
        print(string.format("[Brainstorm] Progress: %d seeds checked for %s + %s", 
          Brainstorm.debug.dual_tag_checks, tag1, tag2))
      end
      return false
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
  local suit_count = deck_data.suit_count or {}

  -- Track distribution for debugging
  if Brainstorm.debug.enabled then
    local fc_bucket = math.floor(face_card_count / 5) * 5
    Brainstorm.debug.distributions.face_cards[fc_bucket] = (
      Brainstorm.debug.distributions.face_cards[fc_bucket] or 0
    ) + 1

    -- Track highest face count
    if face_card_count > Brainstorm.debug.highest_face_count then
      Brainstorm.debug.highest_face_count = face_card_count
      if face_card_count >= 20 then
        print(
          string.format(
            "[Brainstorm] HIGH FACE COUNT: %d face cards found!",
            face_card_count
          )
        )
      end
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
      table.insert(sorted_suits, { suit = suit, count = count })
    end

    if #sorted_suits > 0 then
      table.sort(sorted_suits, function(a, b)
        return a.count > b.count
      end)

      -- Sum the top 2 suit counts
      local top_2_suit_count = sorted_suits[1].count
        + (sorted_suits[2] and sorted_suits[2].count or 0)
      local top_2_suit_percentage = top_2_suit_count / total_cards

      -- Track suit ratio distribution
      if Brainstorm.debug.enabled then
        local ratio_bucket = math.floor(top_2_suit_percentage * 10) * 10
        Brainstorm.debug.distributions.suit_ratios[ratio_bucket] = (
          Brainstorm.debug.distributions.suit_ratios[ratio_bucket] or 0
        ) + 1

        -- Track the highest ratio we've seen
        if
          not Brainstorm.debug.highest_suit_ratio
          or top_2_suit_percentage > Brainstorm.debug.highest_suit_ratio
        then
          Brainstorm.debug.highest_suit_ratio = top_2_suit_percentage
          if top_2_suit_percentage >= 0.75 then
            -- Log exceptional finds
            print(
              string.format(
                "[Brainstorm] HIGH RATIO FOUND: %.1f%% (Spades:%d Hearts:%d Clubs:%d Diamonds:%d)",
                top_2_suit_percentage * 100,
                suit_count.Spades or 0,
                suit_count.Hearts or 0,
                suit_count.Clubs or 0,
                suit_count.Diamonds or 0
              )
            )
          end
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
  local elapsed = os.clock() - Brainstorm.debug.start_time
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

function Brainstorm.print_periodic_update()
  local elapsed = os.clock() - Brainstorm.debug.start_time
  local seeds_per_sec = Brainstorm.debug.seeds_tested / elapsed

  print(
    string.format(
      "[Brainstorm] Progress: %d seeds tested | %.1f seeds/sec | %d face rejections | %d ratio rejections",
      Brainstorm.debug.seeds_tested,
      seeds_per_sec,
      Brainstorm.debug.rejection_reasons.face_cards,
      Brainstorm.debug.rejection_reasons.suit_ratio
    )
  )
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
    -- Initialize debug tracking on first frame
    if Brainstorm.debug.start_time == 0 then
      Brainstorm.debug.start_time = os.clock()
      Brainstorm.debug.last_report_time = os.clock()
    end

    -- Print periodic updates every 5 seconds
    if
      Brainstorm.debug.enabled
      and os.clock() - Brainstorm.debug.last_report_time > 5
    then
      Brainstorm.print_periodic_update()
      Brainstorm.debug.last_report_time = os.clock()
    end

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

      -- Check if we have Erratic deck requirements
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
        -- For Erratic, dynamically adjust based on performance
        local max_seeds = 5 -- Conservative default
        if Brainstorm.config.ar_prefs.spf_int <= 500 then
          max_seeds = seeds_to_try -- Use full speed for low settings
        elseif Brainstorm.config.ar_prefs.spf_int <= 1000 then
          max_seeds = math.min(seeds_to_try, 5) -- Cap at 5 for medium
        else
          max_seeds = math.min(seeds_to_try, 10) -- Cap at 10 for high
        end
        local erratic_seeds_to_try = max_seeds

        -- Log actual speed for debugging
        if Brainstorm.debug.enabled and Brainstorm.debug.seeds_tested == 1 then
          print(
            string.format(
              "[Brainstorm] Testing %d seeds per frame (target: %d/sec)",
              erratic_seeds_to_try,
              Brainstorm.config.ar_prefs.spf_int
            )
          )
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
              Brainstorm.stop_auto_reroll(true)
              G.GAME.used_filter = true
              G.GAME.seeded = false
              break
            end
          end
        end
      else
        -- No Erratic requirements, use DLL normally
        for i = 1, seeds_to_try do
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
              G.GAME.used_filter = true
              G.GAME.seeded = false
              Brainstorm.debug.seeds_tested = Brainstorm.debug.seeds_tested + 1
              Brainstorm.debug.seeds_found = Brainstorm.debug.seeds_found + 1
              Brainstorm.stop_auto_reroll(true)
              break
            else
              -- Tags don't match, continue searching
              Brainstorm.debug.seeds_tested = Brainstorm.debug.seeds_tested + 1
            end
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
      const char* brainstorm(const char* seed, const char* voucher, const char* pack, const char* tag1, const char* tag2, double souls, bool observatory, bool perkeo);
      const char* get_tags(const char* seed);
      void free_result(const char* result);
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

function Brainstorm.stop_auto_reroll(success)
  Brainstorm.ar_active = false
  Brainstorm.ar_frames = 0
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

function Brainstorm.auto_reroll()
  local seed_found = random_string(
    8,
    G.CONTROLLER.cursor_hover.T.x * SEED_X_FACTOR
      + G.CONTROLLER.cursor_hover.T.y * SEED_Y_FACTOR
      + SEED_TIME_FACTOR * G.CONTROLLER.cursor_hover.time
  )

  -- Load native DLL with error handling
  if not init_ffi() then
    print("[Brainstorm] FFI initialization failed")
    return nil
  end
  local success, immolate = pcall(ffi.load, Brainstorm.PATH .. "/Immolate.dll")
  if not success then
    print("[Brainstorm] Failed to load Immolate.dll: " .. tostring(immolate))
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
  local tag2_name = ""
  if
    Brainstorm.config.ar_filters.tag2_name
    and Brainstorm.config.ar_filters.tag2_name ~= ""
  then
    tag2_name = localize({
      type = "name_text",
      set = "Tag",
      key = Brainstorm.config.ar_filters.tag2_name,
    })
  end
  local voucher_name = localize({
    type = "name_text",
    set = "Voucher",
    key = Brainstorm.config.ar_filters.voucher_name,
  })

  -- For dual tag search, we need a different approach
  -- The DLL can only check if a tag exists, not which blind position it's in
  -- For dual tags, we check if at least ONE tag exists, then validate both after restart
  local result = nil
  if tag2_name ~= "" then
    -- For same tag twice, only check once with DLL
    if tag_name == tag2_name then
      local call_success, call_result = pcall(function()
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
      if call_success and call_result then
        result = call_result
        -- Note: We still need to validate BOTH positions have the tag after restart
      end
    else
      -- Different tags: Check if first tag exists
      local call_success1, result1 = pcall(function()
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
      
      -- Only accept seeds that have the first tag
      -- We'll check for the second tag after restart
      if call_success1 and result1 then
        result = result1
        -- Note: This seed has tag1, we'll validate tag2 exists after restart
      end
    end
  else
    -- Single tag check (original logic)
    local call_success, call_result = pcall(function()
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
      print(
        "Error calling native brainstorm function: " .. tostring(call_result)
      )
      return nil
    end
    result = call_result
  end

  seed_found = result and ffi.string(result) or nil
  -- Return the seed without restarting the game
  -- The caller will decide whether to restart based on deck validation
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
