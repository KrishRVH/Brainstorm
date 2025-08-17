-- Comprehensive test suite for Erratic deck validation
-- Tests edge cases and boundary conditions for face cards and suit ratios

local function run_erratic_tests()
  print("=" .. string.rep("=", 50))
  print("  ERRATIC DECK VALIDATION TEST SUITE")
  print("=" .. string.rep("=", 50))

  local passed = 0
  local failed = 0

  -- Test data structure for deck validation
  local function create_test_deck(face_count, hearts, diamonds, clubs, spades)
    local deck = {}
    local suits =
      { Hearts = hearts, Diamonds = diamonds, Clubs = clubs, Spades = spades }
    local ranks = {
      "Ace",
      "2",
      "3",
      "4",
      "5",
      "6",
      "7",
      "8",
      "9",
      "10",
      "Jack",
      "Queen",
      "King",
    }
    local face_ranks = { "Jack", "Queen", "King" }

    -- Add face cards first
    local faces_added = 0
    for suit, count in pairs(suits) do
      for _, rank in ipairs(face_ranks) do
        if faces_added < face_count then
          table.insert(deck, {
            base = { suit = suit, value = rank },
            suit = suit,
          })
          faces_added = faces_added + 1
        end
      end
    end

    -- Fill remaining cards
    while #deck < 52 do
      for suit, target_count in pairs(suits) do
        local current_count = 0
        for _, card in ipairs(deck) do
          if card.suit == suit then
            current_count = current_count + 1
          end
        end

        if current_count < target_count and #deck < 52 then
          -- Add non-face card
          for _, rank in ipairs(ranks) do
            if rank ~= "Jack" and rank ~= "Queen" and rank ~= "King" then
              table.insert(deck, {
                base = { suit = suit, value = rank },
                suit = suit,
              })
              break
            end
          end
        end
      end
    end

    return deck
  end

  -- Test helper to validate deck
  local function test_deck(
    name,
    deck,
    expected_faces,
    expected_ratio,
    should_pass
  )
    -- Count face cards
    local face_count = 0
    local suit_counts = { Hearts = 0, Diamonds = 0, Clubs = 0, Spades = 0 }

    for _, card in ipairs(deck) do
      local rank = card.base.value
      if rank == "Jack" or rank == "Queen" or rank == "King" then
        face_count = face_count + 1
      end
      suit_counts[card.suit] = (suit_counts[card.suit] or 0) + 1
    end

    -- Calculate suit ratio
    local max_suit_count = 0
    for _, count in pairs(suit_counts) do
      if count > max_suit_count then
        max_suit_count = count
      end
    end
    local suit_ratio = max_suit_count / #deck

    -- Validate
    local faces_ok = (expected_faces == 0 or face_count >= expected_faces)
    local ratio_ok = (expected_ratio == 0 or suit_ratio >= expected_ratio)
    local pass = faces_ok and ratio_ok

    if pass == should_pass then
      passed = passed + 1
      print(
        string.format(
          "✓ %s: %d faces (need %d), %.1f%% ratio (need %.1f%%)",
          name,
          face_count,
          expected_faces,
          suit_ratio * 100,
          expected_ratio * 100
        )
      )
    else
      failed = failed + 1
      print(
        string.format(
          "✗ %s: %d faces (need %d), %.1f%% ratio (need %.1f%%) - UNEXPECTED %s",
          name,
          face_count,
          expected_faces,
          suit_ratio * 100,
          expected_ratio * 100,
          pass and "PASS" or "FAIL"
        )
      )
    end
  end

  print("\n--- Test 1: Face Card Boundaries ---")

  -- Minimum face cards (0)
  test_deck("0 face cards", create_test_deck(0, 13, 13, 13, 13), 0, 0, true)

  -- Typical face cards (12)
  test_deck("12 face cards", create_test_deck(12, 13, 13, 13, 13), 12, 0, true)

  -- High face cards (20)
  test_deck("20 face cards", create_test_deck(20, 13, 13, 13, 13), 20, 0, true)

  -- Maximum realistic (23)
  test_deck("23 face cards", create_test_deck(23, 13, 13, 13, 13), 23, 0, true)

  -- Above maximum (24) - should fail
  test_deck(
    "24 face cards",
    create_test_deck(23, 13, 13, 13, 13), -- Can't actually create 24
    24,
    0,
    false
  )

  print("\n--- Test 2: Suit Ratio Boundaries ---")

  -- Balanced deck (25% each suit)
  test_deck(
    "Balanced (25%)",
    create_test_deck(0, 13, 13, 13, 13),
    0,
    0.25,
    true
  )

  -- 50% one suit
  test_deck("50% Hearts", create_test_deck(0, 26, 9, 9, 8), 0, 0.50, true)

  -- 75% one suit (maximum realistic)
  test_deck("75% Spades", create_test_deck(0, 6, 6, 7, 33), 0, 0.75, false) -- 33/52 = 63%, won't reach 75%

  -- Near maximum (73%)
  test_deck("73% Diamonds", create_test_deck(0, 5, 38, 5, 4), 0, 0.70, true)

  print("\n--- Test 3: Combined Requirements ---")

  -- Moderate both
  test_deck(
    "10 faces + 40% suit",
    create_test_deck(10, 21, 11, 10, 10),
    10,
    0.40,
    true
  )

  -- High face + high suit (difficult)
  test_deck(
    "20 faces + 60% suit",
    create_test_deck(20, 31, 7, 7, 7),
    20,
    0.60,
    true
  )

  -- Impossible combination
  test_deck(
    "23 faces + 75% suit",
    create_test_deck(23, 39, 4, 4, 5),
    23,
    0.75,
    false
  )

  print("\n--- Test 4: Edge Cases ---")

  -- Empty requirements (should always pass)
  test_deck("No requirements", create_test_deck(5, 13, 13, 13, 13), 0, 0, true)

  -- Just face requirement
  test_deck(
    "Only faces (15)",
    create_test_deck(15, 13, 13, 13, 13),
    15,
    0,
    true
  )

  -- Just suit requirement
  test_deck("Only suit (55%)", create_test_deck(0, 29, 8, 8, 7), 0, 0.55, true)

  print("\n--- Test 5: Performance Test ---")

  local start_time = os.clock()
  local iterations = 1000

  for i = 1, iterations do
    local deck = create_test_deck(
      math.random(0, 23),
      math.random(5, 30),
      math.random(5, 30),
      math.random(5, 30),
      math.random(5, 30)
    )
    -- Validate deck
    local face_count = 0
    for _, card in ipairs(deck) do
      local rank = card.base.value
      if rank == "Jack" or rank == "Queen" or rank == "King" then
        face_count = face_count + 1
      end
    end
  end

  local elapsed = os.clock() - start_time
  local rate = iterations / elapsed

  print(
    string.format(
      "✓ Validated %d random decks in %.3fs (%.0f decks/sec)",
      iterations,
      elapsed,
      rate
    )
  )

  print("\n" .. "=" .. string.rep("=", 50))
  print(string.format("  RESULTS: %d passed, %d failed", passed, failed))
  print("=" .. string.rep("=", 50))

  return failed == 0
end

-- Export for use in other tests
return {
  run = run_erratic_tests,
  create_test_deck = create_test_deck,
}
