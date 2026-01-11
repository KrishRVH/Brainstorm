# BalatroSource Guide (Seed/Search Relevant)

This guide is derived directly from the in-repo `BalatroSource/` code. Every point
below is backed by the referenced source files; do not extend this guide without
verifying in `BalatroSource/`.

## Scope and Primary Sources
- RNG and seed generation: `BalatroSource/functions/misc_functions.lua`
- Run initialization and core state: `BalatroSource/game.lua`
- Pools, tags, vouchers, packs, and card creation: `BalatroSource/functions/common_events.lua`
- Shop UI construction and shop card creation: `BalatroSource/functions/UI_definitions.lua`
- Shop reroll and booster usage: `BalatroSource/functions/button_callbacks.lua`
- Booster contents: `BalatroSource/card.lua`
- Erratic deck flag: `BalatroSource/back.lua`

## PRNG and Seeds
- `pseudohash(str)` hashes a string to a float in [0,1) using a fixed formula with
  `1.1239285023`, `math.pi`, and `% 1` (see `pseudohash` in
  `BalatroSource/functions/misc_functions.lua`).
- `pseudoseed(key, predict_seed)` drives deterministic RNG:
  - If `key == 'seed'`, it returns `math.random()` directly.
  - If `predict_seed` is supplied, it hashes `key..predict_seed`, transforms it
    with constants `2.134453429141` and `1.72431234`, and averages with
    `pseudohash(predict_seed)`.
  - Otherwise it updates `G.GAME.pseudorandom[key]` in-place and returns an
    average with `G.GAME.pseudorandom.hashed_seed`.
  - Each call mutates the per-key state, so call order matters.
  (See `pseudoseed` in `BalatroSource/functions/misc_functions.lua`.)
- `pseudorandom(seed, min, max)` converts string seeds through `pseudoseed`,
  calls `math.randomseed(seed)`, and returns `math.random()` or `math.random(min, max)`
  (see `pseudorandom` in `BalatroSource/functions/misc_functions.lua`).
- `pseudorandom_element(_t, seed)` builds a deterministic ordering of `_t`:
  it sorts by `sort_id` if present on entries, otherwise by key; then calls
  `math.randomseed(seed)` and selects `math.random(#keys)`. This means the
  sorted order is part of the RNG result (see `pseudorandom_element` in
  `BalatroSource/functions/misc_functions.lua`).
- `random_string(length, seed)` is used to generate user seeds. It seeds
  `math.randomseed(seed)` and builds an uppercase string from digits `1-9`,
  letters `A-N`, or `P-Z` (no `O`) based on `math.random()` thresholds
  (see `random_string` in `BalatroSource/functions/misc_functions.lua`).
- `generate_starting_seed()`:
  - For stake >= 8, it uses profile data to split legendary jokers into two
    sets based on `get_joker_win_sticker(v, true)`. If both sets are non-empty,
    it loops until `get_first_legendary(seed)` returns a key in the
    "lower-sticker" set.
  - Otherwise, it returns a `random_string(8, ...)` seeded with cursor position
    and time.
  (See `generate_starting_seed` and `get_first_legendary` in
  `BalatroSource/functions/misc_functions.lua`, and `get_joker_win_sticker` in
  the same file.)

## Run Initialization and Core State
- `Game:init_game_object()` defines `G.GAME` and includes fields used for seed
  simulation: `pseudorandom`, `round_resets`, `current_round`, `used_vouchers`,
  `banned_keys`, `shop`, `tags`, and rates like `joker_rate`, `tarot_rate`,
  `planet_rate`, and `spectral_rate` (see `BalatroSource/game.lua`).
- `Game:start_run(args)`:
  - If `args.seed` exists, `G.GAME.seeded = true`.
  - `G.GAME.pseudorandom.seed` is set to `args.seed`, `"TUTORIAL"`, or
    `generate_starting_seed()` depending on tutorial state.
  - `G.GAME.pseudorandom.hashed_seed = pseudohash(G.GAME.pseudorandom.seed)`.
  - It initializes `G.GAME.round_resets.blind_tags.Small/Big` from tutorial forced
    tags if present, otherwise via `get_next_tag_key()`.
  - It initializes `G.GAME.current_round.voucher` from a tutorial forced voucher
    if present, otherwise via `get_next_voucher_key()`.
  (See `Game:start_run` in `BalatroSource/game.lua`.)

## Blind Tags
- Tag selection uses `get_next_tag_key(append)`:
  - Builds a pool via `get_current_pool('Tag', nil, nil, append)`.
  - Uses `pseudorandom_element` with `pseudoseed(pool_key)` and resamples if
    `'UNAVAILABLE'`.
  - Pool inclusion checks `requires` discovery and `min_ante` (see `get_current_pool`
    and `get_next_tag_key` in `BalatroSource/functions/common_events.lua`).
- Tags are stored in `G.GAME.round_resets.blind_tags.Small/Big`:
  - Set at `start_run` (see `BalatroSource/game.lua`).
  - Refreshed after a boss blind is defeated (see `BalatroSource/functions/button_callbacks.lua`).

## Vouchers
- Voucher selection uses `get_next_voucher_key(_from_tag)`:
  - Builds a pool with `get_current_pool('Voucher')`.
  - Filters out used vouchers, requires dependencies if present, excludes those
    already in the voucher shop slot, and respects `banned_keys`.
  - Uses `pseudorandom_element` with `pseudoseed(pool_key)` and resamples when
    `'UNAVAILABLE'`.
  (See `get_current_pool` and `get_next_voucher_key` in
  `BalatroSource/functions/common_events.lua`.)
- `G.GAME.current_round.voucher` is:
  - Set at run start (tutorial forced voucher if present, otherwise `get_next_voucher_key`;
    see `BalatroSource/game.lua`).
  - Refreshed after boss blind (`BalatroSource/functions/state_events.lua`).
  - Used to spawn the single voucher slot in the shop (`BalatroSource/game.lua`).
- Redeeming a voucher sets `G.GAME.used_vouchers[center_key] = true` and clears
  `G.GAME.current_round.voucher` (see `Card:redeem` in `BalatroSource/card.lua`).

## Shop Generation
- Shop layout:
  - Joker slots are `G.GAME.shop.joker_max` (default 2).
  - Voucher slot count is 1.
  - Booster slots are 2.
  (See `G.UIDEF.shop` in `BalatroSource/functions/UI_definitions.lua`.)
- Joker/consumable selection uses `create_card_for_shop(area)`:
  - Chooses the card type by weighted rates:
    `joker_rate`, `tarot_rate`, `planet_rate`, `playing_card_rate`,
    `spectral_rate`.
  - Selection uses `pseudorandom(pseudoseed('cdt'..ante))` against total rate.
  - Shop cards call `create_card(..., key_append='sho')`, so Joker rarity uses
    `pseudorandom('rarity'..ante..'sho')` and Joker pools use
    `Joker<rarity>sho<ante>`.
  - If `v_illusion` is used, playing-card type can flip between `Base` and
    `Enhanced` via `pseudorandom(pseudoseed('illusion'))`.
  (See `create_card_for_shop` in `BalatroSource/functions/UI_definitions.lua`
  and `create_card` / `get_current_pool` in
  `BalatroSource/functions/common_events.lua`.)
- `create_card(_type, ...)`:
  - Pulls from pools using `get_current_pool` + `pseudorandom_element` and
    resamples on `'UNAVAILABLE'`.
  - Adds editions to jokers via `poll_edition('edi'..ante)` and applies
    eternal/perishable/rental modifiers based on shop modifiers.
  (See `create_card` and `poll_edition` in `BalatroSource/functions/common_events.lua`.)
- `reroll_shop` only re-rolls the Joker slots; vouchers and boosters are not
  re-rolled (see `G.FUNCS.reroll_shop` in
  `BalatroSource/functions/button_callbacks.lua`).

## Booster Packs
- Pack selection (shop):
  - `G.GAME.current_round.used_packs` is reset to `{}` each round
    (see `new_round` in `BalatroSource/functions/state_events.lua`).
  - When the shop opens, each pack slot sets `used_packs[i] = get_pack('shop_pack').key`
    if unset (see shop creation in `BalatroSource/game.lua`); repeated calls
    advance the same `shop_pack..ante` RNG key.
  - Pack cards are created only if `used_packs[i] ~= 'USED'`.
- `get_pack(_key, _type)`:
  - First shop pack is forced to Buffoon if `G.GAME.first_shop_buffoon` is false
    and `p_buffoon_normal_1` is not banned; it uses `math.random(1, 2)` to pick
    `p_buffoon_normal_1` or `p_buffoon_normal_2`.
  - Otherwise it polls `G.P_CENTER_POOLS['Booster']` with weights using
    `pseudorandom(pseudoseed((_key or 'pack_generic')..ante))`.
  (See `get_pack` in `BalatroSource/functions/common_events.lua`.)
- Buying a booster marks the slot as used: `used_packs[booster_pos] = 'USED'`
  (see `G.FUNCS.buy_from_shop` in `BalatroSource/functions/button_callbacks.lua`).
- Pack contents (opened in `Card:open`):
  - Arcana packs: If `v_omen_globe` is used and `pseudorandom('omen_globe') > 0.8`,
    the card is Spectral; otherwise Tarot.
  - Celestial packs: If `v_telescope` is used and `i == 1`, the first card is
    forced to the most-played hand's planet; otherwise Planet.
  - Standard packs: Type is `Enhanced` if `pseudorandom(pseudoseed('stdset'..ante)) > 0.6`,
    otherwise `Base`; edition and seals are added with `standard_edition`,
    `stdseal`, and `stdsealtype` seeds.
  - Buffoon packs: Card type is Joker.
  (See `Card:open` in `BalatroSource/card.lua`.)
- Buffoon pack sizing (from `G.P_CENTERS`):
  - Normal Buffoon: `extra = 2`, `choose = 1`.
  - Jumbo Buffoon: `extra = 4`, `choose = 1`.
  - Mega Buffoon: `extra = 4`, `choose = 2`.
  (See `p_buffoon_*` entries in `BalatroSource/game.lua`.)
- Pack RNG key appends (used as `key_append` in `create_card`):
  - Arcana Tarot: `ar1`; Arcana via Omen Globe (Spectral): `ar2`.
  - Celestial: `pl1`. Spectral: `spe`. Standard: `sta`. Buffoon: `buf`.
  (See `Card:open` in `BalatroSource/card.lua`.)
- Soul cards:
  - In `create_card`, if the card is "soulable" and the relevant key is not
    banned, then `pseudorandom('soul_'.._type..ante) > 0.997` can force
    `c_soul` (Tarot/Spectral/Tarot_Planet) or `c_black_hole` (Planet/Spectral).
  (See `create_card` in `BalatroSource/functions/common_events.lua`.)
- Using `The Soul`:
  - `Card:use_consumeable` calls `create_card('Joker', ..., legendary=true, key_append='sou')`.
  - Legendary selection uses `get_current_pool('Joker', ..., _legendary=true)` which picks from
    `G.P_JOKER_RARITY_POOLS[4]` with pool key `Joker4` and **no** ante suffix.
  - Joker editions for this path use `poll_edition('edi'..'sou'..ante)`.
  (See `Card:use_consumeable` in `BalatroSource/card.lua` and `create_card` /
  `get_current_pool` in `BalatroSource/functions/common_events.lua`.)

## Erratic Deck
- The Erratic deck sets `G.GAME.starting_params.erratic_suits_and_ranks = true`
  when the selected Back has `randomize_rank_suit` (see `Back:apply_to_run` in
  `BalatroSource/back.lua`).
- During deck creation in `Game:start_run`, if `erratic_suits_and_ranks` is true,
  each iteration uses `pseudorandom_element(G.P_CARDS, pseudoseed('erratic'))`
  to pick a card key before applying deck restrictions and `no_faces`. The
  resulting `card_protos` are sorted, then `card_from_control` creates the cards,
  and `starting_deck_size` is set to `#G.playing_cards`.
  (See deck creation loop in `BalatroSource/game.lua`.)
