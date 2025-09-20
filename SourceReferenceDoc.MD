# Balatro Source Code Reference Document

## Critical Discovery: Balatro's RNG System

### 1. The Core RNG Functions (functions/misc_functions.lua)

#### pseudohash(str) - Line 279
```lua
function pseudohash(str)
    local num = 1
    for i=#str, 1, -1 do
        num = ((1.1239285023/num)*string.byte(str, i)*math.pi + math.pi*i)%1
    end
    return num
end
```
**This is COMPLETELY DIFFERENT from FNV-1a or PCG!** It's a custom hash that:
- Iterates through string backwards
- Uses division, multiplication by pi, and modulo 1
- Returns a float between 0 and 1

#### pseudoseed(key, predict_seed) - Line 298
```lua
function pseudoseed(key, predict_seed)
    if not G.GAME.pseudorandom[key] then 
        G.GAME.pseudorandom[key] = pseudohash(key..(G.GAME.pseudorandom.seed or ''))
    end
    G.GAME.pseudorandom[key] = math.abs(tonumber(string.format("%.13f", (2.134453429141+G.GAME.pseudorandom[key]*1.72431234)%1)))
    return (G.GAME.pseudorandom[key] + (G.GAME.pseudorandom.hashed_seed or 0))/2
end
```
**Key points:**
- Stores RNG state per key in G.GAME.pseudorandom table
- Uses linear congruential formula with specific constants
- Returns average of state and hashed seed

#### pseudorandom(seed, min, max) - Line 315
```lua
function pseudorandom(seed, min, max)
    if type(seed) == 'string' then seed = pseudoseed(seed) end
    math.randomseed(seed)
    if min and max then return math.random(min, max)
    else return math.random() end
end
```
**Uses Lua's math.randomseed with the pseudoseed value!**

### 2. Shop Generation Process

#### Voucher Selection (functions/common_events.lua:1901)
```lua
function get_next_voucher_key(_from_tag)
    local _pool, _pool_key = get_current_pool('Voucher')
    if _from_tag then _pool_key = 'Voucher_fromtag' end
    local center = pseudorandom_element(_pool, pseudoseed(_pool_key))
    -- Resampling logic if UNAVAILABLE
    return center
end
```
**Context string: 'Voucher' (NOT 'Voucher' for FNV hash!)**

#### Pack Selection (functions/common_events.lua:1944)
```lua
function get_pack(_key, _type)
    -- Special case for first buffoon pack
    local poll = pseudorandom(pseudoseed((_key or 'pack_generic')..G.GAME.round_resets.ante))*cume
    -- Weighted selection from G.P_CENTER_POOLS['Booster']
end
```
**Context string: 'pack_generic' + ante number**

#### Tag Selection (functions/common_events.lua:1914)
```lua
function get_next_tag_key(append)
    local _pool, _pool_key = get_current_pool('Tag', nil, nil, append)
    local _tag = pseudorandom_element(_pool, pseudoseed(_pool_key))
    return _tag
end
```
**Context strings for tags:**
- Small blind: calls with no append
- Big blind: calls with no append (but different timing)

### 3. Shop Creation (game.lua:3139-3168)

The shop creates:
1. **Voucher**: Using `get_next_voucher_key()` stored in `G.GAME.current_round.voucher`
2. **Packs**: 2 packs using `get_pack('shop_pack')` stored in `G.GAME.current_round.used_packs[1]` and `[2]`
3. **Tags**: 
   - Small: `get_next_tag_key()` stored in `G.GAME.round_resets.blind_tags.Small`
   - Big: `get_next_tag_key()` stored in `G.GAME.round_resets.blind_tags.Big`

### 4. Critical Seed Format

From game.lua:2164:
```lua
self.GAME.pseudorandom.seed = args.seed or generate_starting_seed()
```

Seeds are 8 characters, using base-36 (0-9, A-Z):
- Valid characters: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
- Max value: 36^8 - 1 = 2,821,109,907,455

### 5. The ACTUAL Algorithm We Need

**WE ARE USING THE WRONG RNG ENTIRELY!**

The correct algorithm:
1. Hash the seed using `pseudohash()` (NOT FNV-1a!)
2. For each context (voucher, pack, tag):
   - Call `pseudoseed(context_key)` which:
     - Hashes `context_key + seed`
     - Updates state with linear congruential formula
     - Returns average of state and hashed_seed
3. Use result with `math.randomseed()` to select from pools

### 6. Pool Indices

#### Vouchers (32 total, starting at enum 162)
- Index 0: Overstock
- Index 2: Clearance Sale
- Index 16: Tarot Merchant
- etc.

#### Packs (15 types, 3 variants each)
- Index 0-2: Arcana (normal, jumbo, mega)
- Index 3-5: Celestial
- Index 6-8: Standard
- Index 9-11: Buffoon
- Index 12-14: Spectral

#### Tags (30 total, starting at enum 310)
- Index 6: Investment Tag
- Index 17: Double Tag
- etc.

### 7. Why Our Current Implementation is Wrong

**Current (WRONG) approach:**
- Using FNV-1a hash
- Using PCG algorithm
- Using wrong context strings ("StandardPack", "JumboStandardPack", etc.)

**Correct approach:**
- Use Balatro's `pseudohash()` function
- Use Balatro's `pseudoseed()` with state tracking
- Use correct context keys:
  - Voucher: "Voucher"
  - Pack: "shop_pack" or "pack_generic" + ante
  - Tags: "Tag" with pool modifiers

### 8. File Structure Summary

```
BalatroSource/extracted/
├── game.lua                    # Main game logic, shop creation
├── functions/
│   ├── misc_functions.lua      # RNG FUNCTIONS (pseudohash, pseudoseed, pseudorandom)
│   ├── common_events.lua       # get_next_voucher_key, get_pack, get_next_tag_key
│   ├── state_events.lua        # Round/ante progression
│   ├── button_callbacks.lua    # UI interactions
│   └── UI_definitions.lua      # Shop UI creation
├── tag.lua                      # Tag effects
├── card.lua                     # Card logic
└── globals.lua                  # Global constants
```

## CRITICAL FIXES NEEDED

1. **Implement pseudohash() correctly** - NOT FNV-1a!
2. **Implement pseudoseed() with state tracking** - NOT PCG!
3. **Use correct context strings** - NOT "StandardPack" etc!
4. **Track RNG state per context key** - Like G.GAME.pseudorandom table
5. **Use math.randomseed() for final selection** - NOT custom range mapping!

## The Real Problem

We've been implementing a completely different RNG system. Balatro uses:
- A custom hash function with pi and division
- A stateful linear congruential generator per context
- Lua's built-in math.random with custom seeds
- Specific context keys that don't match what we thought

This explains why the GPU is finding seeds but they don't match what the game generates!