local ffi = require("ffi")

ffi.cdef[[
    typedef struct {
        int suit;
        int rank;
    } Card;

    typedef struct {
        Card cards[52];
        int face_count;
        int suit_counts[4];
        double max_suit_ratio;
    } Deck;

    typedef struct {
        bool erratic_deck;
        bool no_faces;
        int min_face_cards;
        int max_face_cards;
        double min_suit_ratio;
        double max_suit_ratio;
        char target_suit;
        
        bool check_vouchers;
        char required_vouchers[10][32];
        int num_required_vouchers;
        
        bool check_tags;
        char required_tags[10][32];
        int num_required_tags;
    } FilterConfig;

    typedef struct {
        uint64_t seed;
        Deck deck;
        bool matches_filters;
        char vouchers[10][32];
        int num_vouchers;
        char tags[10][32];
        int num_tags;
    } SeedResult;

    SeedResult* test_seeds(uint64_t start_seed, int count, const FilterConfig* config, int* num_results);
    void free_results(SeedResult* results);
    void test_performance();
]]

local immolate = ffi.load("immolate")

local ImmolateLib = {}

function ImmolateLib.create_config()
    return ffi.new("FilterConfig")
end

function ImmolateLib.test_seeds(start_seed, count, config)
    local num_results = ffi.new("int[1]")
    local results = immolate.test_seeds(start_seed, count, config, num_results)
    
    local lua_results = {}
    for i = 0, num_results[0] - 1 do
        local result = results[i]
        local lua_result = {
            seed = tonumber(result.seed),
            face_count = result.deck.face_count,
            suit_counts = {
                result.deck.suit_counts[0],
                result.deck.suit_counts[1],
                result.deck.suit_counts[2],
                result.deck.suit_counts[3]
            },
            max_suit_ratio = result.deck.max_suit_ratio,
            vouchers = {},
            tags = {}
        }
        
        for j = 0, result.num_vouchers - 1 do
            table.insert(lua_result.vouchers, ffi.string(result.vouchers[j]))
        end
        
        for j = 0, result.num_tags - 1 do
            table.insert(lua_result.tags, ffi.string(result.tags[j]))
        end
        
        table.insert(lua_results, lua_result)
    end
    
    immolate.free_results(results)
    return lua_results
end

function ImmolateLib.test_single_seed(seed, config)
    local results = ImmolateLib.test_seeds(seed, 1, config)
    return results[1]
end

function ImmolateLib.batch_test(start_seed, batch_size, config, callback)
    local current_seed = start_seed
    local total_found = 0
    local total_tested = 0
    
    while true do
        local results = ImmolateLib.test_seeds(current_seed, batch_size, config)
        total_tested = total_tested + batch_size
        
        if #results > 0 then
            for _, result in ipairs(results) do
                if callback(result) then
                    return result
                end
                total_found = total_found + 1
            end
        end
        
        current_seed = current_seed + batch_size
        
        if total_tested % 100000 == 0 then
            print(string.format("Tested %d seeds, found %d matches", total_tested, total_found))
        end
    end
end

function ImmolateLib.configure_for_brainstorm(brainstorm_config)
    local config = ImmolateLib.create_config()
    
    config.erratic_deck = brainstorm_config.erratic_deck or false
    config.no_faces = brainstorm_config.no_faces or false
    
    if brainstorm_config.face_cards then
        config.min_face_cards = brainstorm_config.face_cards.min or 0
        config.max_face_cards = brainstorm_config.face_cards.max or 52
    end
    
    if brainstorm_config.suit_ratio then
        config.min_suit_ratio = brainstorm_config.suit_ratio.min or 0
        config.max_suit_ratio = brainstorm_config.suit_ratio.max or 1
        
        if brainstorm_config.suit_ratio.target then
            local suit_map = {spades = 0, hearts = 1, clubs = 2, diamonds = 3}
            config.target_suit = suit_map[brainstorm_config.suit_ratio.target:lower()] or 0
        end
    end
    
    if brainstorm_config.vouchers and #brainstorm_config.vouchers > 0 then
        config.check_vouchers = true
        config.num_required_vouchers = math.min(#brainstorm_config.vouchers, 10)
        for i = 1, config.num_required_vouchers do
            ffi.copy(config.required_vouchers[i-1], brainstorm_config.vouchers[i])
        end
    end
    
    if brainstorm_config.tags and #brainstorm_config.tags > 0 then
        config.check_tags = true
        config.num_required_tags = math.min(#brainstorm_config.tags, 10)
        for i = 1, config.num_required_tags do
            ffi.copy(config.required_tags[i-1], brainstorm_config.tags[i])
        end
    end
    
    return config
end

function ImmolateLib.test_performance()
    immolate.test_performance()
end

return ImmolateLib