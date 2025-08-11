#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

#define DECK_SIZE 52
#define MAX_FACE_CARDS 12
#define ERRATIC_DECK_SIZE 52

typedef enum {
    SUIT_SPADES = 0,
    SUIT_HEARTS = 1,
    SUIT_CLUBS = 2,
    SUIT_DIAMONDS = 3
} Suit;

typedef enum {
    RANK_2 = 2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_9, RANK_10,
    RANK_J = 11, RANK_Q = 12, RANK_K = 13, RANK_A = 14
} Rank;

typedef struct {
    Suit suit;
    Rank rank;
} Card;

typedef struct {
    Card cards[DECK_SIZE];
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

static uint32_t rng_state;

static double pseudohash(const char* str, uint64_t seed) {
    double num = 1.0;
    int len = strlen(str);
    char combined[256];
    snprintf(combined, sizeof(combined), "%s%llu", str, seed);
    len = strlen(combined);
    
    for (int i = len - 1; i >= 0; i--) {
        num = fmod((1.1239285023 / num) * combined[i] * M_PI + M_PI * i, 1.0);
    }
    return num;
}

static void seed_rng(uint64_t seed) {
    rng_state = (uint32_t)(seed ^ (seed >> 32));
    if (rng_state == 0) rng_state = 1;
}

static uint32_t xorshift32() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static double pseudorandom() {
    return xorshift32() / (double)UINT32_MAX;
}

static int pseudorandom_int(int min, int max) {
    return min + (xorshift32() % (max - min + 1));
}

static Card get_card_from_index(int index) {
    Card card;
    const Suit suits[] = {SUIT_SPADES, SUIT_HEARTS, SUIT_CLUBS, SUIT_DIAMONDS};
    const Rank ranks[] = {RANK_A, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, 
                          RANK_7, RANK_8, RANK_9, RANK_10, RANK_J, RANK_Q, RANK_K};
    
    card.suit = suits[index % 4];
    card.rank = ranks[index / 4];
    return card;
}

static void generate_erratic_deck(Deck* deck, uint64_t seed) {
    memset(deck, 0, sizeof(Deck));
    
    double erratic_hash = pseudohash("erratic", seed);
    seed_rng((uint64_t)(erratic_hash * UINT64_MAX));
    
    for (int i = 0; i < DECK_SIZE; i++) {
        int random_index = pseudorandom_int(0, 51);
        deck->cards[i] = get_card_from_index(random_index);
        
        if (deck->cards[i].rank >= RANK_J) {
            deck->face_count++;
        }
        deck->suit_counts[deck->cards[i].suit]++;
    }
    
    int max_suit_count = 0;
    for (int i = 0; i < 4; i++) {
        if (deck->suit_counts[i] > max_suit_count) {
            max_suit_count = deck->suit_counts[i];
        }
    }
    deck->max_suit_ratio = (double)max_suit_count / DECK_SIZE;
}

static void generate_normal_deck(Deck* deck, uint64_t seed, bool no_faces) {
    memset(deck, 0, sizeof(Deck));
    
    int card_index = 0;
    for (int suit = 0; suit < 4; suit++) {
        for (int rank = 2; rank <= 14; rank++) {
            if (no_faces && (rank == RANK_J || rank == RANK_Q || rank == RANK_K)) {
                continue;
            }
            
            deck->cards[card_index].suit = suit;
            deck->cards[card_index].rank = rank;
            
            if (rank >= RANK_J) {
                deck->face_count++;
            }
            deck->suit_counts[suit]++;
            card_index++;
        }
    }
    
    deck->max_suit_ratio = 0.25;
}

static bool check_deck_filters(const Deck* deck, const FilterConfig* config) {
    if (config->min_face_cards > 0 && deck->face_count < config->min_face_cards) {
        return false;
    }
    
    if (config->max_face_cards > 0 && deck->face_count > config->max_face_cards) {
        return false;
    }
    
    if (config->min_suit_ratio > 0) {
        bool found_suit = false;
        for (int i = 0; i < 4; i++) {
            double ratio = (double)deck->suit_counts[i] / DECK_SIZE;
            if (ratio >= config->min_suit_ratio) {
                if (config->target_suit == 0 || config->target_suit == i) {
                    found_suit = true;
                    break;
                }
            }
        }
        if (!found_suit) return false;
    }
    
    if (config->max_suit_ratio > 0 && deck->max_suit_ratio > config->max_suit_ratio) {
        return false;
    }
    
    return true;
}

static void generate_vouchers(SeedResult* result, uint64_t seed) {
    const char* voucher_pool[] = {
        "v_overstock_norm", "v_clearance_sale", "v_tarot_merchant",
        "v_planet_merchant", "v_hone", "v_glow_up", "v_reroll_surplus",
        "v_crystal_ball", "v_telescope", "v_grabber", "v_wasteful",
        "v_seed_money", "v_blank", "v_magic_trick", "v_hieroglyph",
        "v_directors_cut", "v_paint_brush", "v_overstock_plus",
        "v_liquidation", "v_tarot_tycoon", "v_planet_tycoon",
        "v_observatory", "v_nacho_tong", "v_recyclomancy",
        "v_money_tree", "v_antimatter", "v_illusion", "v_petroglyph",
        "v_retcon", "v_palette"
    };
    int num_vouchers = sizeof(voucher_pool) / sizeof(voucher_pool[0]);
    
    double shop_hash = pseudohash("shop_voucher", seed);
    seed_rng((uint64_t)(shop_hash * UINT64_MAX));
    
    result->num_vouchers = 0;
    if (pseudorandom() < 0.4) {
        int voucher_index = pseudorandom_int(0, num_vouchers - 1);
        strcpy(result->vouchers[result->num_vouchers++], voucher_pool[voucher_index]);
    }
}

static void generate_tags(SeedResult* result, uint64_t seed) {
    const char* tag_pool[] = {
        "tag_uncommon", "tag_rare", "tag_negative", "tag_foil",
        "tag_holo", "tag_polychrome", "tag_investment", "tag_voucher",
        "tag_boss", "tag_standard", "tag_charm", "tag_meteor",
        "tag_buffoon", "tag_handy", "tag_garbage", "tag_coupon",
        "tag_double", "tag_juggle", "tag_d_six", "tag_top_up",
        "tag_speed", "tag_orbital", "tag_economy"
    };
    int num_tags = sizeof(tag_pool) / sizeof(tag_pool[0]);
    
    double tag_hash = pseudohash("tag_generate", seed);
    seed_rng((uint64_t)(tag_hash * UINT64_MAX));
    
    result->num_tags = 0;
    if (pseudorandom() < 0.3) {
        int tag_index = pseudorandom_int(0, num_tags - 1);
        strcpy(result->tags[result->num_tags++], tag_pool[tag_index]);
    }
}

static bool check_voucher_tags(const SeedResult* result, const FilterConfig* config) {
    if (config->check_vouchers) {
        for (int i = 0; i < config->num_required_vouchers; i++) {
            bool found = false;
            for (int j = 0; j < result->num_vouchers; j++) {
                if (strcmp(config->required_vouchers[i], result->vouchers[j]) == 0) {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
    }
    
    if (config->check_tags) {
        for (int i = 0; i < config->num_required_tags; i++) {
            bool found = false;
            for (int j = 0; j < result->num_tags; j++) {
                if (strcmp(config->required_tags[i], result->tags[j]) == 0) {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
    }
    
    return true;
}

DLL_EXPORT SeedResult* test_seeds(uint64_t start_seed, int count, const FilterConfig* config, int* num_results) {
    SeedResult* results = malloc(sizeof(SeedResult) * count);
    int found = 0;
    
    for (int i = 0; i < count; i++) {
        uint64_t seed = start_seed + i;
        SeedResult* result = &results[found];
        result->seed = seed;
        
        if (config->erratic_deck) {
            generate_erratic_deck(&result->deck, seed);
        } else {
            generate_normal_deck(&result->deck, seed, config->no_faces);
        }
        
        if (!check_deck_filters(&result->deck, config)) {
            continue;
        }
        
        generate_vouchers(result, seed);
        generate_tags(result, seed);
        
        if (!check_voucher_tags(result, config)) {
            continue;
        }
        
        result->matches_filters = true;
        found++;
    }
    
    *num_results = found;
    return realloc(results, sizeof(SeedResult) * found);
}

DLL_EXPORT void free_results(SeedResult* results) {
    free(results);
}

DLL_EXPORT void test_performance() {
    FilterConfig config = {0};
    config.erratic_deck = true;
    config.min_face_cards = 20;
    config.min_suit_ratio = 0.5;
    
    clock_t start = clock();
    int num_results;
    SeedResult* results = test_seeds(0, 100000, &config, &num_results);
    clock_t end = clock();
    
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Tested 100,000 seeds in %.2f seconds\n", time_taken);
    printf("Found %d matching seeds\n", num_results);
    printf("Seeds per second: %.0f\n", 100000.0 / time_taken);
    
    if (num_results > 0) {
        printf("\nFirst matching seed: %llu\n", results[0].seed);
        printf("Face cards: %d\n", results[0].deck.face_count);
        printf("Suit distribution: S=%d H=%d C=%d D=%d\n",
               results[0].deck.suit_counts[0],
               results[0].deck.suit_counts[1],
               results[0].deck.suit_counts[2],
               results[0].deck.suit_counts[3]);
    }
    
    free_results(results);
}

#ifdef STANDALONE_TEST
int main() {
    test_performance();
    return 0;
}
#endif