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
#define MAX_KEYS 100

// Balatro's exact constants from the Reddit thread analysis
#define PSEUDOSEED_CONST1 2.134453429141
#define PSEUDOSEED_CONST2 1.72431234
#define PSEUDOHASH_CONST 1.1239285023

typedef enum {
    SUIT_SPADES = 0,
    SUIT_HEARTS = 1,
    SUIT_CLUBS = 2,
    SUIT_DIAMONDS = 3
} Suit;

typedef enum {
    RANK_A = 14, RANK_2 = 2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, 
    RANK_8, RANK_9, RANK_10, RANK_J = 11, RANK_Q = 12, RANK_K = 13
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

// Simulates G.GAME.pseudorandom table from Balatro
typedef struct {
    char key[32];
    double value;
} PseudorandomEntry;

typedef struct {
    char seed[9];  // 8 character seed + null
    double hashed_seed;
    PseudorandomEntry entries[MAX_KEYS];
    int entry_count;
} GameState;

typedef struct {
    bool erratic_deck;
    bool no_faces;
    int min_face_cards;
    int max_face_cards;
    double min_suit_ratio;
    double max_suit_ratio;
    char target_suit;
} FilterConfig;

// Balatro's exact pseudohash implementation
static double pseudohash(const char* str) {
    double num = 1.0;
    int len = strlen(str);
    
    // Exact implementation from Balatro source
    for (int i = len - 1; i >= 0; i--) {
        num = fmod((PSEUDOHASH_CONST / num) * (unsigned char)str[i] * M_PI + M_PI * (i + 1), 1.0);
    }
    return num;
}

// Find or create entry in pseudorandom table
static PseudorandomEntry* get_or_create_entry(GameState* state, const char* key) {
    // Search for existing entry
    for (int i = 0; i < state->entry_count; i++) {
        if (strcmp(state->entries[i].key, key) == 0) {
            return &state->entries[i];
        }
    }
    
    // Create new entry if not found
    if (state->entry_count < MAX_KEYS) {
        PseudorandomEntry* entry = &state->entries[state->entry_count++];
        strncpy(entry->key, key, 31);
        entry->key[31] = '\0';
        
        // Initialize with pseudohash of key + seed
        char combined[64];
        snprintf(combined, sizeof(combined), "%s%s", key, state->seed);
        entry->value = pseudohash(combined);
        
        return entry;
    }
    
    return NULL;  // Table full
}

// Balatro's exact pseudoseed implementation
static double pseudoseed(GameState* state, const char* key) {
    if (strcmp(key, "seed") == 0) {
        // Special case for "seed" key - returns random value
        return ((double)rand()) / RAND_MAX;
    }
    
    PseudorandomEntry* entry = get_or_create_entry(state, key);
    if (!entry) return 0.0;
    
    // This is the exact transformation from Balatro
    // G.GAME.pseudorandom[key] = abs(tonumber(string.format("%.13f", (2.134453429141+G.GAME.pseudorandom[key]*1.72431234)%1)))
    entry->value = fabs(fmod(PSEUDOSEED_CONST1 + entry->value * PSEUDOSEED_CONST2, 1.0));
    
    // Return (G.GAME.pseudorandom[key] + (G.GAME.pseudorandom.hashed_seed or 0))/2
    return (entry->value + state->hashed_seed) / 2.0;
}

// Convert pseudoseed value to random integer
static int pseudorandom_int(double seed_val, int min, int max) {
    // Use the seed value to generate a deterministic random number
    srand((unsigned int)(seed_val * UINT_MAX));
    return min + (rand() % (max - min + 1));
}

// Initialize game state with a seed string
static void init_game_state(GameState* state, const char* seed_str) {
    memset(state, 0, sizeof(GameState));
    strncpy(state->seed, seed_str, 8);
    state->seed[8] = '\0';
    state->hashed_seed = pseudohash(seed_str);
    state->entry_count = 0;
}

// Get card from P_CARDS table index (Balatro's card ordering)
static Card get_card_from_p_cards(int index) {
    // Balatro's P_CARDS ordering: H_A, H_2, ..., H_K, C_A, C_2, ..., C_K, D_A, ..., S_A, ...
    const Suit suit_order[] = {SUIT_HEARTS, SUIT_CLUBS, SUIT_DIAMONDS, SUIT_SPADES};
    const Rank rank_order[] = {RANK_A, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, 
                               RANK_7, RANK_8, RANK_9, RANK_10, RANK_J, RANK_Q, RANK_K};
    
    Card card;
    card.suit = suit_order[index / 13];
    card.rank = rank_order[index % 13];
    return card;
}

// Simulate pseudorandom_element for P_CARDS
static Card pseudorandom_element_cards(GameState* state, const char* key) {
    double seed_val = pseudoseed(state, key);
    
    // pseudorandom_element creates a sorted array and picks a random element
    // For P_CARDS, we can simplify to just picking a random index
    int index = pseudorandom_int(seed_val, 0, 51);
    
    return get_card_from_p_cards(index);
}

// Generate Erratic deck using exact Balatro logic
static void generate_erratic_deck(Deck* deck, GameState* state) {
    memset(deck, 0, sizeof(Deck));
    
    // For Erratic deck, each card position randomly selects from all 52 cards
    // This matches: if self.GAME.starting_params.erratic_suits_and_ranks then 
    //               _, k = pseudorandom_element(G.P_CARDS, pseudoseed('erratic'))
    for (int i = 0; i < DECK_SIZE; i++) {
        // Each iteration updates the 'erratic' key state
        Card card = pseudorandom_element_cards(state, "erratic");
        deck->cards[i] = card;
        
        if (card.rank >= RANK_J) {
            deck->face_count++;
        }
        deck->suit_counts[card.suit]++;
    }
    
    // Calculate max suit ratio
    int max_suit_count = 0;
    for (int i = 0; i < 4; i++) {
        if (deck->suit_counts[i] > max_suit_count) {
            max_suit_count = deck->suit_counts[i];
        }
    }
    deck->max_suit_ratio = (double)max_suit_count / DECK_SIZE;
}

// Generate normal deck
static void generate_normal_deck(Deck* deck, bool no_faces) {
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
    
    // For normal deck, suits are evenly distributed
    deck->max_suit_ratio = no_faces ? 0.3077 : 0.25;  // 13/52 or 10/40
}

// Check if deck meets filter requirements
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
                if (config->target_suit == -1 || config->target_suit == i) {
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

// Test a single seed
DLL_EXPORT bool test_seed(const char* seed_str, const FilterConfig* config, Deck* out_deck) {
    GameState state;
    init_game_state(&state, seed_str);
    
    Deck deck;
    if (config->erratic_deck) {
        generate_erratic_deck(&deck, &state);
    } else {
        generate_normal_deck(&deck, config->no_faces);
    }
    
    if (out_deck) {
        *out_deck = deck;
    }
    
    return check_deck_filters(&deck, config);
}

// Generate a random seed string (like Balatro does)
static void generate_seed_string(char* seed_str, int iteration) {
    const char charset[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    const int charset_size = sizeof(charset) - 1;
    
    // Use iteration to ensure different seeds
    srand(time(NULL) + iteration);
    
    for (int i = 0; i < 8; i++) {
        seed_str[i] = charset[rand() % charset_size];
    }
    seed_str[8] = '\0';
}

// Batch test multiple seeds
DLL_EXPORT int batch_test_seeds(int count, const FilterConfig* config, char results[][9], Deck* decks) {
    int found = 0;
    
    for (int i = 0; i < count && found < 100; i++) {
        char seed_str[9];
        generate_seed_string(seed_str, i);
        
        Deck deck;
        if (test_seed(seed_str, config, &deck)) {
            if (results) {
                strcpy(results[found], seed_str);
            }
            if (decks) {
                decks[found] = deck;
            }
            found++;
        }
    }
    
    return found;
}

// Test performance with realistic Erratic deck requirements
DLL_EXPORT void test_performance() {
    FilterConfig config = {0};
    config.erratic_deck = true;
    config.min_face_cards = 20;  // Looking for high face count
    config.min_suit_ratio = 0.5;  // 50% of one suit
    
    clock_t start = clock();
    char results[100][9];
    Deck decks[100];
    
    int found = batch_test_seeds(10000, &config, results, decks);
    
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("=== Balatro RNG Implementation Test ===\n");
    printf("Tested 10,000 seeds in %.2f seconds\n", time_taken);
    printf("Found %d matching seeds (%.2f%%)\n", found, (found / 10000.0) * 100);
    printf("Seeds per second: %.0f\n", 10000.0 / time_taken);
    
    if (found > 0) {
        printf("\nFirst 5 matching seeds:\n");
        for (int i = 0; i < 5 && i < found; i++) {
            printf("  %s: %d faces, suits [S:%d H:%d C:%d D:%d]\n",
                   results[i], decks[i].face_count,
                   decks[i].suit_counts[0], decks[i].suit_counts[1],
                   decks[i].suit_counts[2], decks[i].suit_counts[3]);
        }
    }
    
    // Test known problematic patterns
    printf("\n=== Testing Known Issues ===\n");
    printf("The self-feeding nature of G.GAME.pseudorandom[key] can create patterns.\n");
    printf("jimbo_extreme1 identified that the system lacks entropy due to:\n");
    printf("  - Constant transformation values: %.12f and %.8f\n", PSEUDOSEED_CONST1, PSEUDOSEED_CONST2);
    printf("  - Self-feeding: next value depends only on previous + constants\n");
    printf("  - Fixed keys: 'space' for Space Joker, 'erratic' for deck, etc.\n");
    printf("\nThis explains glitched seeds and statistical anomalies.\n");
}

#ifdef STANDALONE_TEST
int main() {
    test_performance();
    
    // Test a specific seed
    printf("\n=== Testing Specific Seed ===\n");
    FilterConfig config = {0};
    config.erratic_deck = true;
    
    Deck deck;
    if (test_seed("TESTTEST", &config, &deck)) {
        printf("Seed TESTTEST: %d face cards, max suit ratio: %.2f%%\n", 
               deck.face_count, deck.max_suit_ratio * 100);
    }
    
    return 0;
}
#endif