// Erratic Deck support for Immolate
// Adds the missing RNG types and deck initialization

#ifndef ERRATIC_SUPPORT_CL
#define ERRATIC_SUPPORT_CL

// Add to the rtype enum if not already defined
#ifndef R_Erratic
#define R_Erratic 999  // Custom RNG type for Erratic deck generation
#endif

// Balatro's P_CARDS ordering for Erratic deck
// Order: H_A, H_2...H_K, C_A...C_K, D_A...D_K, S_A...S_K
__constant item P_CARDS[52] = {
    H_A, H_2, H_3, H_4, H_5, H_6, H_7, H_8, H_9, H_10, H_J, H_Q, H_K,
    C_A, C_2, C_3, C_4, C_5, C_6, C_7, C_8, C_9, C_10, C_J, C_Q, C_K,
    D_A, D_2, D_3, D_4, D_5, D_6, D_7, D_8, D_9, D_10, D_J, D_Q, D_K,
    S_A, S_2, S_3, S_4, S_5, S_6, S_7, S_8, S_9, S_10, S_J, S_Q, S_K
};

// Initialize deck based on deck type
void init_deck(instance* inst, item deck[52]) {
    if (inst->params.deck == Erratic_Deck) {
        // Erratic deck: each position randomly selects from all 52 cards
        for (int i = 0; i < 52; i++) {
            // Use the 'erratic' RNG key with card index for deterministic generation
            // This maintains state across calls like Balatro does
            int card_idx = randint(inst, 
                (__private ntype[]){N_Type, N_CardIndex}, 
                (__private int[]){R_Erratic, i}, 
                2, 
                0, 51);
            deck[i] = P_CARDS[card_idx];
        }
    } else {
        // Normal deck generation (existing behavior)
        int idx = 0;
        for (int suit = 0; suit < 4; suit++) {
            for (int rank = 0; rank < 13; rank++) {
                // Skip face cards if no_faces is set
                if (inst->params.deckSize < 52 && (rank == 9 || rank == 10 || rank == 11)) {
                    continue;
                }
                deck[idx++] = CARDS[suit * 13 + rank + 1];  // +1 because CARDS[0] is size
            }
        }
    }
}

// Helper function to set deck type
void set_deck(instance* inst, item deck_type) {
    inst->params.deck = deck_type;
    if (deck_type == Erratic_Deck) {
        inst->params.deckSize = 52;  // Erratic always has 52 cards
    }
}

// Utility functions for deck analysis
typedef struct DeckStats {
    int face_count;
    int suit_counts[4];
    float max_suit_ratio;
    int most_common_card;
    int most_common_count;
} deck_stats;

deck_stats analyze_deck(item deck[52]) {
    deck_stats stats;
    stats.face_count = 0;
    for (int i = 0; i < 4; i++) stats.suit_counts[i] = 0;
    
    int card_counts[52];
    for (int i = 0; i < 52; i++) card_counts[i] = 0;
    
    for (int i = 0; i < 52; i++) {
        item card = deck[i];
        
        // Count this specific card
        if (card >= C_2 && card <= S_A) {
            card_counts[card - C_2]++;
        }
        
        // Check if face card
        int rank = (card - C_2) % 13;
        if (rank == 9 || rank == 10 || rank == 11) {  // J, Q, K
            stats.face_count++;
        }
        
        // Count suit
        int suit = (card - C_2) / 13;
        if (suit >= 0 && suit < 4) {
            stats.suit_counts[suit]++;
        }
    }
    
    // Find max suit ratio
    int max_suit = 0;
    for (int i = 0; i < 4; i++) {
        if (stats.suit_counts[i] > max_suit) {
            max_suit = stats.suit_counts[i];
        }
    }
    stats.max_suit_ratio = (float)max_suit / 52.0f;
    
    // Find most common card
    stats.most_common_count = 0;
    stats.most_common_card = -1;
    for (int i = 0; i < 52; i++) {
        if (card_counts[i] > stats.most_common_count) {
            stats.most_common_count = card_counts[i];
            stats.most_common_card = i + C_2;
        }
    }
    
    return stats;
}

#endif // ERRATIC_SUPPORT_CL