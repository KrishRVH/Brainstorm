// Erratic Deck filter optimized for Brainstorm mod requirements
// Finds seeds with specific face card counts and suit ratios

#include "lib/immolate.cl"

// Quick implementation that works with existing Immolate structure
long filter(instance* inst) {
    // User-configurable parameters (can be adjusted)
    const int MIN_FACE_CARDS = 20;      // Minimum face cards
    const int MAX_FACE_CARDS = 52;      // Maximum face cards (52 = no limit)
    const float MIN_SUIT_RATIO = 0.5;   // Minimum suit ratio (0.5 = 50%)
    const int TARGET_SUIT = -1;         // -1 = any suit, 0=C, 1=D, 2=H, 3=S
    
    set_deck(inst, Erratic_Deck);
    
    // Generate the deck using Erratic algorithm
    item deck[52];
    for (int i = 0; i < 52; i++) {
        // Each card position gets a random card from all 52
        // Using 'erratic' as the RNG key like Balatro does
        double seed_val = random(inst, 
            (__private ntype[]){N_Type, N_Ante, N_CardIndex}, 
            (__private int[]){R_Card, 1, i}, 
            3);
        
        // Convert to card index (0-51)
        int card_idx = (int)(seed_val * 52) % 52;
        
        // Map to actual card
        // Order matches Balatro's P_CARDS
        __constant item CARD_ORDER[52] = {
            H_A, H_2, H_3, H_4, H_5, H_6, H_7, H_8, H_9, H_10, H_J, H_Q, H_K,
            C_A, C_2, C_3, C_4, C_5, C_6, C_7, C_8, C_9, C_10, C_J, C_Q, C_K,
            D_A, D_2, D_3, D_4, D_5, D_6, D_7, D_8, D_9, D_10, D_J, D_Q, D_K,
            S_A, S_2, S_3, S_4, S_5, S_6, S_7, S_8, S_9, S_10, S_J, S_Q, S_K
        };
        deck[i] = CARD_ORDER[card_idx];
    }
    
    // Count face cards and suits
    int face_count = 0;
    int suit_counts[4] = {0, 0, 0, 0};
    int card_counts[52] = {0};
    
    for (int i = 0; i < 52; i++) {
        item card = deck[i];
        
        // Track individual card frequency
        if (card >= C_2 && card <= S_A) {
            card_counts[card - C_2]++;
        }
        
        // Determine card properties
        int card_num = card - C_2;
        int suit = card_num / 13;
        int rank = card_num % 13;
        
        // Count face cards (J=9, Q=10, K=11 in 0-indexed)
        if (rank == 9 || rank == 10 || rank == 11) {
            face_count++;
        }
        
        // Count suits
        if (suit >= 0 && suit < 4) {
            suit_counts[suit]++;
        }
    }
    
    // Check face card requirements
    if (face_count < MIN_FACE_CARDS || face_count > MAX_FACE_CARDS) {
        return 0;
    }
    
    // Check suit ratio requirements
    int max_suit_count = 0;
    int max_suit_idx = -1;
    for (int i = 0; i < 4; i++) {
        if (suit_counts[i] > max_suit_count) {
            max_suit_count = suit_counts[i];
            max_suit_idx = i;
        }
    }
    
    float suit_ratio = (float)max_suit_count / 52.0f;
    if (suit_ratio < MIN_SUIT_RATIO) {
        return 0;
    }
    
    // Check target suit if specified
    if (TARGET_SUIT >= 0 && max_suit_idx != TARGET_SUIT) {
        return 0;
    }
    
    // Check for glitched seeds (extreme duplicates)
    int max_duplicates = 0;
    for (int i = 0; i < 52; i++) {
        if (card_counts[i] > max_duplicates) {
            max_duplicates = card_counts[i];
        }
    }
    
    // Calculate score
    long score = 0;
    
    // Base score from face cards and suit ratio
    score = face_count * 100 + (long)(suit_ratio * 1000);
    
    // Huge bonus for glitched seeds
    if (max_duplicates >= 40) {
        score += max_duplicates * 10000;
    }
    
    // Bonus for extreme combinations
    if (face_count >= 25 && suit_ratio >= 0.70) {
        score += 5000;
    }
    
    // Special check for the infamous 7LB2WVPK pattern
    if (max_duplicates == 52) {
        return 999999;  // Maximum score for fully glitched seed
    }
    
    return score;
}