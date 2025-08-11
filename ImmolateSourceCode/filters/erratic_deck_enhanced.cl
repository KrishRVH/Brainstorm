// Enhanced Erratic Deck filter with face card and suit ratio support
// Based on our reverse-engineering of Balatro's Erratic deck generation

#include "lib/immolate.cl"

// Initialize Erratic deck using the exact algorithm from Balatro
void init_erratic_deck(instance* inst, item deck[52]) {
    // Set deck type to Erratic
    inst->params.deck = Erratic_Deck;
    
    // For Erratic deck, each card position randomly selects from all 52 cards
    // This matches: if self.GAME.starting_params.erratic_suits_and_ranks then 
    //               _, k = pseudorandom_element(G.P_CARDS, pseudoseed('erratic'))
    
    for (int i = 0; i < 52; i++) {
        // Each card gets its own RNG call with 'erratic' key
        // The node system will maintain state across calls
        item card = randchoice(inst, 
            (__private ntype[]){N_Type, N_CardIndex}, 
            (__private int[]){R_Erratic, i}, 
            2, 
            CARDS);
        deck[i] = card;
    }
}

// Count face cards in deck (J, Q, K)
int count_face_cards(item deck[52]) {
    int count = 0;
    for (int i = 0; i < 52; i++) {
        item card = deck[i];
        // Face cards are J, Q, K of any suit
        if (card == C_J || card == D_J || card == H_J || card == S_J ||
            card == C_Q || card == D_Q || card == H_Q || card == S_Q ||
            card == C_K || card == D_K || card == H_K || card == S_K) {
            count++;
        }
    }
    return count;
}

// Calculate suit distribution
void get_suit_counts(item deck[52], int suit_counts[4]) {
    for (int i = 0; i < 4; i++) suit_counts[i] = 0;
    
    for (int i = 0; i < 52; i++) {
        item card = deck[i];
        // Determine suit (0=Clubs, 1=Diamonds, 2=Hearts, 3=Spades)
        int suit = -1;
        if (card >= C_2 && card <= C_A) suit = 0;
        else if (card >= D_2 && card <= D_A) suit = 1;
        else if (card >= H_2 && card <= H_A) suit = 2;
        else if (card >= S_2 && card <= S_A) suit = 3;
        
        if (suit >= 0) suit_counts[suit]++;
    }
}

// Main filter function - returns score based on criteria
long filter(instance* inst) {
    // Configuration - adjust these for different searches
    const int MIN_FACE_CARDS = 20;  // Minimum face cards required
    const float MIN_SUIT_RATIO = 0.5;  // Minimum ratio for dominant suit (50%)
    
    set_deck(inst, Erratic_Deck);
    item deck[52];
    init_erratic_deck(inst, deck);
    
    // Count face cards
    int face_count = count_face_cards(deck);
    if (face_count < MIN_FACE_CARDS) return 0;
    
    // Calculate suit distribution
    int suit_counts[4];
    get_suit_counts(deck, suit_counts);
    
    // Find maximum suit ratio
    int max_suit = suit_counts[0];
    for (int i = 1; i < 4; i++) {
        if (suit_counts[i] > max_suit) max_suit = suit_counts[i];
    }
    float max_ratio = (float)max_suit / 52.0f;
    
    if (max_ratio < MIN_SUIT_RATIO) return 0;
    
    // Score based on how good the seed is
    // Higher face count and suit ratio = higher score
    long score = face_count * 100 + (long)(max_ratio * 1000);
    
    // Bonus for extremely rare combinations
    if (face_count >= 25 && max_ratio >= 0.7) score += 10000;
    
    // Check for glitched seeds (like 52 of same card)
    int card_counts[52];
    for (int i = 0; i < 52; i++) card_counts[i] = 0;
    for (int i = 0; i < 52; i++) {
        card_counts[deck[i] - C_2]++;
    }
    for (int i = 0; i < 52; i++) {
        if (card_counts[i] >= 52) {
            // Found a glitched seed!
            return 999999;
        }
    }
    
    return score;
}

// Alternative filter for finding specific patterns
long filter_glitched_only(instance* inst) {
    set_deck(inst, Erratic_Deck);
    item deck[52];
    init_erratic_deck(inst, deck);
    
    // Count occurrences of each card
    int card_counts[52];
    for (int i = 0; i < 52; i++) card_counts[i] = 0;
    for (int i = 0; i < 52; i++) {
        card_counts[deck[i] - C_2]++;
    }
    
    // Find maximum duplicates
    int max_duplicates = 0;
    for (int i = 0; i < 52; i++) {
        if (card_counts[i] > max_duplicates) {
            max_duplicates = card_counts[i];
        }
    }
    
    // Only return seeds with extreme duplication (glitched seeds)
    if (max_duplicates >= 40) {
        return max_duplicates;
    }
    
    return 0;
}