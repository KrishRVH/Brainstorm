#include <iostream>
#include <iomanip>
#include "ImmolateCPP/src/balatro_correct.hpp"

int main() {
    std::cout << std::fixed << std::setprecision(17);
    
    std::cout << "=== Testing seed 7NTPKW6P ===\n\n";
    
    // This is the seed from your screenshot
    BalatroRNG rng("7NTPKW6P");
    
    std::cout << "Seed hash: " << rng.get_hashed() << "\n";
    std::cout << "Expected:  " << pseudohash("7NTPKW6P") << "\n\n";
    
    // Generate shop contents
    std::cout << "Shop generation:\n\n";
    
    // Voucher
    double voucher_val = rng.pseudoseed("Voucher");
    int voucher_index = static_cast<int>(voucher_val * 32.0);
    std::cout << "Voucher:\n";
    std::cout << "  Raw value: " << voucher_val << "\n";
    std::cout << "  Index (0-31): " << voucher_index << "\n";
    
    // Map to actual voucher names (partial list)
    const char* voucher_names[] = {
        "Overstock", "Clearance Sale", "Hone", "Reroll Surplus",
        "Crystal Ball", "Telescope", "Grabber", "Wasteful",
        "Tarot Merchant", "Planet Merchant", "Seed Money", "Blank",
        "Magic Trick", "Hieroglyph", "Directors Cut", "Retcon",
        "Paint Brush", "Overstock Plus", "Liquidation", "Glow Up",
        "Reroll Glut", "Omen Globe", "Observatory", "Nacho Tong",
        "Recyclomancy", "Money Tree", "Antimatter", "Illusion",
        "Petroglyph", "Curator", "Unknown", "Unknown"
    };
    
    if (voucher_index < 32) {
        std::cout << "  Voucher: " << voucher_names[voucher_index] << "\n";
    }
    std::cout << "\n";
    
    // Packs
    double pack1_val = rng.pseudoseed("shop_pack1");
    double pack2_val = rng.pseudoseed("shop_pack1");
    
    int pack1_index = static_cast<int>(pack1_val * 15.0);
    int pack2_index = static_cast<int>(pack2_val * 15.0);
    
    const char* pack_names[] = {
        "Arcana Pack", "Jumbo Arcana Pack", "Mega Arcana Pack",
        "Celestial Pack", "Jumbo Celestial Pack", "Mega Celestial Pack",
        "Spectral Pack", "Jumbo Spectral Pack", "Mega Spectral Pack",
        "Standard Pack", "Jumbo Standard Pack", "Mega Standard Pack",
        "Buffoon Pack", "Jumbo Buffoon Pack", "Mega Buffoon Pack"
    };
    
    std::cout << "Pack 1:\n";
    std::cout << "  Raw value: " << pack1_val << "\n";
    std::cout << "  Index (0-14): " << pack1_index << "\n";
    if (pack1_index < 15) {
        std::cout << "  Pack: " << pack_names[pack1_index] << "\n";
    }
    std::cout << "\n";
    
    std::cout << "Pack 2:\n";
    std::cout << "  Raw value: " << pack2_val << "\n";
    std::cout << "  Index (0-14): " << pack2_index << "\n";
    if (pack2_index < 15) {
        std::cout << "  Pack: " << pack_names[pack2_index] << "\n";
    }
    std::cout << "\n";
    
    // Tags
    double tag_small_val = rng.pseudoseed("Tag_small");
    double tag_big_val = rng.pseudoseed("Tag_big");
    
    int tag_small_index = static_cast<int>(tag_small_val * 24.0);
    int tag_big_index = static_cast<int>(tag_big_val * 24.0);
    
    const char* tag_names[] = {
        "Uncommon Tag", "Rare Tag", "Negative Tag", "Foil Tag",
        "Holographic Tag", "Polychrome Tag", "Investment Tag", "Voucher Tag",
        "Boss Tag", "Standard Tag", "Charm Tag", "Meteor Tag",
        "Buffoon Tag", "Handy Tag", "Garbage Tag", "Ethereal Tag",
        "Coupon Tag", "Double Tag", "Juggle Tag", "D6 Tag",
        "Top-up Tag", "Speed Tag", "Orbital Tag", "Economy Tag"
    };
    
    std::cout << "Tags:\n";
    std::cout << "  Small tag index: " << tag_small_index << "\n";
    if (tag_small_index < 24) {
        std::cout << "  Small tag: " << tag_names[tag_small_index] << "\n";
    }
    std::cout << "  Big tag index: " << tag_big_index << "\n";
    if (tag_big_index < 24) {
        std::cout << "  Big tag: " << tag_names[tag_big_index] << "\n";
    }
    
    std::cout << "\n=== First Shop Override ===\n";
    std::cout << "Note: Balatro forces a Buffoon Pack in the first shop\n";
    std::cout << "So one pack will always be Buffoon Pack regardless of RNG\n";
    
    return 0;
}