#include <iostream>
#include "ImmolateCPP/src/items.hpp"

int main() {
    std::cout << "Testing actual enum values from items.hpp:" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Vouchers:" << std::endl;
    std::cout << "  Overstock = " << static_cast<int>(Item::Overstock) << std::endl;
    std::cout << "  Clearance_Sale = " << static_cast<int>(Item::Clearance_Sale) << std::endl;
    std::cout << std::endl;
    
    std::cout << "Packs:" << std::endl;
    std::cout << "  Arcana_Pack = " << static_cast<int>(Item::Arcana_Pack) << std::endl;
    std::cout << "  Buffoon_Pack = " << static_cast<int>(Item::Buffoon_Pack) << std::endl;
    std::cout << "  Spectral_Pack = " << static_cast<int>(Item::Spectral_Pack) << std::endl;
    std::cout << std::endl;
    
    std::cout << "Tags:" << std::endl;
    std::cout << "  TAG_BEGIN = " << static_cast<int>(Item::TAG_BEGIN) << std::endl;
    std::cout << "  Uncommon_Tag = " << static_cast<int>(Item::Uncommon_Tag) << std::endl;
    std::cout << "  Investment_Tag = " << static_cast<int>(Item::Investment_Tag) << std::endl;
    std::cout << "  Charm_Tag = " << static_cast<int>(Item::Charm_Tag) << std::endl;
    std::cout << "  Double_Tag = " << static_cast<int>(Item::Double_Tag) << std::endl;
    std::cout << "  TAG_END = " << static_cast<int>(Item::TAG_END) << std::endl;
    std::cout << std::endl;
    
    std::cout << "Correct offsets should be:" << std::endl;
    std::cout << "  Voucher base (Overstock): " << static_cast<int>(Item::Overstock) << std::endl;
    std::cout << "  Pack base (Arcana_Pack): " << static_cast<int>(Item::Arcana_Pack) << std::endl;
    std::cout << "  Tag base (first tag after TAG_BEGIN): " << static_cast<int>(Item::Uncommon_Tag) << std::endl;
    
    return 0;
}