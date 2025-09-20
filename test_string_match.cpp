#include <iostream>
#include <cstring>
#include "ImmolateCPP/src/items.hpp"

int main() {
    const char* test_strings[] = {
        "Double Tag",
        "Investment Tag",
        "Clearance Sale",
        "Spectral Pack"
    };
    
    for (const char* str : test_strings) {
        Item item = stringToItem(str);
        std::cout << "stringToItem(\"" << str << "\") = " << static_cast<int>(item);
        if (item == Item::RETRY) {
            std::cout << " (RETRY - NOT FOUND!)";
        }
        std::cout << std::endl;
    }
    
    // Also test with std::string
    std::cout << "\nWith std::string:" << std::endl;
    std::string double_tag = "Double Tag";
    Item item = stringToItem(double_tag);
    std::cout << "stringToItem(std::string(\"Double Tag\")) = " << static_cast<int>(item);
    if (item == Item::RETRY) {
        std::cout << " (RETRY - NOT FOUND!)";
    }
    std::cout << std::endl;
    
    return 0;
}