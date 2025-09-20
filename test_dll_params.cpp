/*
 * Test program to verify DLL parameter passing
 * Compile with: x86_64-w64-mingw32-g++ -o test_dll_params.exe test_dll_params.cpp
 * Run on Windows to test actual DLL
 */

#include <windows.h>
#include <iostream>
#include <string>

typedef const char* (*BrainstormFunc)(
    const char* seed,
    const char* voucher, 
    const char* pack,
    const char* tag1,
    const char* tag2,
    double souls,
    bool observatory,
    bool perkeo
);

typedef void (*FreeResultFunc)(const char*);

int main() {
    std::cout << "=== DLL Parameter Test ===" << std::endl;
    
    // Load DLL
    HMODULE dll = LoadLibraryA("Immolate.dll");
    if (!dll) {
        std::cout << "ERROR: Failed to load Immolate.dll" << std::endl;
        std::cout << "Make sure this test is run in the same directory as Immolate.dll" << std::endl;
        return 1;
    }
    
    std::cout << "DLL loaded successfully" << std::endl;
    
    // Get function
    BrainstormFunc brainstorm = (BrainstormFunc)GetProcAddress(dll, "brainstorm");
    FreeResultFunc free_result = (FreeResultFunc)GetProcAddress(dll, "free_result");
    
    if (!brainstorm) {
        std::cout << "ERROR: Failed to get brainstorm function" << std::endl;
        FreeLibrary(dll);
        return 1;
    }
    
    std::cout << "Functions loaded successfully\n" << std::endl;
    
    // Test 1: All empty parameters
    std::cout << "Test 1: Empty parameters" << std::endl;
    std::cout << "Calling: brainstorm(\"TESTTEST\", \"\", \"\", \"\", \"\", 0, false, false)" << std::endl;
    
    const char* result1 = brainstorm("TESTTEST", "", "", "", "", 0.0, false, false);
    std::cout << "Result: " << (result1 ? result1 : "NULL") << std::endl;
    if (result1 && free_result) free_result(result1);
    std::cout << std::endl;
    
    // Test 2: Tags only (what the game is doing)
    std::cout << "Test 2: Tags only" << std::endl;
    std::cout << "Calling: brainstorm(\"TESTTEST\", \"\", \"\", \"Double Tag\", \"Investment Tag\", 0, false, false)" << std::endl;
    
    const char* result2 = brainstorm("TESTTEST", "", "", "Double Tag", "Investment Tag", 0.0, false, false);
    std::cout << "Result: " << (result2 ? result2 : "NULL") << std::endl;
    if (result2 && free_result) free_result(result2);
    std::cout << std::endl;
    
    // Test 3: Voucher and pack
    std::cout << "Test 3: Voucher and Pack" << std::endl;
    std::cout << "Calling: brainstorm(\"TESTTEST\", \"Clearance Sale\", \"Spectral Pack\", \"\", \"\", 0, false, false)" << std::endl;
    
    const char* result3 = brainstorm("TESTTEST", "Clearance Sale", "Spectral Pack", "", "", 0.0, false, false);
    std::cout << "Result: " << (result3 ? result3 : "NULL") << std::endl;
    if (result3 && free_result) free_result(result3);
    std::cout << std::endl;
    
    // Test 4: All filters
    std::cout << "Test 4: All filters" << std::endl;
    std::cout << "Calling: brainstorm(\"TESTTEST\", \"Clearance Sale\", \"Spectral Pack\", \"Double Tag\", \"Investment Tag\", 0, false, false)" << std::endl;
    
    const char* result4 = brainstorm("TESTTEST", "Clearance Sale", "Spectral Pack", "Double Tag", "Investment Tag", 0.0, false, false);
    std::cout << "Result: " << (result4 ? result4 : "NULL") << std::endl;
    if (result4 && free_result) free_result(result4);
    std::cout << std::endl;
    
    std::cout << "=== Check gpu_driver.log for detailed parameter logging ===" << std::endl;
    
    FreeLibrary(dll);
    return 0;
}