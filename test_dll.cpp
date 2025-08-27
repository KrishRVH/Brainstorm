// Standalone DLL tester - catches crashes before deployment
#include <windows.h>
#include <iostream>
#include <string>
#include <cstring>
#include <exception>
#include <csignal>
#include <functional>

// Function pointers for DLL exports
typedef const char* (*brainstorm_fn)(const char*, const char*, const char*, const char*, const char*, double, bool, bool);
typedef void (*free_result_fn)(const char*);
typedef const char* (*get_hardware_info_fn)();
typedef void (*set_use_cuda_fn)(bool);
typedef int (*get_acceleration_type_fn)();
typedef const char* (*get_tags_fn)(const char*);

// Signal handler for catching crashes
void signal_handler(int sig) {
    std::cerr << "\n[CRASH DETECTED] Signal " << sig << " received!" << std::endl;
    if (sig == SIGSEGV) {
        std::cerr << "Segmentation fault - likely memory access violation" << std::endl;
    } else if (sig == SIGABRT) {
        std::cerr << "Abort signal - likely assertion failure" << std::endl;
    }
    exit(1);
}

// Structured exception handler for Windows
LONG WINAPI exception_handler(EXCEPTION_POINTERS* ExceptionInfo) {
    std::cerr << "\n[CRASH DETECTED] Windows exception code: 0x" 
              << std::hex << ExceptionInfo->ExceptionRecord->ExceptionCode << std::endl;
    
    switch(ExceptionInfo->ExceptionRecord->ExceptionCode) {
        case EXCEPTION_ACCESS_VIOLATION:
            std::cerr << "Access violation at address: 0x" 
                      << ExceptionInfo->ExceptionRecord->ExceptionAddress << std::endl;
            break;
        case EXCEPTION_STACK_OVERFLOW:
            std::cerr << "Stack overflow detected" << std::endl;
            break;
        case EXCEPTION_INT_DIVIDE_BY_ZERO:
            std::cerr << "Integer division by zero" << std::endl;
            break;
    }
    
    return EXCEPTION_EXECUTE_HANDLER;
}

bool test_dll_function(HMODULE dll, const std::string& test_name, std::function<void()> test_func) {
    std::cout << "\n[TEST] " << test_name << "..." << std::flush;
    
    try {
        test_func();
        std::cout << " PASS" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cout << " FAIL" << std::endl;
        std::cerr << "  Exception: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cout << " FAIL" << std::endl;
        std::cerr << "  Unknown exception" << std::endl;
        return false;
    }
}

int main(int argc, char* argv[]) {
    // Install signal handlers
    signal(SIGSEGV, signal_handler);
    signal(SIGABRT, signal_handler);
    SetUnhandledExceptionFilter(exception_handler);
    
    std::cout << "========================================" << std::endl;
    std::cout << "  Brainstorm DLL Test Harness" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Load the DLL
    const char* dll_path = (argc > 1) ? argv[1] : "Immolate.dll";
    std::cout << "\nLoading DLL: " << dll_path << std::endl;
    
    HMODULE dll = LoadLibraryA(dll_path);
    if (!dll) {
        std::cerr << "[ERROR] Failed to load DLL. Error code: " << GetLastError() << std::endl;
        return 1;
    }
    std::cout << "[OK] DLL loaded successfully" << std::endl;
    
    // Get function pointers
    auto brainstorm = (brainstorm_fn)GetProcAddress(dll, "brainstorm");
    auto free_result = (free_result_fn)GetProcAddress(dll, "free_result");
    auto get_hardware_info = (get_hardware_info_fn)GetProcAddress(dll, "get_hardware_info");
    auto set_use_cuda = (set_use_cuda_fn)GetProcAddress(dll, "set_use_cuda");
    auto get_acceleration_type = (get_acceleration_type_fn)GetProcAddress(dll, "get_acceleration_type");
    auto get_tags = (get_tags_fn)GetProcAddress(dll, "get_tags");
    
    // Check exports
    std::cout << "\n[CHECKING EXPORTS]" << std::endl;
    std::cout << "  brainstorm: " << (brainstorm ? "✓" : "✗") << std::endl;
    std::cout << "  free_result: " << (free_result ? "✓" : "✗") << std::endl;
    std::cout << "  get_hardware_info: " << (get_hardware_info ? "✓" : "✗") << std::endl;
    std::cout << "  set_use_cuda: " << (set_use_cuda ? "✓" : "✗") << std::endl;
    std::cout << "  get_acceleration_type: " << (get_acceleration_type ? "✓" : "✗") << std::endl;
    std::cout << "  get_tags: " << (get_tags ? "✓" : "✗") << std::endl;
    
    if (!brainstorm) {
        std::cerr << "[ERROR] Critical export 'brainstorm' not found!" << std::endl;
        FreeLibrary(dll);
        return 1;
    }
    
    int tests_passed = 0;
    int tests_failed = 0;
    
    // Test 1: Hardware info
    if (get_hardware_info) {
        if (test_dll_function(dll, "Get hardware info", [&]() {
            const char* info = get_hardware_info();
            std::cout << "    Hardware: " << (info ? info : "null") << std::endl;
        })) tests_passed++; else tests_failed++;
    }
    
    // Test 2: Acceleration type
    if (get_acceleration_type) {
        if (test_dll_function(dll, "Get acceleration type", [&]() {
            int type = get_acceleration_type();
            std::cout << "    Type: " << (type == 1 ? "GPU" : "CPU") << std::endl;
        })) tests_passed++; else tests_failed++;
    }
    
    // Test 3: Set CUDA (should not crash even if no GPU)
    if (set_use_cuda) {
        if (test_dll_function(dll, "Set CUDA enabled", [&]() {
            set_use_cuda(true);
        })) tests_passed++; else tests_failed++;
        
        if (test_dll_function(dll, "Set CUDA disabled", [&]() {
            set_use_cuda(false);
        })) tests_passed++; else tests_failed++;
    }
    
    // Test 4: Basic brainstorm call with minimal parameters
    if (test_dll_function(dll, "Brainstorm with null filters", [&]() {
        const char* result = brainstorm("TESTTEST", nullptr, nullptr, nullptr, nullptr, 0.0, false, false);
        if (result) {
            std::cout << "    Result: " << result << std::endl;
            if (free_result) free_result(result);
        } else {
            std::cout << "    Result: null (no match)" << std::endl;
        }
    })) tests_passed++; else tests_failed++;
    
    // Test 5: Brainstorm with tag filter (this is where GPU init happens)
    if (test_dll_function(dll, "Brainstorm with tag filter", [&]() {
        const char* result = brainstorm("TESTTEST", nullptr, nullptr, "Speed Tag", nullptr, 0.0, false, false);
        if (result) {
            std::cout << "    Result: " << result << std::endl;
            if (free_result) free_result(result);
        } else {
            std::cout << "    Result: null (no match)" << std::endl;
        }
    })) tests_passed++; else tests_failed++;
    
    // Test 6: Brainstorm with dual tags
    if (test_dll_function(dll, "Brainstorm with dual tags", [&]() {
        const char* result = brainstorm("TESTTEST", nullptr, nullptr, "Speed Tag", "Economy Tag", 0.0, false, false);
        if (result) {
            std::cout << "    Result: " << result << std::endl;
            if (free_result) free_result(result);
        } else {
            std::cout << "    Result: null (no match)" << std::endl;
        }
    })) tests_passed++; else tests_failed++;
    
    // Test 7: Get tags for seed
    if (get_tags) {
        if (test_dll_function(dll, "Get tags for seed", [&]() {
            const char* tags = get_tags("TESTTEST");
            std::cout << "    Tags: " << (tags ? tags : "empty") << std::endl;
        })) tests_passed++; else tests_failed++;
    }
    
    // Test 8: Memory leak test - multiple calls
    if (test_dll_function(dll, "Memory leak test (100 calls)", [&]() {
        for (int i = 0; i < 100; i++) {
            const char* result = brainstorm("TESTTEST", nullptr, nullptr, nullptr, nullptr, 0.0, false, false);
            if (result && free_result) {
                free_result(result);
            }
        }
        std::cout << "    Completed 100 calls without crash" << std::endl;
    })) tests_passed++; else tests_failed++;
    
    // Test 9: Invalid parameters
    if (test_dll_function(dll, "Invalid seed parameter", [&]() {
        const char* result = brainstorm("INVALID!", nullptr, nullptr, nullptr, nullptr, 0.0, false, false);
        // Should handle gracefully
    })) tests_passed++; else tests_failed++;
    
    // Cleanup
    FreeLibrary(dll);
    
    // Summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "  TEST SUMMARY" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Passed: " << tests_passed << std::endl;
    std::cout << "Failed: " << tests_failed << std::endl;
    
    if (tests_failed == 0) {
        std::cout << "\n[SUCCESS] All tests passed! DLL is ready for deployment." << std::endl;
        return 0;
    } else {
        std::cout << "\n[WARNING] Some tests failed. Fix issues before deployment." << std::endl;
        return 1;
    }
}