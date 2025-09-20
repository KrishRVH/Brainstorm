/*
 * Debug Configuration and Logging System
 * Provides comprehensive debugging with runtime toggle
 */

#ifndef DEBUG_HPP
#define DEBUG_HPP

#include <cstdio>
#include <cstdarg>
#include <string>
#include <chrono>
#include <mutex>

// ============================================================================
// DEBUG CONFIGURATION
// ============================================================================

class DebugSystem {
private:
    static bool debug_enabled;
    static FILE* log_file;
    static std::mutex log_mutex;
    static std::chrono::steady_clock::time_point start_time;
    
public:
    // Initialize debug system
    static void init(bool enable_debug = false) {
        debug_enabled = enable_debug;
        
        if (debug_enabled) {
            std::lock_guard<std::mutex> lock(log_mutex);
            const char* log_path = "C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm\\debug_full.log";
            log_file = fopen(log_path, "w");  // Clear log on init
            if (log_file) {
                fprintf(log_file, "=== DEBUG SESSION STARTED ===\n");
                fprintf(log_file, "Debug system initialized with full instrumentation\n\n");
                fflush(log_file);
            }
            start_time = std::chrono::steady_clock::now();
        }
    }
    
    // Check if debug is enabled
    static bool is_enabled() {
        return debug_enabled;
    }
    
    // Log with timestamp and module
    static void log(const char* module, const char* format, ...) {
        if (!debug_enabled || !log_file) return;
        
        std::lock_guard<std::mutex> lock(log_mutex);
        
        // Get elapsed time
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
        
        // Print timestamp and module
        fprintf(log_file, "[%8lld ms] [%-12s] ", elapsed.count(), module);
        
        // Print formatted message
        va_list args;
        va_start(args, format);
        vfprintf(log_file, format, args);
        va_end(args);
        
        fprintf(log_file, "\n");
        fflush(log_file);
    }
    
    // Log hex dump for binary data
    static void log_hex(const char* module, const char* label, const void* data, size_t size) {
        if (!debug_enabled || !log_file) return;
        
        std::lock_guard<std::mutex> lock(log_mutex);
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
        
        fprintf(log_file, "[%8lld ms] [%-12s] %s (%zu bytes):\n", 
                elapsed.count(), module, label, size);
        
        const uint8_t* bytes = (const uint8_t*)data;
        for (size_t i = 0; i < size; i += 16) {
            fprintf(log_file, "  %04zx: ", i);
            
            // Hex bytes
            for (size_t j = 0; j < 16 && i + j < size; j++) {
                fprintf(log_file, "%02x ", bytes[i + j]);
            }
            
            // ASCII representation
            fprintf(log_file, " |");
            for (size_t j = 0; j < 16 && i + j < size; j++) {
                uint8_t c = bytes[i + j];
                fprintf(log_file, "%c", (c >= 32 && c < 127) ? c : '.');
            }
            fprintf(log_file, "|\n");
        }
        fflush(log_file);
    }
    
    // Assert with detailed info
    static void assert_equal(const char* module, const char* name, 
                            uint32_t expected, uint32_t actual) {
        if (!debug_enabled) return;
        
        if (expected != actual) {
            log(module, "ASSERTION FAILED: %s - expected %u, got %u", 
                name, expected, actual);
        } else {
            log(module, "Assertion passed: %s = %u", name, actual);
        }
    }
    
    // Cleanup
    static void shutdown() {
        if (log_file) {
            fprintf(log_file, "\n=== DEBUG SESSION ENDED ===\n");
            fclose(log_file);
            log_file = nullptr;
        }
    }
};

// Static member initialization
inline bool DebugSystem::debug_enabled = false;
inline FILE* DebugSystem::log_file = nullptr;
inline std::mutex DebugSystem::log_mutex;
inline std::chrono::steady_clock::time_point DebugSystem::start_time;

// ============================================================================
// CONVENIENCE MACROS
// ============================================================================

#define DEBUG_LOG(module, ...) DebugSystem::log(module, __VA_ARGS__)
#define DEBUG_HEX(module, label, data, size) DebugSystem::log_hex(module, label, data, size)
#define DEBUG_ASSERT(module, name, expected, actual) DebugSystem::assert_equal(module, name, expected, actual)

// Performance timer for debug mode
class DebugTimer {
    const char* module;
    const char* operation;
    std::chrono::steady_clock::time_point start;
    
public:
    DebugTimer(const char* mod, const char* op) : module(mod), operation(op) {
        if (DebugSystem::is_enabled()) {
            start = std::chrono::steady_clock::now();
            DEBUG_LOG(module, "Starting: %s", operation);
        }
    }
    
    ~DebugTimer() {
        if (DebugSystem::is_enabled()) {
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start
            );
            DEBUG_LOG(module, "Completed: %s (took %lld us)", operation, elapsed.count());
        }
    }
};

#endif // DEBUG_HPP