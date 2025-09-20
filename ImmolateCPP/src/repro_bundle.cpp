/*
 * Repro Bundle Generator
 * Creates one-click reproducible bundles for bug reports
 */

#include <windows.h>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <vector>
#include <cstdio>
#include "pool_hash.hpp"

// Get GPU info via CUDA
extern "C" {
    const char* get_gpu_device_name();
    int get_gpu_compute_capability();
    int get_cuda_driver_version();
    int get_cuda_runtime_version();
    double brainstorm_get_throughput();
}

// Get current pool info
extern "C" {
    const char* get_current_pool_json();
    const char* get_current_pool_id();
    uint32_t get_current_pool_version();
}

// Generate repro bundle
extern "C" __declspec(dllexport)
bool brainstorm_generate_repro_bundle(const char* output_path, const char* seeds[], int seed_count) {
    // Create output directory
    std::string bundle_dir = output_path ? output_path : "repro_bundle";
    CreateDirectoryA(bundle_dir.c_str(), NULL);
    
    // Get timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm* tm = std::localtime(&time_t);
    
    std::ostringstream timestamp;
    timestamp << std::put_time(tm, "%Y%m%d_%H%M%S");
    
    // 1. Save pools.json
    {
        std::string pools_path = bundle_dir + "/pools.json";
        std::ofstream pools_file(pools_path);
        if (!pools_file) return false;
        
        const char* pool_json = get_current_pool_json();
        if (pool_json) {
            pools_file << pool_json;
        }
        pools_file.close();
    }
    
    // 2. Save seeds.txt
    {
        std::string seeds_path = bundle_dir + "/seeds.txt";
        std::ofstream seeds_file(seeds_path);
        if (!seeds_file) return false;
        
        for (int i = 0; i < seed_count; i++) {
            if (seeds[i] && strlen(seeds[i]) == 8) {
                seeds_file << seeds[i] << "\n";
            }
        }
        seeds_file.close();
    }
    
    // 3. Save driver_summary.json
    {
        std::string summary_path = bundle_dir + "/driver_summary.json";
        std::ofstream summary_file(summary_path);
        if (!summary_file) return false;
        
        summary_file << "{\n";
        summary_file << "  \"timestamp\": \"" << timestamp.str() << "\",\n";
        summary_file << "  \"mod_version\": \"v1.0.0-GA\",\n";
        summary_file << "  \"dll_checksum\": \"664b8e3c35c13bfe3cc6f1b171762daca3fe27a69521d3eb5de7429c994065b9\",\n";
        
        // GPU info
        const char* gpu_name = get_gpu_device_name();
        if (gpu_name) {
            summary_file << "  \"gpu_device\": \"" << gpu_name << "\",\n";
        }
        summary_file << "  \"compute_capability\": " << get_gpu_compute_capability() << ",\n";
        summary_file << "  \"cuda_driver_version\": " << get_cuda_driver_version() << ",\n";
        summary_file << "  \"cuda_runtime_version\": " << get_cuda_runtime_version() << ",\n";
        
        // Pool info
        const char* pool_id = get_current_pool_id();
        if (pool_id) {
            summary_file << "  \"pool_id\": \"" << pool_id << "\",\n";
        }
        summary_file << "  \"pool_version\": " << get_current_pool_version() << ",\n";
        
        // Performance
        double throughput = brainstorm_get_throughput();
        summary_file << "  \"throughput_seeds_per_sec\": " << throughput << ",\n";
        
        // Build flags
        summary_file << "  \"build_flags\": {\n";
        summary_file << "    \"fast_math\": false,\n";
        summary_file << "    \"fmad\": false,\n";
        summary_file << "    \"prec_div\": true,\n";
        summary_file << "    \"prec_sqrt\": true\n";
        summary_file << "  },\n";
        
        // Windows info
        OSVERSIONINFOEXA osvi;
        ZeroMemory(&osvi, sizeof(OSVERSIONINFOEXA));
        osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEXA);
        if (GetVersionExA((OSVERSIONINFOA*)&osvi)) {
            summary_file << "  \"windows_version\": \"" 
                        << osvi.dwMajorVersion << "."
                        << osvi.dwMinorVersion << "."
                        << osvi.dwBuildNumber << "\",\n";
        }
        
        summary_file << "  \"issue_template\": \"Please describe the issue and attach gpu_driver.log\"\n";
        summary_file << "}\n";
        summary_file.close();
    }
    
    // 4. Copy recent logs (last 200 lines)
    {
        std::string log_src = "C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm\\gpu_driver.log";
        std::string log_dst = bundle_dir + "/gpu_driver_tail.log";
        
        std::ifstream src(log_src);
        if (src) {
            // Read all lines
            std::vector<std::string> lines;
            std::string line;
            while (std::getline(src, line)) {
                lines.push_back(line);
            }
            src.close();
            
            // Write last 200 lines
            std::ofstream dst(log_dst);
            size_t start = lines.size() > 200 ? lines.size() - 200 : 0;
            for (size_t i = start; i < lines.size(); i++) {
                dst << lines[i] << "\n";
            }
            dst.close();
        }
    }
    
    // 5. Create README
    {
        std::string readme_path = bundle_dir + "/README.txt";
        std::ofstream readme(readme_path);
        readme << "Brainstorm Repro Bundle\n";
        readme << "=======================\n\n";
        readme << "Generated: " << timestamp.str() << "\n";
        readme << "Version: v1.0.0-GA\n\n";
        readme << "Contents:\n";
        readme << "- pools.json: Exact pool snapshot at time of issue\n";
        readme << "- seeds.txt: Seeds that exhibited the issue\n";
        readme << "- driver_summary.json: System and GPU configuration\n";
        readme << "- gpu_driver_tail.log: Last 200 lines of GPU driver log\n\n";
        readme << "To reproduce:\n";
        readme << "1. Load pools.json via brainstorm_update_pools()\n";
        readme << "2. Run differential_runner pools.json seeds.txt\n";
        readme << "3. Compare with expected results\n\n";
        readme << "Submit this bundle when reporting issues.\n";
        readme.close();
    }
    
    return true;
}

// Simplified version for single seed
extern "C" __declspec(dllexport)
bool brainstorm_save_repro(const char* seed) {
    const char* seeds[] = { seed };
    return brainstorm_generate_repro_bundle(nullptr, seeds, 1);
}