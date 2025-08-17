// Brainstorm Unified DLL - CPU + GPU acceleration with runtime detection
// Provides transparent GPU acceleration when available, CPU fallback otherwise

#include "functions.hpp"
#include "search.hpp"
#include <cstring>
#include <cstdlib>
#include <string>
#include <iostream>

// Check for CUDA at compile time
#ifdef GPU_ENABLED
#ifdef GPU_DYNAMIC_LOAD
#include "gpu/cuda_wrapper.hpp"
#else
#include <cuda_runtime.h>
#endif
#include "gpu/gpu_searcher.hpp"
#endif

// Define IMMOLATE_API for DLL export
#ifndef IMMOLATE_API
#ifdef _WIN32
#ifdef BUILDING_DLL
#define IMMOLATE_API __declspec(dllexport)
#else
#define IMMOLATE_API __declspec(dllimport)
#endif
#else
#define IMMOLATE_API
#endif
#endif

// Global filter settings (shared between CPU and GPU)
Item BRAINSTORM_VOUCHER = Item::RETRY;
Item BRAINSTORM_PACK = Item::RETRY;
Item BRAINSTORM_TAG1 = Item::RETRY;
Item BRAINSTORM_TAG2 = Item::RETRY;
long BRAINSTORM_SOULS = 0;
bool BRAINSTORM_OBSERVATORY = false;
bool BRAINSTORM_PERKEO = false;

// GPU state management
static bool g_cuda_available = false;
static bool g_use_cuda = true;  // User preference
static int g_device_count = 0;
static std::string g_hardware_info = "CPU: No GPU acceleration";

#ifdef GPU_ENABLED
static GPUSearcher* g_gpu_searcher = nullptr;
#endif

// Initialize GPU support if available
static void initialize_gpu() {
#ifdef GPU_ENABLED
    if (g_gpu_searcher) return;  // Already initialized
    
#ifdef GPU_DYNAMIC_LOAD
    // Use dynamic CUDA loader
    extern CudaWrapper g_cuda;
    if (!g_cuda.init()) {
        g_cuda_available = false;
        g_hardware_info = "CUDA runtime not found";
        std::cerr << "[GPU] CUDA runtime DLL not found" << std::endl;
        return;
    }
    
    cudaError_t err = g_cuda.cudaGetDeviceCount(&g_device_count);
    if (err == cudaSuccess && g_device_count > 0) {
        cudaDeviceProp props;
        err = g_cuda.cudaGetDeviceProperties(&props, 0);
        
        if (err == cudaSuccess && props.major >= 6) {
            g_cuda_available = true;
            g_hardware_info = std::string("GPU: ") + props.name + 
                             " (" + std::to_string(props.multiProcessorCount) + 
                             " SMs, " + std::to_string(props.totalGlobalMem / (1024*1024)) + " MB)";
#else
    // Use static CUDA linking
    cudaError_t err = cudaGetDeviceCount(&g_device_count);
    if (err == cudaSuccess && g_device_count > 0) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, 0);
        
        // Check for capable GPU (Compute 6.0+ for decent performance)
        if (props.major >= 6) {
            g_cuda_available = true;
            g_hardware_info = std::string("GPU: ") + props.name + 
                             " (" + std::to_string(props.multiProcessorCount) + 
                             " SMs, " + std::to_string(props.totalGlobalMem / (1024*1024)) + " MB)";
#endif
            
            // Create GPU searcher instance
            g_gpu_searcher = new GPUSearcher();
            
            std::cout << "[Brainstorm] " << g_hardware_info << std::endl;
        } else {
            g_hardware_info = std::string("GPU: ") + props.name + " (too old, needs Compute 6.0+)";
            std::cout << "[Brainstorm] " << g_hardware_info << std::endl;
        }
    } else {
        g_hardware_info = "CPU: No CUDA-capable GPU found";
    }
#else
    g_hardware_info = "CPU: Compiled without GPU support";
#endif
}

// CPU implementation - existing filter logic
long filter_cpu(Instance inst) {
    // Check tags first (cheapest operation)
    if (BRAINSTORM_TAG1 != Item::RETRY || BRAINSTORM_TAG2 != Item::RETRY) {
        Item smallBlindTag = inst.nextTag(1);
        Item bigBlindTag = inst.nextTag(1);
        
        if (BRAINSTORM_TAG2 == Item::RETRY) {
            if (smallBlindTag != BRAINSTORM_TAG1 && bigBlindTag != BRAINSTORM_TAG1) {
                return 0;
            }
        }
        else if (BRAINSTORM_TAG1 != BRAINSTORM_TAG2) {
            bool hasTag1 = (smallBlindTag == BRAINSTORM_TAG1 || bigBlindTag == BRAINSTORM_TAG1);
            bool hasTag2 = (smallBlindTag == BRAINSTORM_TAG2 || bigBlindTag == BRAINSTORM_TAG2);
            if (!hasTag1 || !hasTag2) {
                return 0;
            }
        }
        else {
            if (smallBlindTag != BRAINSTORM_TAG1 || bigBlindTag != BRAINSTORM_TAG1) {
                return 0;
            }
        }
    }
    
    // Check voucher if specified
    if (BRAINSTORM_VOUCHER != Item::RETRY) {
        inst.initLocks(1, false, false);
        Item firstVoucher = inst.nextVoucher(1);
        if (firstVoucher != BRAINSTORM_VOUCHER) {
            return 0;
        }
    }
    
    // Check pack if specified
    if (BRAINSTORM_PACK != Item::RETRY) {
        inst.cache.generatedFirstPack = true;
        if (inst.nextPack(1) != BRAINSTORM_PACK) {
            return 0;
        }
    }
    
    // Check special conditions
    if (BRAINSTORM_OBSERVATORY) {
        inst.initLocks(1, false, false);
        Item firstVoucher = inst.nextVoucher(1);
        if (firstVoucher != Item::Telescope) {
            return 0;
        }
        inst.cache.generatedFirstPack = true;
        Item pack = inst.nextPack(1);
        if (pack != Item::Mega_Celestial_Pack) {
            return 0;
        }
    }
    
    if (BRAINSTORM_PERKEO) {
        Item smallBlindTag = inst.nextTag(1);
        Item bigBlindTag = inst.nextTag(1);
        
        if (smallBlindTag != Item::Investment_Tag && bigBlindTag != Item::Investment_Tag) {
            return 0;
        }
        
        auto tarots = inst.nextArcanaPack(5, 1);
        bool found_soul = false;
        
        for (int t = 0; t < 5; t++) {
            if (tarots[t] == Item::The_Soul) {
                found_soul = true;
                break;
            }
        }
        
        if (!found_soul) {
            return 0;
        }
    }
    
    if (BRAINSTORM_SOULS > 0) {
        for (int i = 1; i <= BRAINSTORM_SOULS; i++) {
            auto tarots = inst.nextArcanaPack(5, 1);
            bool found_soul = false;
            for (int t = 0; t < 5; t++) {
                if (tarots[t] == Item::The_Soul) {
                    found_soul = true;
                    break;
                }
            }
            if (!found_soul) {
                return 0;
            }
        }
    }
    
    return 1;
}

// Main search function - dispatches to GPU or CPU
std::string brainstorm_internal(
    std::string seed, 
    std::string voucher,
    std::string pack, 
    std::string tag1,
    std::string tag2,
    double souls,
    bool observatory,
    bool perkeo
) {
    // Set global filters
    BRAINSTORM_VOUCHER = stringToItem(voucher);
    BRAINSTORM_PACK = stringToItem(pack);
    BRAINSTORM_TAG1 = stringToItem(tag1);
    BRAINSTORM_TAG2 = stringToItem(tag2);
    BRAINSTORM_SOULS = souls;
    BRAINSTORM_OBSERVATORY = observatory;
    BRAINSTORM_PERKEO = perkeo;
    
    // Decide whether to use GPU or CPU
    bool use_gpu = g_cuda_available && g_use_cuda;
    
#ifdef GPU_ENABLED
    if (use_gpu && g_gpu_searcher) {
        // Use GPU acceleration
        FilterParams params;
        params.tag1 = (BRAINSTORM_TAG1 != Item::RETRY) ? static_cast<uint32_t>(BRAINSTORM_TAG1) : 0xFFFFFFFF;
        params.tag2 = (BRAINSTORM_TAG2 != Item::RETRY) ? static_cast<uint32_t>(BRAINSTORM_TAG2) : 0xFFFFFFFF;
        params.voucher = (BRAINSTORM_VOUCHER != Item::RETRY) ? static_cast<uint32_t>(BRAINSTORM_VOUCHER) : 0xFFFFFFFF;
        params.pack = (BRAINSTORM_PACK != Item::RETRY) ? static_cast<uint32_t>(BRAINSTORM_PACK) : 0xFFFFFFFF;
        params.require_souls = (BRAINSTORM_SOULS > 0);
        params.require_observatory = BRAINSTORM_OBSERVATORY;
        params.require_perkeo = BRAINSTORM_PERKEO;
        
        std::string result = g_gpu_searcher->search(seed, params);
        if (!result.empty()) {
            return result;
        }
    }
#endif
    
    // Fall back to CPU implementation
    Search search(filter_cpu, seed, 1, 100000000);
    search.exitOnFind = true;
    return search.search();
}

// C interface for DLL export
extern "C" {
    // Main search function with dual tag support
    IMMOLATE_API const char* brainstorm(
        const char* seed,
        const char* voucher,
        const char* pack,
        const char* tag1,
        const char* tag2,
        double souls,
        bool observatory,
        bool perkeo
    ) {
        // Initialize GPU on first call
        static bool initialized = false;
        if (!initialized) {
            initialize_gpu();
            initialized = true;
        }
        
        std::string cpp_seed(seed);
        std::string cpp_voucher(voucher);
        std::string cpp_pack(pack);
        std::string cpp_tag1(tag1);
        std::string cpp_tag2(tag2);
        
        std::string result = brainstorm_internal(
            cpp_seed, cpp_voucher, cpp_pack, 
            cpp_tag1, cpp_tag2, souls, 
            observatory, perkeo
        );
        
        return strdup(result.c_str());
    }
    
    // Get tags for a specific seed
    IMMOLATE_API const char* get_tags(const char* seed) {
        std::string cpp_seed(seed);
        Seed s(cpp_seed);
        Instance inst(s);
        Item smallBlindTag = inst.nextTag(1);
        Item bigBlindTag = inst.nextTag(1);
        
        std::string formatted = itemToString(smallBlindTag) + "|" + itemToString(bigBlindTag);
        return strdup(formatted.c_str());
    }
    
    // Free memory allocated by DLL
    IMMOLATE_API void free_result(const char* result) {
        free((void*)result);
    }
    
    // GPU/CUDA control functions
    IMMOLATE_API int get_acceleration_type() {
        // 0 = CPU, 1 = GPU
        return (g_cuda_available && g_use_cuda) ? 1 : 0;
    }
    
    IMMOLATE_API const char* get_hardware_info() {
        return g_hardware_info.c_str();
    }
    
    IMMOLATE_API void set_use_cuda(bool enable) {
        g_use_cuda = enable;
        
        // Re-initialize if enabling and not initialized
        if (enable && !g_cuda_available) {
            initialize_gpu();
        }
        
        std::cout << "[Brainstorm] CUDA " << (enable ? "enabled" : "disabled") 
                  << " by user preference" << std::endl;
    }
}