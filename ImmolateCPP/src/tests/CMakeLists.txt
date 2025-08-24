# CMakeLists.txt for Brainstorm ImmolateCPP unit tests

cmake_minimum_required(VERSION 3.10)
project(BrainstormTests CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable testing
enable_testing()

# Find required packages
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/../)
include_directories(${GTEST_INCLUDE_DIRS})

# Compiler flags for testing
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

# Add debug flags for memory safety testing
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -DDEBUG")

# AddressSanitizer flags (enable with -DUSE_ASAN=ON)
option(USE_ASAN "Use AddressSanitizer" OFF)
if(USE_ASAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")
endif()

# ThreadSanitizer flags (enable with -DUSE_TSAN=ON)  
option(USE_TSAN "Use ThreadSanitizer" OFF)
if(USE_TSAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=thread")
endif()

# Define source files from parent directory
set(BRAINSTORM_SOURCES
    ../util.cpp
    ../seed.cpp
    ../rng.cpp
    ../items.cpp
    ../functions.cpp
    # Add other source files as needed
)

# Critical functions test executable
add_executable(test_critical_functions 
    test_critical_functions.cpp
    ${BRAINSTORM_SOURCES}
)

target_link_libraries(test_critical_functions 
    ${GTEST_LIBRARIES}
    ${GTEST_MAIN_LIBRARIES}
    Threads::Threads
)

# Memory safety test executable
add_executable(test_memory_safety
    test_memory_safety.cpp
    ${BRAINSTORM_SOURCES}
)

target_link_libraries(test_memory_safety
    ${GTEST_LIBRARIES} 
    ${GTEST_MAIN_LIBRARIES}
    Threads::Threads
)

# GPU tests (only if CUDA is available)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    option(BUILD_GPU_TESTS "Build GPU/CUDA tests" ON)
    
    if(BUILD_GPU_TESTS)
        # GPU searcher test
        add_executable(test_gpu_searcher
            test_gpu_searcher.cpp
            ../gpu/gpu_searcher.cpp
            ../gpu/gpu_searcher_dynamic.cpp
            ${BRAINSTORM_SOURCES}
        )
        
        target_link_libraries(test_gpu_searcher
            ${GTEST_LIBRARIES}
            ${GTEST_MAIN_LIBRARIES}
            Threads::Threads
            ${CUDA_LIBRARIES}
        )
        
        target_compile_definitions(test_gpu_searcher PRIVATE GPU_ENABLED)
        
        add_test(NAME GPUSearcherTest COMMAND test_gpu_searcher)
    endif()
endif()

# Register tests with CTest
add_test(NAME CriticalFunctionsTest COMMAND test_critical_functions)
add_test(NAME MemorySafetyTest COMMAND test_memory_safety)

# Test targets
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS test_critical_functions test_memory_safety
    COMMENT "Running all unit tests"
)

# Memory testing targets
add_custom_target(test_with_asan
    COMMAND ${CMAKE_COMMAND} -DUSE_ASAN=ON -B${CMAKE_BINARY_DIR}/asan -S${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}/asan
    COMMAND cd ${CMAKE_BINARY_DIR}/asan && ctest --verbose
    COMMENT "Running tests with AddressSanitizer"
)

add_custom_target(test_with_valgrind
    COMMAND valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes ./test_memory_safety
    DEPENDS test_memory_safety
    COMMENT "Running memory safety tests with Valgrind"
)

# Coverage target (requires gcov/lcov)
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
if(ENABLE_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} --coverage")
    
    add_custom_target(coverage
        COMMAND lcov --directory . --capture --output-file coverage.info
        COMMAND lcov --remove coverage.info '/usr/*' --output-file coverage.info
        COMMAND lcov --list coverage.info
        COMMAND genhtml -o coverage coverage.info
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating test coverage report"
    )
endif()

# Documentation
add_custom_target(test_docs
    COMMAND echo "Test Documentation:"
    COMMAND echo "=================="
    COMMAND echo ""
    COMMAND echo "Basic usage:"
    COMMAND echo "  make run_tests                    - Run all unit tests"
    COMMAND echo "  make test_with_asan              - Run tests with AddressSanitizer"
    COMMAND echo "  make test_with_valgrind          - Run tests with Valgrind"
    COMMAND echo ""
    COMMAND echo "Individual test executables:"
    COMMAND echo "  ./test_critical_functions        - Core functionality tests"
    COMMAND echo "  ./test_memory_safety             - Memory leak/safety tests"
    COMMAND echo ""
    COMMAND echo "Build options:"
    COMMAND echo "  -DUSE_ASAN=ON                   - Enable AddressSanitizer"
    COMMAND echo "  -DUSE_TSAN=ON                   - Enable ThreadSanitizer"
    COMMAND echo "  -DENABLE_COVERAGE=ON            - Enable code coverage"
    COMMAND echo "  -DBUILD_GPU_TESTS=ON            - Build CUDA/GPU tests"
    COMMENT "Displaying test documentation"
)

# Set test properties
set_tests_properties(CriticalFunctionsTest PROPERTIES
    TIMEOUT 60
    FAIL_REGULAR_EXPRESSION "FAILED;ERROR;SEGFAULT"
)

set_tests_properties(MemorySafetyTest PROPERTIES  
    TIMEOUT 120
    FAIL_REGULAR_EXPRESSION "FAILED;ERROR;SEGFAULT;LEAK"
)

# Print configuration info
message(STATUS "Brainstorm Test Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Use AddressSanitizer: ${USE_ASAN}")
message(STATUS "  Use ThreadSanitizer: ${USE_TSAN}")
message(STATUS "  Enable Coverage: ${ENABLE_COVERAGE}")
message(STATUS "  CUDA Available: ${CUDA_FOUND}")
if(CUDA_FOUND)
    message(STATUS "  Build GPU Tests: ${BUILD_GPU_TESTS}")
endif()