cmake_minimum_required(VERSION 3.12)
project(ParallelBFS LANGUAGES CXX)

# Set output directories to keep build files organized
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# OpenMP configuration
find_package(OpenMP REQUIRED)

# Platform-specific settings
if(WIN32)
    # Windows-specific settings
    add_definitions(-DNOMINMAX)  # Disable min/max macros
    if(MINGW)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
else()
    # Linux/macOS settings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Include directories
include_directories(include)

# Main BFS executable (if you still want it)
add_executable(parallel_bfs 
    src/main.cpp
    src/parallel_bfs.cpp
)
target_link_libraries(parallel_bfs 
    PRIVATE 
    OpenMP::OpenMP_CXX
)

# Benchmark executable
add_executable(bfs_benchmark
    src/benchmark.cpp
    src/parallel_bfs.cpp  # Include BFS implementation
)
target_link_libraries(bfs_benchmark
    PRIVATE 
    OpenMP::OpenMP_CXX
)

# Installation settings (optional)
install(TARGETS parallel_bfs bfs_benchmark
    RUNTIME DESTINATION bin
)

# Add tests (optional)
enable_testing()
add_test(NAME parallel_bfs_test
    COMMAND parallel_bfs
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

# Add benchmark test
add_test(NAME bfs_benchmark_test
    COMMAND bfs_benchmark 1  # Run with 1 thread for quick test
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

# Only copy data directory if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data")
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})
endif()