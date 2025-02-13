cmake_minimum_required(VERSION 3.30)
project(fluids LANGUAGES CXX C)

# Set standards
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD 23)

# Ensure build type is set
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type" FORCE)
endif ()

# Find dependencies
find_package(OpenMP REQUIRED)
find_package(OpenGL REQUIRED)

# CUDA support check
include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 23)

    file(WRITE ${CMAKE_BINARY_DIR}/cuda_check.cu [[
        #include <cuda_runtime.h>
        int main() {
            int deviceCount;
            cudaError_t error = cudaGetDeviceCount(&deviceCount);
            return (error == cudaSuccess && deviceCount > 0) ? 0 : 1;
        }
    ]])

    try_run(RUN_RESULT COMPILE_RESULT
            ${CMAKE_BINARY_DIR}
            ${CMAKE_BINARY_DIR}/cuda_check.cu
            CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
            RUN_OUTPUT_VARIABLE RUN_OUTPUT
            COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT
    )

    if (COMPILE_RESULT AND RUN_RESULT EQUAL 0)
        set(CUDA_SUPPORT TRUE)
        add_compile_definitions(CUDA_SUPPORT=1)
        message(STATUS "CUDA-capable GPU detected.")
    else ()
        set(CUDA_SUPPORT FALSE)
        message(STATUS "NO CUDA-capable GPU detected - CUDA support disabled.")
    endif ()
else ()
    set(CUDA_SUPPORT FALSE)
    message(STATUS "CUDA compiler not found - CUDA support disabled.")
endif ()

# NOTE: test bypass, do not enable unless you have to debug CUDA code without a GPU
#set(CUDA_SUPPORT TRUE)
#add_compile_definitions(CUDA_SUPPORT=1)

# Set build dependant flags
if (CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building in Debug mode: No optimizations.")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
    if (CUDA_SUPPORT)
        set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g -G")
    endif ()
else ()
    message(STATUS "Building in Release mode: Enabling full optimizations.")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native")
    if (CUDA_SUPPORT)
        set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo --use_fast_math")
    endif ()
endif ()

# Set flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic") #-Werror
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic") #-Werror
if (CUDA_SUPPORT)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall,-Wextra,-Wno-unknown-pragmas --Werror cross-execution-space-call") #-Wpedantic
endif ()

# Add ImGui sources
file(GLOB IMGUI_SOURCES lib/imgui/*.cpp lib/imgui/backends/*.cpp)

# Add source files
set(SOURCE_FILES
        src/main.cpp
        src/gui.cpp
        src/renderer.cpp
        src/utils.cpp
        src/fluids/fluid_matrix.cpp # C++ source file (serial/openmp)
        ${IMGUI_SOURCES}
)
if (CUDA_SUPPORT)
    list(APPEND SOURCE_FILES src/fluids/fluid_matrix.cu)
endif ()

add_executable(fluids ${SOURCE_FILES})

if (CUDA_SUPPORT)
    # Set CUDA separable compilation
    set_target_properties(fluids PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    # Add OpenMP support for CUDA
    target_compile_options(fluids PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>)
endif ()

# Link libraries
target_link_libraries(fluids PRIVATE
        glfw
        OpenGL::GL
        GLEW
        OpenMP::OpenMP_CXX
        m
        X11
)

# Include directories
target_include_directories(fluids PRIVATE lib/imgui/ lib/imgui/backends)

# ---------------------------------------------------------
# Testing setup
# ---------------------------------------------------------

# Add test files
set(TEST_FILES
        tests/test_performance.cpp
        tests/test_fluid_matrix.hpp
        src/fluids/fluid_matrix.cpp
        src/utils.cpp
)
if (CUDA_SUPPORT)
    list(APPEND TEST_FILES src/fluids/fluid_matrix.cu)
endif ()

add_executable(test_fluids ${TEST_FILES})

if (CUDA_SUPPORT)
    # Set CUDA separable compilation
    set_target_properties(test_fluids PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    # Add OpenMP support for CUDA
    target_compile_options(test_fluids PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>)
endif ()

# Copy relevant flags and settings from main target
target_compile_options(test_fluids PRIVATE
        $<$<CONFIG:Release>:-O3 -march=native>
        $<$<CONFIG:Debug>:-O0 -g>
)

# Link only necessary libraries
target_link_libraries(test_fluids PRIVATE
        OpenMP::OpenMP_CXX
)
