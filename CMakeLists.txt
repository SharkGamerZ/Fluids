# Enable vcpkg support
if (DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Setting CMAKE_TOOLCHAIN_FILE to vcpkg: $ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif ()

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
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
if (NOT WIN32)
    find_package(X11 REQUIRED)
endif ()

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
        # if not handled in the source code 'FATAL_ERROR' should be the message type
        message(STATUS "NO CUDA-capable GPU detected - CUDA support disabled.")
    endif ()
else ()
    set(CUDA_SUPPORT FALSE)
    # if not handled in the source code 'FATAL_ERROR' should be the message type
    message(STATUS "CUDA compiler not found - CUDA support disabled.")
endif ()

# NOTE: test bypass, do not enable unless you have to debug CUDA code without a GPU
#set(CUDA_SUPPORT TRUE)
#add_compile_definitions(CUDA_SUPPORT=1)

# ---------------------------------------------------------
# Shared setup
# ---------------------------------------------------------

add_library(commons INTERFACE)
target_compile_options(commons INTERFACE
        # Debug configuration
        $<$<CONFIG:Debug>:
            $<$<COMPILE_LANGUAGE:CXX,C>:
                $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O0 -g>
                $<$<C_COMPILER_ID:GNU,Clang,AppleClang>:-O0 -g>
                $<$<CXX_COMPILER_ID:MSVC>:/Od /Zi /MDd>
                $<$<C_COMPILER_ID:MSVC>:/Od /Zi /MDd>
            >
        >

        # Release configuration
        $<$<CONFIG:Release>:
            $<$<COMPILE_LANGUAGE:CXX,C>:
                $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3 -march=native>
                $<$<C_COMPILER_ID:GNU,Clang,AppleClang>:-O3 -march=native>
                $<$<CXX_COMPILER_ID:MSVC>:/O2 /MD>
                $<$<C_COMPILER_ID:MSVC>:/O2 /MD>
            >
        >

        # Warning configuration
        $<$<COMPILE_LANGUAGE:CXX,C>:
            $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
            $<$<C_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
            $<$<CXX_COMPILER_ID:MSVC>:/W4>
            $<$<C_COMPILER_ID:MSVC>:/W4>
        >
)

if (CUDA_SUPPORT)
    target_compile_options(commons INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:
                $<$<CONFIG:Debug>:-O0 -g -G>
                $<$<CONFIG:Release>:-O3 -lineinfo --use_fast_math>
                $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Xcompiler=-Wall,-Wextra,-Wno-unknown-pragmas,-fopenmp>
                $<$<CXX_COMPILER_ID:MSVC>:-Xcompiler=/W4,/openmp>
                --Werror cross-execution-space-call
            >
    )
endif ()

# ---------------------------------------------------------
# Default setup
# ---------------------------------------------------------

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
    set_target_properties(fluids PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()

# Link libraries
target_link_libraries(fluids PRIVATE
        commons
        glfw
        OpenGL::GL
        $<IF:$<PLATFORM_ID:Windows>,GLEW::GLEW,GLEW>
        OpenMP::OpenMP_CXX
        $<$<NOT:$<PLATFORM_ID:Windows>>:m>
        $<$<NOT:$<PLATFORM_ID:Windows>>:X11>
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
    set_target_properties(test_fluids PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()

# Link only necessary libraries
target_link_libraries(test_fluids PRIVATE
        commons
        OpenMP::OpenMP_CXX
)
