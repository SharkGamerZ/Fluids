cmake_minimum_required(VERSION 3.30)
project(fluids LANGUAGES CUDA CXX C)

# Set standards
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD 23)

# Ensure build type is set
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type" FORCE)
endif ()

# Find dependencies
find_package(OpenMP REQUIRED)
find_package(OpenGL REQUIRED)

# Set build dependant flags
if (CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building in Debug mode: No optimizations.")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g -G")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
else ()
    message(STATUS "Building in Release mode: Enabling full optimizations.")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo --use_fast_math")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -flto")
endif ()

# Set flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall,-Wextra,-Wno-unknown-pragmas --Werror cross-execution-space-call") #-Wpedantic
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic") #-Werror
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic") #-Werror

# Add ImGui sources
file(GLOB IMGUI_SOURCES lib/imgui/*.cpp lib/imgui/backends/*.cpp)

# Add source files
set(SOURCE_FILES
        src/main.cpp
        src/gui.cpp
        src/renderer.cpp
        src/utils.cpp
        src/fluids/fluid_matrix.cpp # C++ source file (serial/openmp)
        src/fluids/fluid_matrix.cu  # CUDA source file
        ${IMGUI_SOURCES}
)

add_executable(fluids ${SOURCE_FILES})

# Set CUDA separable compilation
set_target_properties(fluids PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add OpenMP support for CUDA
target_compile_options(fluids PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>)

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
