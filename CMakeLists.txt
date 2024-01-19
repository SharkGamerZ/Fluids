cmake_minimum_required(VERSION 3.26)
project(fluids CUDA CXX)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra") #-Werror -Wpedantic

set(SOURCE_FILES
        src/gui.cpp
        src/gui.hpp
        src/main.cpp
        src/main.hpp
        src/fluid2d.cpp
        src/fluid2d.hpp
)

add_subdirectory(lib/imgui)

# Aggiungi tutte le sorgenti del progetto
add_executable(fluids ${SOURCE_FILES})

# Aggiungi le opzioni di compilazione per CUDA
target_compile_options(fluids PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp> )

# Link alle librerie necessarie
target_link_libraries(fluids PRIVATE -lglfw -lGL -lGLEW -lm -lX11 imgui)

target_include_directories(fluids PRIVATE lib/imgui/)

