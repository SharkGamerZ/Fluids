cmake_minimum_required(VERSION 3.27)
project(fluids CUDA CXX)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CXX_STANDARD 20)


set(SOURCE_FILES
        src/main.cpp
        
)

# Aggiungi tutte le sorgenti del progetto
add_executable(fluids ${SOURCE_FILES})

add_subdirectory(lib/imgui)

# Aggiungi le opzioni di compilazione per CUDA
target_compile_options(fluids PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp> )

# Link alle librerie necessarie
target_link_libraries(fluids PRIVATE -lglfw -lGL -lm -lX11 imgui)

target_include_directories(fluids PRIVATE lib/imgui/)


# Per generare il file per sonarlint
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)