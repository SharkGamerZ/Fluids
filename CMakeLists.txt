cmake_minimum_required(VERSION 3.26)
project(fluids CUDA CXX)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CXX_STANDARD 23)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic") #-Werror

set(SOURCE_FILES
        src/gui.cpp
        src/main.cpp
        src/fluids/FluidMatrix.cpp
        src/Utils.cpp
)

set(TEST_SOURCE_FILES
        test/test.cu
        src/Utils.cpp
)


add_subdirectory(lib/imgui/)

# Aggiungi tutte le sorgenti del progetto
add_executable(fluids ${SOURCE_FILES})
add_executable(test ${TEST_SOURCE_FILES})

# Aggiungi le opzioni di compilazione per CUDA
target_compile_options(fluids PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>)
target_compile_options(test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>)

# Link alle librerie necessarie
target_link_libraries(fluids PRIVATE -lglfw -lGL -lGLEW -lm -lX11 imgui -fopenmp)
target_link_libraries(fluids PRIVATE OpenMP::OpenMP_CXX)

target_include_directories(fluids PRIVATE lib/imgui/)


target_link_libraries(test PRIVATE OpenMP::OpenMP_CXX)
