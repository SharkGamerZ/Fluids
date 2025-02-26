#include "test_fluid_matrix.hpp"
#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Function to measure the time taken by a single execution of any function
template<typename Func, typename... Args>
std::chrono::microseconds measure_time(Func &&func, Args &&...args) {
    const auto start = std::chrono::high_resolution_clock::now();
    std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start);
}

// Function to perform multiple runs and collect performance metrics
template<typename Func, typename... Args>
void test_function_performance(const std::string &func_name, const int num_runs, Func &&func, Args &&...args) {
    using namespace std::chrono;
    std::vector<long long> times;
    times.reserve(num_runs);

    // Measure time for multiple runs
    for (int i = 0; i < num_runs; i++) {
        auto duration = measure_time(std::forward<Func>(func), std::forward<Args>(args)...);
        times.push_back(duration.count());
    }

    // Calculate statistics
    const auto min_time = *std::ranges::min_element(times);
    const auto max_time = *std::ranges::max_element(times);
    const long long sum_time = std::accumulate(times.begin(), times.end(), 0LL);
    const auto avg_time = sum_time / num_runs;

    // Print results
    std::cout << "Function: " << func_name << '\n';
    std::cout << "  Fastest: " << min_time << " μs\n";
    std::cout << "  Slowest: " << max_time << " μs\n";
    std::cout << "  Average: " << avg_time << " μs\n";
}

// Function to generate random values for FluidMatrix parameters
void generate_random_fluid_matrix_params(FluidMatrix &fluidMatrix, const int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution diff_dist(0.0, 1.0); // Random diffusion between 0.0 and 1.0
    std::uniform_real_distribution visc_dist(0.0, 1.0); // Random viscosity between 0.0 and 1.0
    std::uniform_real_distribution dt_dist(0.001, 0.1); // Random delta time between 0.001 and 0.1

    fluidMatrix.size = size;
    fluidMatrix.dt = dt_dist(gen);
    fluidMatrix.diff = diff_dist(gen);
    fluidMatrix.visc = visc_dist(gen);

    // Resize the matrices
    fluidMatrix.density.resize(fluidMatrix.size * fluidMatrix.size);
    fluidMatrix.density_prev.resize(fluidMatrix.size * fluidMatrix.size);
    fluidMatrix.vX.resize(fluidMatrix.size * fluidMatrix.size);
    fluidMatrix.vY.resize(fluidMatrix.size * fluidMatrix.size);
    fluidMatrix.vX_prev.resize(fluidMatrix.size * fluidMatrix.size);
    fluidMatrix.vY_prev.resize(fluidMatrix.size * fluidMatrix.size);

    // Initialize the matrices with some random values (just for testing)
    std::ranges::generate(fluidMatrix.density, [&gen, &diff_dist] { return diff_dist(gen); });
    std::ranges::generate(fluidMatrix.density_prev, [&gen, &diff_dist] { return diff_dist(gen); });
    std::ranges::generate(fluidMatrix.vX, [&gen, &visc_dist] { return visc_dist(gen); });
    std::ranges::generate(fluidMatrix.vY, [&gen, &visc_dist] { return visc_dist(gen); });
    std::ranges::generate(fluidMatrix.vX, [&gen, &visc_dist] { return visc_dist(gen); });
    std::ranges::generate(fluidMatrix.vY_prev, [&gen, &visc_dist] { return visc_dist(gen); });
}

int main() {
    // Create a FluidMatrix object with randomized parameters
    TestFluidMatrix fluidMatrix(5, 0.0, 0.0, 0.0);
    generate_random_fluid_matrix_params(fluidMatrix, 5);


    TestFluidMatrix OMP_fluidMatrix = fluidMatrix;
    TestFluidMatrix CUDA_fluidMatrix = fluidMatrix;


    if (fluidMatrix.density != OMP_fluidMatrix.density) {
        std::cerr << "OMP_FluidMatrix copy failed\n";
    }

    if (fluidMatrix.density != CUDA_fluidMatrix.density) {
        std::cerr << "CUDA_FluidMatrix copy failed\n";
    }

    int num_runs = 1;

    printf("Matrix Size: %d\n", fluidMatrix.size);
    
    // DIFFUSE
    test_function_performance("diffuse", num_runs, [&fluidMatrix] { fluidMatrix.diffuse(X, fluidMatrix.density, fluidMatrix.density_prev, fluidMatrix.visc, fluidMatrix.dt); });
    test_function_performance("OMP_diffuse", num_runs, [&OMP_fluidMatrix] { OMP_fluidMatrix.OMP_diffuse(X, OMP_fluidMatrix.density, OMP_fluidMatrix.density_prev, OMP_fluidMatrix.visc, OMP_fluidMatrix.dt); });
    test_function_performance("CUDA_diffuse", num_runs, [&CUDA_fluidMatrix] { CUDA_fluidMatrix.CUDA_diffuse(X, CUDA_fluidMatrix.density, CUDA_fluidMatrix.density_prev, CUDA_fluidMatrix.visc, CUDA_fluidMatrix.dt); });

    // Compare the results of the two functions
    if (fluidMatrix.density != OMP_fluidMatrix.density) {
        std::cerr << "diffuse and OMP_diffuse produced different results\n";
        // Print where the difference is
        for (int i = 0; i < fluidMatrix.size * fluidMatrix.size; i++) {
            if (fluidMatrix.density[i] != OMP_fluidMatrix.density[i]) {
                std::cerr << "i: " << i << " fluidMatrix.density[i]: " << fluidMatrix.density[i] << " OMP_fluidMatrix.density[i]: " << OMP_fluidMatrix.density[i] << '\n';
            }
        }
    }

    if (fluidMatrix.density != CUDA_fluidMatrix.density) {
        std::cerr << "diffuse and CUDA_diffuse produced different results\n";
        // Print where the difference is
        for (int i = 0; i < fluidMatrix.size * fluidMatrix.size; i++) {
            if (fluidMatrix.density[i] != CUDA_fluidMatrix.density[i]) {
                std::cerr << "i: " << i << " fluidMatrix.density[i]: " << fluidMatrix.density[i] << " CUDA_fluidMatrix.density[i]: " << CUDA_fluidMatrix.density[i] << '\n';
            }
        }
    }


    // ADVECT
    test_function_performance("advect", num_runs, [&fluidMatrix] { fluidMatrix.advect(ZERO, fluidMatrix.density, fluidMatrix.density_prev, fluidMatrix.vX, fluidMatrix.vY, fluidMatrix.dt); });
    test_function_performance("OMP_advect", num_runs, [&OMP_fluidMatrix] { OMP_fluidMatrix.OMP_advect(ZERO, OMP_fluidMatrix.density, OMP_fluidMatrix.density_prev, OMP_fluidMatrix.vX, OMP_fluidMatrix.vY, OMP_fluidMatrix.dt); });
    test_function_performance("CUDA_advect", num_runs, [&CUDA_fluidMatrix] { CUDA_fluidMatrix.CUDA_advect(ZERO, CUDA_fluidMatrix.density, CUDA_fluidMatrix.density_prev, CUDA_fluidMatrix.vX, CUDA_fluidMatrix.vY, CUDA_fluidMatrix.dt); });

    // Compare the results of the two functions
    if (fluidMatrix.density != OMP_fluidMatrix.density) {
        std::cerr << "advect and OMP_advect produced different results\n";
        // Print where the difference is
        for (int i = 0; i < fluidMatrix.size * fluidMatrix.size; i++) {
            if (fluidMatrix.density[i] != OMP_fluidMatrix.density[i]) {
                std::cerr << "i: " << i << " fluidMatrix.density[i]: " << fluidMatrix.density[i] << " OMP_fluidMatrix.density[i]: " << OMP_fluidMatrix.density[i] << '\n';
            }
        }
    }

    if (fluidMatrix.density != CUDA_fluidMatrix.density) {
        std::cerr << "advect and CUDA_advect produced different results\n";
        // Print where the difference is
        for (int i = 0; i < fluidMatrix.size * fluidMatrix.size; i++) {
            if (fluidMatrix.density[i] != CUDA_fluidMatrix.density[i]) {
                std::cerr << "i: " << i << " fluidMatrix.density[i]: " << fluidMatrix.density[i] << " CUDA_fluidMatrix.density[i]: " << CUDA_fluidMatrix.density[i] << '\n';
            }
        }
    }


    // PROJECT
    test_function_performance("project", num_runs, [&fluidMatrix] { fluidMatrix.project(fluidMatrix.vX, fluidMatrix.vY, fluidMatrix.vX_prev, fluidMatrix.vY_prev); });
    test_function_performance("OMP_project", num_runs, [&OMP_fluidMatrix] { OMP_fluidMatrix.OMP_project(OMP_fluidMatrix.vX, OMP_fluidMatrix.vY, OMP_fluidMatrix.vX_prev, OMP_fluidMatrix.vY_prev); });
    test_function_performance("CUDA_project", num_runs, [&CUDA_fluidMatrix] { CUDA_fluidMatrix.CUDA_project(CUDA_fluidMatrix.vX, CUDA_fluidMatrix.vY, CUDA_fluidMatrix.vX_prev, CUDA_fluidMatrix.vY_prev); });

    // Compare the results of the two functions
    if (fluidMatrix.vX != OMP_fluidMatrix.vX) {
        std::cerr << "project and OMP_project produced different results\n";
    }

    if (fluidMatrix.vX != CUDA_fluidMatrix.vX) {
        std::cerr << "project and CUDA_project produced different results\n";
    }

    return EXIT_SUCCESS;
}
