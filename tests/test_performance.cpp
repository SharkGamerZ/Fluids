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
    TestFluidMatrix fluidMatrix(0, 0.0, 0.0, 0.0);
    generate_random_fluid_matrix_params(fluidMatrix, 400);

    // Example usage with member function
    test_function_performance("lin_solve", 200, [&fluidMatrix] { fluidMatrix.lin_solve(X, fluidMatrix.density, fluidMatrix.density_prev, fluidMatrix.diff); });

    // Example usage with different arguments
    test_function_performance("OMP_lin_solve", 200, [&fluidMatrix] { fluidMatrix.OMP_lin_solve(X, fluidMatrix.density, fluidMatrix.density_prev, fluidMatrix.diff); });

    return EXIT_SUCCESS;
}
