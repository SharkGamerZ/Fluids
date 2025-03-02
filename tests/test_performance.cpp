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

void compareMatrixes(std::vector<double> m1, std::vector<double> m2, bool printMatrix) {
    int size = sqrt(m1.size());

    if (m1 != m2) {
        std::cerr << "project and CUDA_project produced different results\n";

        if (!printMatrix) return;
        // Print where the difference is
        for (int i = 0; i < size * size; i++) {
            if (m1[i] != m2[i]) {
                // Check if the cell is a boundary cell
                if (i % size == 0 || i % size == size - 1 || i / size == 0 || i / size == size - 1) {
                    std::cerr<< "Boundary cell:";
                }
                std::cerr << "i: " << i << " fluidMatrix.vX[i]: " << m1[i] << " CUDA_fluidMatrix.vX[i]: " << m2[i] << '\n';
            }
        }
    }
}


std::vector<double> solvePerfectMatrix(uint32_t size, const std::vector<double> &oldValue, double diffusionRate) {
    const int N = size - 2;  // Active grid (excluding boundaries)
    const int matrixSize = N * N;  // Total number of unknowns

    // Construct the coefficient matrix A
    std::vector<std::vector<double>> A(matrixSize, std::vector<double>(matrixSize, 0.0));
    std::vector<double> b(matrixSize, 0.0);  // Right-hand side vector

    double c = diffusionRate;
    double cRecip = 1.0 / (1 + 4 * c);

    // Construct A based on the finite difference stencil
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            A[idx][idx] = 1 + 4 * c;  // Center coefficient

            if (i > 0) A[idx][idx - N] = -c;  // Up
            if (i < N - 1) A[idx][idx + N] = -c;  // Down
            if (j > 0) A[idx][idx - 1] = -c;  // Left
            if (j < N - 1) A[idx][idx + 1] = -c;  // Right

            b[idx] = oldValue[(i + 1) * size + (j + 1)];  // Skip boundaries
        }
    }

    // LU Decomposition
    std::vector<std::vector<double>> L(matrixSize, std::vector<double>(matrixSize, 0.0));
    std::vector<std::vector<double>> U = A;  // Copy of A to decompose

    for (int i = 0; i < matrixSize; i++) {
        L[i][i] = 1.0;  // L has 1s on the diagonal
        for (int j = i; j < matrixSize; j++) {
            for (int k = 0; k < i; k++)
                U[i][j] -= L[i][k] * U[k][j];
        }
        for (int j = i + 1; j < matrixSize; j++) {
            for (int k = 0; k < i; k++)
                U[j][i] -= L[j][k] * U[k][i];
            L[j][i] = U[j][i] / U[i][i];
        }
    }

    // Forward substitution: solve Ly = b
    std::vector<double> y(matrixSize, 0.0);
    for (int i = 0; i < matrixSize; i++) {
        y[i] = b[i];
        for (int j = 0; j < i; j++)
            y[i] -= L[i][j] * y[j];
    }

    // Backward substitution: solve Ux = y
    std::vector<double> x(matrixSize, 0.0);
    for (int i = matrixSize - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < matrixSize; j++)
            x[i] -= U[i][j] * x[j];
        x[i] /= U[i][i];
    }

    // Convert x back to full grid format
    std::vector<double> result(size * size, 0.0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[(i + 1) * size + (j + 1)] = x[i * N + j];
        }
    }

    return result;
}


int main() {
    // Create a FluidMatrix object with randomized parameters
    int size = 20;
    TestFluidMatrix fluidMatrix(size, 0.0, 0.0, 0.0);
    generate_random_fluid_matrix_params(fluidMatrix, size);


    TestFluidMatrix OMP_fluidMatrix = fluidMatrix;
    if (fluidMatrix.density != OMP_fluidMatrix.density) { std::cerr << "OMP_FluidMatrix copy failed\n"; }

    TestFluidMatrix CUDA_fluidMatrix = fluidMatrix;
    if (fluidMatrix.density != CUDA_fluidMatrix.density) { std::cerr << "CUDA_FluidMatrix copy failed\n"; }

    int num_runs = 20;

    printf("Matrix Size: %d\n", fluidMatrix.size);

    std::vector<double> result = solvePerfectMatrix(fluidMatrix.size, fluidMatrix.density, fluidMatrix.diff);
    test_function_performance("solvePerfectMatrix", num_runs, [&fluidMatrix, &result] { fluidMatrix.OMP_gauss_lin_solve(ZERO, fluidMatrix.density, fluidMatrix.density_prev, fluidMatrix.diff); });
    
    // DIFFUSE
    test_function_performance("diffuse", num_runs, [&fluidMatrix] { fluidMatrix.diffuse(X, fluidMatrix.density, fluidMatrix.density_prev, fluidMatrix.visc, fluidMatrix.dt); });
    test_function_performance("OMP_diffuse", num_runs, [&OMP_fluidMatrix] { OMP_fluidMatrix.OMP_diffuse(X, OMP_fluidMatrix.density, OMP_fluidMatrix.density_prev, OMP_fluidMatrix.visc, OMP_fluidMatrix.dt); });
    test_function_performance("CUDA_diffuse", num_runs, [&CUDA_fluidMatrix] { CUDA_fluidMatrix.CUDA_diffuse(X, CUDA_fluidMatrix.density, CUDA_fluidMatrix.density_prev, CUDA_fluidMatrix.visc, CUDA_fluidMatrix.dt); });

    compareMatrixes(fluidMatrix.density, OMP_fluidMatrix.density, false);
    compareMatrixes(fluidMatrix.density, CUDA_fluidMatrix.density, false);

    // ADVECT
    test_function_performance("advect", num_runs, [&fluidMatrix] { fluidMatrix.advect(ZERO, fluidMatrix.density, fluidMatrix.density_prev, fluidMatrix.vX, fluidMatrix.vY, fluidMatrix.dt); });
    test_function_performance("OMP_advect", num_runs, [&OMP_fluidMatrix] { OMP_fluidMatrix.OMP_advect(ZERO, OMP_fluidMatrix.density, OMP_fluidMatrix.density_prev, OMP_fluidMatrix.vX, OMP_fluidMatrix.vY, OMP_fluidMatrix.dt); });
    test_function_performance("CUDA_advect", num_runs, [&CUDA_fluidMatrix] { CUDA_fluidMatrix.CUDA_advect(ZERO, CUDA_fluidMatrix.density, CUDA_fluidMatrix.density_prev, CUDA_fluidMatrix.vX, CUDA_fluidMatrix.vY, CUDA_fluidMatrix.dt); });

    compareMatrixes(fluidMatrix.density, OMP_fluidMatrix.density, false);
    compareMatrixes(fluidMatrix.density, CUDA_fluidMatrix.density, false);

    // PROJECT
    test_function_performance("project", num_runs, [&fluidMatrix] { fluidMatrix.project(fluidMatrix.vX, fluidMatrix.vY, fluidMatrix.vX_prev, fluidMatrix.vY_prev); });
    test_function_performance("OMP_project", num_runs, [&OMP_fluidMatrix] { OMP_fluidMatrix.OMP_project(OMP_fluidMatrix.vX, OMP_fluidMatrix.vY, OMP_fluidMatrix.vX_prev, OMP_fluidMatrix.vY_prev); });
    test_function_performance("CUDA_project", num_runs, [&CUDA_fluidMatrix] { CUDA_fluidMatrix.CUDA_project(CUDA_fluidMatrix.vX, CUDA_fluidMatrix.vY, CUDA_fluidMatrix.vX_prev, CUDA_fluidMatrix.vY_prev); });

    compareMatrixes(fluidMatrix.vX, OMP_fluidMatrix.vX, false);
    compareMatrixes(fluidMatrix.vX, CUDA_fluidMatrix.vX, false);

    return EXIT_SUCCESS;
}
