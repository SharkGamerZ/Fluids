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
    std::uniform_real_distribution diff_dist(0.0, 100.0); // Random diffusion between 0.0 and 1.0
    std::uniform_real_distribution visc_dist(0.0, 100.0); // Random viscosity between 0.0 and 1.0
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

void compareMatrixes(std::vector<double> m1, std::vector<double> m2, bool printMatrix, FILE *fp) {
    int size = sqrt(m1.size());

    if (m1 != m2) {
        std::cerr << "produced different results\n";

        // Calculate min, max and avg difference
        double min_diff = std::numeric_limits<double>::max();
        double max_diff = std::numeric_limits<double>::min();
        double avg_diff = 0.0;
        int diff_count = 0;
        
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                // Check if the cell is a boundary cell
                if (i == 0 || i == size - 1 || j == 0 || j == size - 1) continue;

                if (m1[i * size + j] != m2[i * size + j]) {
                    min_diff = std::min(min_diff, std::abs(m1[i * size + j] - m2[i * size + j]));
                    max_diff = std::max(max_diff, std::abs(m1[i * size + j] - m2[i * size + j]));
                    avg_diff += std::abs(m1[i * size + j] - m2[i * size + j]);
                    diff_count++;
                }
            }
        }

        avg_diff /= diff_count;

        std::cerr << "Min difference: " << min_diff << '\n';
        std::cerr << "Max difference: " << max_diff << '\n';
        std::cerr << "Average difference: " << avg_diff << '\n';

        std::cout << std::endl;

        fprintf(fp, "%3.7f ", avg_diff);

        if (!printMatrix) return;
        // Print where the difference is
        for (int i = 0; i < size * size; i++) {
            if (m1[i] != m2[i]) {
                // Check if the cell is a boundary cell
                if (i % size == 0 || i % size == size - 1 || i / size == 0 || i / size == size - 1) {
                    std::cerr<< "Boundary cell:";
                }
                std::cerr << "i: " << i << " m1.vX[i]: " << m1[i] << " m2.vX[i]: " << m2[i] << '\n';
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
    #pragma omp parallel for collapse(2)
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
        
        #pragma omp parallel for
        for (int j = i; j < matrixSize; j++) {
            double sum = 0.0;
            for (int k = 0; k < i; k++)
                sum += L[i][k] * U[k][j]; 
            U[i][j] = A[i][j] - sum;
        }

        #pragma omp parallel for
        for (int j = i + 1; j < matrixSize; j++) {
            double sum = 0.0;
            for (int k = 0; k < i; k++)
                sum += L[j][k] * U[k][i];
            L[j][i] = (A[j][i] - sum) / U[i][i];
        }
    }

    // Forward substitution: solve Ly = b
    std::vector<double> y(matrixSize, 0.0);
    #pragma omp parallel for
    for (int i = 0; i < matrixSize; i++) {
        double sum = 0.0;
        for (int j = 0; j < i; j++)
            sum += L[i][j] * y[j];
        y[i] = b[i] - sum;
    }

    // Backward substitution: solve Ux = y
    std::vector<double> x(matrixSize, 0.0);
    #pragma omp parallel for
    for (int i = matrixSize - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < matrixSize; j++)
            sum += U[i][j] * x[j];
        x[i] = (y[i] - sum) / U[i][i];
    }

    // Convert x back to full grid format
    std::vector<double> result(size * size, 0.0);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[(i + 1) * size + (j + 1)] = x[i * N + j];
        }
    }

    return result;
}

int main() {
    // Create a FluidMatrix object with randomized parameters
    int size = 50;
    TestFluidMatrix fluidMatrix(size, 0.0, 0.0, 0.0);
    generate_random_fluid_matrix_params(fluidMatrix, size);


    TestFluidMatrix OMP_fluidMatrix = fluidMatrix;
    if (fluidMatrix.density != OMP_fluidMatrix.density) { std::cerr << "OMP_FluidMatrix copy failed\n"; }

    int num_runs = 10;

    printf("Matrix Size: %d\n", fluidMatrix.size);

    std::vector<double> result = solvePerfectMatrix(fluidMatrix.size, fluidMatrix.density, fluidMatrix.diff);

    // Open file to write results
    FILE *fp = fopen("results.txt", "w");
    if (fp == NULL) {
        std::cerr << "Error opening file\n";
        return EXIT_FAILURE;
    }

    // Write iterations, serial, omp and cuda header
    fprintf(fp, "%10s", "Iterations");
    fprintf(fp, "%10s","Serial ");
    fprintf(fp, "%10s","OMP ");
    fprintf(fp, "\n");

    for (int i = 0; i <= 1000; i++) {
        // Write number of itertion in the file
        fprintf(fp, "%10d ", i);

        TestFluidMatrix fluidMatrix_copy = fluidMatrix;
        TestFluidMatrix OMP_fluidMatrix_copy = OMP_fluidMatrix;

        GAUSS_ITERATIONS = i;
        JACOBI_ITERATIONS = i;

        std::cout << "Iterations: " << i << std::endl;

        // DIFFUSE
        test_function_performance("diffuse", num_runs, [&fluidMatrix] { fluidMatrix.diffuse(X, fluidMatrix.density, fluidMatrix.density_prev, fluidMatrix.visc, fluidMatrix.dt); });
        test_function_performance("OMP_diffuse", num_runs, [&OMP_fluidMatrix] { OMP_fluidMatrix.OMP_diffuse(X, OMP_fluidMatrix.density, OMP_fluidMatrix.density_prev, OMP_fluidMatrix.visc, OMP_fluidMatrix.dt); });

        printf("Comparing perfect and serial diffuse()\n");
        compareMatrixes(result, fluidMatrix.density, false, fp);

        printf("Comparing perfect and OMP diffuse()\n");
        compareMatrixes(result, OMP_fluidMatrix.density, false, fp);

        std::cout << std::endl << std::endl;

        fluidMatrix = fluidMatrix_copy;
        OMP_fluidMatrix = OMP_fluidMatrix_copy;

        fprintf(fp, "\n");
    }
}
