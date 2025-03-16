#include "test_fluid_matrix.hpp"
#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <random>
#include <ranges>
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

// Helper function to run a given function multiple times and return the median runtime
template<typename Func>
double measure_median_time(Func func, const int iterations) {
    std::vector<double> times(iterations);
    for (int i = 0; i < iterations; ++i) {
        times[i] = static_cast<double>(measure_time(func).count());
    }
    std::ranges::sort(times);
    return times[iterations / 2];
}

// Function to perform multiple runs and collect performance metrics
template<typename Func, typename... Args>
void test_function_performance(const std::string &func_name, const int num_runs, Func &&func, Args &&...args) {
    using namespace std::chrono;
    std::vector<long long> times(num_runs);

    // Measure time for multiple runs
    for (int i = 0; i < num_runs; i++) {
        auto duration = measure_time(std::forward<Func>(func), std::forward<Args>(args)...);
        times[i] = duration.count();
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
    std::uniform_real_distribution dt_dist(0.001, 0.1);   // Random delta time between 0.001 and 0.1

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

void compareMatrixes(const std::vector<double> &m1, const std::vector<double> &m2, const bool printMatrix, FILE *fp) {
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

        std::fprintf(fp, "%3.7f ", avg_diff);

        if (!printMatrix) return;
        // Print where the difference is
        for (int i = 0; i < size * size; i++) {
            if (m1[i] != m2[i]) {
                // Check if the cell is a boundary cell
                if (i % size == 0 || i % size == size - 1 || i / size == 0 || i / size == size - 1) {
                    std::cerr << "Boundary cell:";
                }
                std::cerr << "i: " << i << " m1.vX[i]: " << m1[i] << " m2.vX[i]: " << m2[i] << '\n';
            }
        }
    }
}

std::vector<double> solvePerfectMatrix(const uint32_t size, const std::vector<double> &oldValue, const double diffusionRate) {
    const int N = size - 2;       // Active grid (excluding boundaries)
    const int matrixSize = N * N; // Total number of unknowns

    // Construct the coefficient matrix A
    std::vector<std::vector<double>> A(matrixSize, std::vector<double>(matrixSize, 0.0));
    std::vector<double> b(matrixSize, 0.0); // Right-hand side vector

    double c = diffusionRate;

// Construct A based on the finite difference stencil
#pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            A[idx][idx] = 1 + 4 * c; // Center coefficient

            if (i > 0) A[idx][idx - N] = -c;     // Up
            if (i < N - 1) A[idx][idx + N] = -c; // Down
            if (j > 0) A[idx][idx - 1] = -c;     // Left
            if (j < N - 1) A[idx][idx + 1] = -c; // Right

            b[idx] = oldValue[(i + 1) * size + (j + 1)]; // Skip boundaries
        }
    }

    // LU Decomposition
    std::vector<std::vector<double>> L(matrixSize, std::vector<double>(matrixSize, 0.0));
    std::vector<std::vector<double>> U = A; // Copy of A to decompose

    for (int i = 0; i < matrixSize; i++) {
        L[i][i] = 1.0; // L has 1s on the diagonal

#pragma omp parallel for
        for (int j = i; j < matrixSize; j++) {
            double sum = 0.0;
            for (int k = 0; k < i; k++) sum += L[i][k] * U[k][j];
            U[i][j] = A[i][j] - sum;
        }

#pragma omp parallel for
        for (int j = i + 1; j < matrixSize; j++) {
            double sum = 0.0;
            for (int k = 0; k < i; k++) sum += L[j][k] * U[k][i];
            L[j][i] = (A[j][i] - sum) / U[i][i];
        }
    }

    // Forward substitution: solve Ly = b
    std::vector<double> y(matrixSize, 0.0);
#pragma omp parallel for
    for (int i = 0; i < matrixSize; i++) {
        double sum = 0.0;
        for (int j = 0; j < i; j++) sum += L[i][j] * y[j];
        y[i] = b[i] - sum;
    }

    // Backward substitution: solve Ux = y
    std::vector<double> x(matrixSize, 0.0);
#pragma omp parallel for
    for (int i = matrixSize - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < matrixSize; j++) sum += U[i][j] * x[j];
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

/// Test the performance of linear solvers on growing matrix sizes.
///
/// The results are exported to a CSV file for further analysis.
void test_linear_solvers() {
    constexpr int max_iterations = 100;
    constexpr int func_repeat = 10;
    constexpr int start_size = 50;
    constexpr int end_size = 1000;
    constexpr int size_increment = 50;

    // Open CSV to write results (file is cleared if it already exists)
    std::ofstream file("results.csv");
    if (!file.is_open()) {
        std::cerr << "Error opening file\n";
        return;
    }

    // Write CSV header
    file << "Matrix_Size,Iterations,Gauss_Serial,Gauss_OMP";
#ifdef CUDA_SUPPORT
    file << ",Gauss_CUDA";
#endif
    file << ",Jacobi_Serial,Jacobi_OMP";
#ifdef CUDA_SUPPORT
    file << ",Jacobi_CUDA";
#endif
    file << "\n";

    for (int size: std::views::iota(start_size, end_size + 1) | std::views::stride(size_increment)) {
        std::cout << "Matrix Size: " << size << std::endl;

        // Create a FluidMatrix object with randomized parameters
        TestFluidMatrix fluidMatrix(size, 0.0, 0.0, 0.0);
        generate_random_fluid_matrix_params(fluidMatrix, size);

        TestFluidMatrix OMP_fluidMatrix = fluidMatrix;
        if (fluidMatrix.density != OMP_fluidMatrix.density) {
            std::cerr << "OMP_FluidMatrix copy failed\n";
        }

#ifdef CUDA_SUPPORT
        TestFluidMatrix CUDA_fluidMatrix = fluidMatrix;
        if (fluidMatrix.density != CUDA_fluidMatrix.density) {
            std::cerr << "CUDA_FluidMatrix copy failed\n";
        }
#endif

        for (int i: std::views::iota(1, max_iterations + 1)) {
            TestFluidMatrix fluidMatrix_copy = fluidMatrix;
            TestFluidMatrix OMP_fluidMatrix_copy = OMP_fluidMatrix;
#ifdef CUDA_SUPPORT
            TestFluidMatrix CUDA_fluidMatrix_copy = CUDA_fluidMatrix;
#endif

            GAUSS_ITERATIONS = i;
            JACOBI_ITERATIONS = i;

            std::cout << "Iterations: " << i << std::endl;

            // GAUSS
            auto gauss_serial_time =
                    measure_median_time([&fluidMatrix] { fluidMatrix.gauss_lin_solve(X, fluidMatrix.density, fluidMatrix.density_prev, fluidMatrix.visc); }, func_repeat);
            auto gauss_omp_time = measure_median_time(
                    [&OMP_fluidMatrix] { OMP_fluidMatrix.OMP_gauss_lin_solve(X, OMP_fluidMatrix.density, OMP_fluidMatrix.density_prev, OMP_fluidMatrix.visc); }, func_repeat);
#ifdef CUDA_SUPPORT
            auto gauss_cuda_time = measure_median_time(
                    [&CUDA_fluidMatrix] {
                        CUDA_fluidMatrix.CUDA_lin_solve(X, CUDA_fluidMatrix.d_density, CUDA_fluidMatrix.d_density_prev, CUDA_fluidMatrix.visc,
                                                        1.0 / (1 + 4 * CUDA_fluidMatrix.visc));
                    },
                    func_repeat);
#endif

            // JACOBI
            auto jacobi_serial_time =
                    measure_median_time([&fluidMatrix] { fluidMatrix.jacobi_lin_solve(X, fluidMatrix.density, fluidMatrix.density_prev, fluidMatrix.visc); }, func_repeat);
            auto jacobi_omp_time = measure_median_time(
                    [&OMP_fluidMatrix] { OMP_fluidMatrix.OMP_jacobi_lin_solve(X, OMP_fluidMatrix.density, OMP_fluidMatrix.density_prev, OMP_fluidMatrix.visc); }, func_repeat);
#ifdef CUDA_SUPPORT
            auto jacobi_cuda_time = measure_median_time(
                    [&CUDA_fluidMatrix] {
                        CUDA_fluidMatrix.CUDA_lin_solve(X, CUDA_fluidMatrix.d_density, CUDA_fluidMatrix.d_density_prev, CUDA_fluidMatrix.visc,
                                                        1.0 / (1 + 4 * CUDA_fluidMatrix.visc));
                    },
                    func_repeat);
#endif

            // Write results to file in CSV format
            std::ostringstream oss;
            oss << size << "," << i << ",";
            oss << gauss_serial_time << "," << gauss_omp_time;
#ifdef CUDA_SUPPORT
            oss << "," << gauss_cuda_time;
#endif
            oss << "," << jacobi_serial_time << "," << jacobi_omp_time;
#ifdef CUDA_SUPPORT
            oss << "," << jacobi_cuda_time;
#endif
            oss << "\n";

            file << oss.str();
            file.flush();
        }
    }

    file.close();
}

int main() {
    // Compare performance of linear solvers on growing matrix sizes
    test_linear_solvers();
}
