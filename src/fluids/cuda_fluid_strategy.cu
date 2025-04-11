#include "cuda_fluid_strategy.cuh"
#include "fluid_data_model.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>

#define gpuErrchk(ans) gpuAssert((ans), __FILE__, __LINE__)

inline void gpuAssert(const cudaError_t code, const char *file, const int line) {
    if (code != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error: " << cudaGetErrorString(code) << " at " << file << ":" << line;
        const std::string err_msg = oss.str();

        std::cerr << err_msg << std::endl;
        cudaDeviceReset();
        throw std::runtime_error(err_msg);
    }
}

__device__ int cuda_index(const int i, const int j, const int size) {
    return i * size + j;
}

__global__ void addVelocity_kernel(double *d_vX, double *d_vY, const int x, const int y, const int size, const double amountX, const double amountY) {
    d_vX[cuda_index(y, x, size)] += amountY;
    d_vY[cuda_index(y, x, size)] += amountX;
}

__global__ void addDensity_kernel(double *d_density, const int x, const int y, const int size, const double amount) {
    d_density[cuda_index(y, x, size)] += amount;
}

__global__ void advect_kernel(const int size, double *d, const double *d0, const double *vX, const double *vY, const double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < 1 || i >= size - 1 || j < 1 || j >= size - 1) return;

    int i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1, dt0;

    dt0 = dt * (size - 2);

    x = i - dt0 * vX[cuda_index(i, j, size)];
    y = j - dt0 * vY[cuda_index(i, j, size)];

    if (x < 0.5f) x = 0.5f;
    if (x > size - 2 + 0.5f) x = size - 2 + 0.5f;
    i0 = (int) x;
    i1 = i0 + 1;

    if (y < 0.5f) y = 0.5f;
    if (y > size - 2 + 0.5f) y = size - 2 + 0.5f;
    j0 = (int) y;
    j1 = j0 + 1;

    s1 = x - i0;
    s0 = 1 - s1;
    t1 = y - j0;
    t0 = 1 - t1;

    d[cuda_index(i, j, size)] = s0 * (t0 * d0[cuda_index(i0, j0, size)] +
                                      t1 * d0[cuda_index(i0, j1, size)]) +
                               s1 * (t0 * d0[cuda_index(i1, j0, size)] +
                                     t1 * d0[cuda_index(i1, j1, size)]);
}

__global__ void project_kernel(const int size, const double *vX, const double *vY, double *vY_prev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < 1 || i >= size - 1 || j < 1 || j >= size - 1) return;

    vY_prev[cuda_index(i, j, size)] = -0.5 * (vX[cuda_index(i + 1, j, size)] -
                                             vX[cuda_index(i - 1, j, size)] +
                                             vY[cuda_index(i, j + 1, size)] -
                                             vY[cuda_index(i, j - 1, size)]) * (size - 2);
}

__global__ void update_velocity_kernel(const int size, double *vX, double *vY, const double *vX_prev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < 1 || i >= size - 1 || j < 1 || j >= size - 1) return;

    vX[cuda_index(i, j, size)] -= 0.5 * (vX_prev[cuda_index(i + 1, j, size)] -
                                         vX_prev[cuda_index(i - 1, j, size)]) / (size - 2);
    vY[cuda_index(i, j, size)] -= 0.5 * (vX_prev[cuda_index(i, j + 1, size)] -
                                         vX_prev[cuda_index(i, j - 1, size)]) / (size - 2);
}

__global__ void lin_solve_kernel(const int size, const double *d_value, const double *d_oldValue, double *d_newValue, const double diffusionRate, const double cRecip) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < 1 || i >= size - 1 || j < 1 || j >= size - 1) return;

    d_newValue[cuda_index(i, j, size)] = (d_oldValue[cuda_index(i, j, size)] +
                                         diffusionRate * (d_value[cuda_index(i + 1, j, size)] +
                                                         d_value[cuda_index(i - 1, j, size)] +
                                                         d_value[cuda_index(i, j + 1, size)] +
                                                         d_value[cuda_index(i, j - 1, size)])) * cRecip;
}

__global__ void fade_density_kernel(const int size, double *density) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 0 || i >= size * size) return;

    double d = density[i];
    density[i] = (d - 0.005f < 0) ? 0 : d - 0.005f;
}

__global__ void set_bnd_kernel(const Axis mode, double *d_value, const int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= size || j >= size) return;

    // Top boundary (j == 0) and bottom boundary (j == size-1)
    if (j == 0 && i > 0 && i < size - 1) {
        d_value[cuda_index(i, 0, size)] = (mode == Axis::Y) ?
            -d_value[cuda_index(i, 1, size)] : d_value[cuda_index(i, 1, size)];
    }
    if (j == size - 1 && i > 0 && i < size - 1) {
        d_value[cuda_index(i, size - 1, size)] = (mode == Axis::Y) ?
            -d_value[cuda_index(i, size - 2, size)] : d_value[cuda_index(i, size - 2, size)];
    }

    // Left boundary (i == 0) and right boundary (i == size-1)
    if (i == 0 && j > 0 && j < size - 1) {
        d_value[cuda_index(0, j, size)] = (mode == Axis::X) ?
            -d_value[cuda_index(1, j, size)] : d_value[cuda_index(1, j, size)];
    }
    if (i == size - 1 && j > 0 && j < size - 1) {
        d_value[cuda_index(size - 1, j, size)] = (mode == Axis::X) ?
            -d_value[cuda_index(size - 2, j, size)] : d_value[cuda_index(size - 2, j, size)];
    }

    // Corners
    if (i == 0 && j == 0) {
        d_value[cuda_index(0, 0, size)] = 0.5 * (d_value[cuda_index(1, 0, size)] +
                                                d_value[cuda_index(0, 1, size)]);
    }
    if (i == 0 && j == size - 1) {
        d_value[cuda_index(0, size - 1, size)] = 0.5 * (d_value[cuda_index(1, size - 1, size)] +
                                                       d_value[cuda_index(0, size - 2, size)]);
    }
    if (i == size - 1 && j == 0) {
        d_value[cuda_index(size - 1, 0, size)] = 0.5 * (d_value[cuda_index(size - 2, 0, size)] +
                                                       d_value[cuda_index(size - 1, 1, size)]);
    }
    if (i == size - 1 && j == size - 1) {
        d_value[cuda_index(size - 1, size - 1, size)] = 0.5 * (d_value[cuda_index(size - 2, size - 1, size)] +
                                                              d_value[cuda_index(size - 1, size - 2, size)]);
    }
}

// Constructor
CUDAFluidStrategy::CUDAFluidStrategy() {
    // Constructor is empty as initialization happens when step is first called
}

// Destructor
CUDAFluidStrategy::~CUDAFluidStrategy() {
    destroy();
}

void CUDAFluidStrategy::init(const FluidDataModel &dataModel) {
    const size_t size_bytes = dataModel.size * dataModel.size * sizeof(double);

    gpuErrchk(cudaMalloc(&d_density, size_bytes));
    gpuErrchk(cudaMalloc(&d_density_prev, size_bytes));
    gpuErrchk(cudaMalloc(&d_vX, size_bytes));
    gpuErrchk(cudaMalloc(&d_vX_prev, size_bytes));
    gpuErrchk(cudaMalloc(&d_vY, size_bytes));
    gpuErrchk(cudaMalloc(&d_vY_prev, size_bytes));
    gpuErrchk(cudaMalloc(&d_newValue, size_bytes));
}

void CUDAFluidStrategy::destroy() {
    if (d_density != nullptr) {
        gpuErrchk(cudaFree(d_density));
        gpuErrchk(cudaFree(d_density_prev));
        gpuErrchk(cudaFree(d_vX));
        gpuErrchk(cudaFree(d_vX_prev));
        gpuErrchk(cudaFree(d_vY));
        gpuErrchk(cudaFree(d_vY_prev));
        gpuErrchk(cudaFree(d_newValue));

        d_density = nullptr;
        d_density_prev = nullptr;
        d_vX = nullptr;
        d_vX_prev = nullptr;
        d_vY = nullptr;
        d_vY_prev = nullptr;
        d_newValue = nullptr;
    }
}

void CUDAFluidStrategy::copyToDevice(const FluidDataModel &dataModel) const {
    const size_t size_bytes = dataModel.size * dataModel.size * sizeof(double);

    gpuErrchk(cudaMemcpy(d_density, dataModel.density.data(), size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_density_prev, dataModel.density_prev.data(), size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_vX, dataModel.vX.data(), size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_vX_prev, dataModel.vX_prev.data(), size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_vY, dataModel.vY.data(), size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_vY_prev, dataModel.vY_prev.data(), size_bytes, cudaMemcpyHostToDevice));
}

void CUDAFluidStrategy::copyToHost(FluidDataModel &dataModel) const {
    const size_t size_bytes = dataModel.size * dataModel.size * sizeof(double);

    gpuErrchk(cudaMemcpyAsync(dataModel.density.data(), d_density, size_bytes, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpyAsync(dataModel.vX.data(), d_vX, size_bytes, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpyAsync(dataModel.vY.data(), d_vY, size_bytes, cudaMemcpyDeviceToHost));

    // Calculate vorticity on CPU since it's not frequently needed for simulation steps
    const double h = 1.0 / (dataModel.size - 2);
    for (int i = 1; i < dataModel.size - 1; i++) {
        for (int j = 1; j < dataModel.size - 1; j++) {
            int idx = FluidDataModel::index(i, j, dataModel.size);
            double dv_dx = (dataModel.vY[FluidDataModel::index(i + 1, j, dataModel.size)] -
                           dataModel.vY[FluidDataModel::index(i - 1, j, dataModel.size)]) / (2 * h);
            double du_dy = (dataModel.vX[FluidDataModel::index(i, j + 1, dataModel.size)] -
                           dataModel.vX[FluidDataModel::index(i, j - 1, dataModel.size)]) / (2 * h);
            dataModel.vorticity[idx] = dv_dx - du_dy;
        }
    }
}

void CUDAFluidStrategy::step(FluidDataModel &dataModel) {
    // Initialize CUDA memory if not already done
    if (d_density == nullptr) {
        init(dataModel);
        copyToDevice(dataModel);
    }

    // Velocity
    {
        std::swap(d_vX_prev, d_vX);
        diffuse(dataModel, Axis::X, d_vX, d_vX_prev, dataModel.visc, dataModel.dt);

        std::swap(d_vY_prev, d_vY);
        diffuse(dataModel, Axis::Y, d_vY, d_vY_prev, dataModel.visc, dataModel.dt);

        project(dataModel, d_vX, d_vY, d_vX_prev, d_vY_prev);

        std::swap(d_vX_prev, d_vX);
        std::swap(d_vY_prev, d_vY);

        advect(dataModel, Axis::X, d_vX, d_vX_prev, d_vX_prev, d_vY_prev, dataModel.dt);
        advect(dataModel, Axis::Y, d_vY, d_vY_prev, d_vX_prev, d_vY_prev, dataModel.dt);

        project(dataModel, d_vX, d_vY, d_vX_prev, d_vY_prev);
    }

    // Density
    {
        std::swap(d_density_prev, d_density);
        diffuse(dataModel, Axis::ZERO, d_density, d_density_prev, dataModel.visc, dataModel.dt);

        std::swap(d_density_prev, d_density);
        advect(dataModel, Axis::ZERO, d_density, d_density_prev, d_vX, d_vY, dataModel.dt);
    }

    fadeDensity(dataModel, d_density);

    // Copy results back to host
    copyToHost(dataModel);
}

void CUDAFluidStrategy::addDensity(FluidDataModel &dataModel, const uint32_t x, const uint32_t y, const double amount) {
    // Initialize CUDA memory if not already done
    if (d_density == nullptr) {
        init(dataModel);
        copyToDevice(dataModel);
    }

    if (x < 0 || x >= dataModel.size || y < 0 || y >= dataModel.size) return;

    // Add density directly on the device
    addDensity_kernel<<<1, 1>>>(d_density, x, y, dataModel.size, amount);
    gpuErrchk(cudaPeekAtLastError());

    // Update host value as well for consistency
    dataModel.density[FluidDataModel::index(y, x, dataModel.size)] += amount;
}

void CUDAFluidStrategy::addVelocity(FluidDataModel &dataModel, const uint32_t x, const uint32_t y, const double amountX, const double amountY) {
    // Initialize CUDA memory if not already done
    if (d_vX == nullptr) {
        init(dataModel);
        copyToDevice(dataModel);
    }

    if (x < 0 || x >= dataModel.size || y < 0 || y >= dataModel.size) return;

    // Add velocity directly on the device
    addVelocity_kernel<<<1, 1>>>(d_vX, d_vY, x, y, dataModel.size, amountX, amountY);
    gpuErrchk(cudaPeekAtLastError());

    // Update host values as well for consistency
    uint32_t idx = FluidDataModel::index(y, x, dataModel.size);
    dataModel.vX[idx] += amountY;
    dataModel.vY[idx] += amountX;
}

void CUDAFluidStrategy::reset(FluidDataModel &dataModel) {
    std::ranges::fill(dataModel.density, 0);
    std::ranges::fill(dataModel.density_prev, 0);
    std::ranges::fill(dataModel.vX, 0);
    std::ranges::fill(dataModel.vY, 0);
    std::ranges::fill(dataModel.vX_prev, 0);
    std::ranges::fill(dataModel.vY_prev, 0);
    std::ranges::fill(dataModel.vorticity, 0);

    // Reset device memory if it's been allocated
    if (d_density != nullptr) {
        const size_t size_bytes = dataModel.size * dataModel.size * sizeof(double);
        gpuErrchk(cudaMemset(d_density, 0, size_bytes));
        gpuErrchk(cudaMemset(d_density_prev, 0, size_bytes));
        gpuErrchk(cudaMemset(d_vX, 0, size_bytes));
        gpuErrchk(cudaMemset(d_vX_prev, 0, size_bytes));
        gpuErrchk(cudaMemset(d_vY, 0, size_bytes));
        gpuErrchk(cudaMemset(d_vY_prev, 0, size_bytes));
    }
}

void CUDAFluidStrategy::diffuse(const FluidDataModel &dataModel, const Axis mode, double *current, const double *previous, const double diffusion, const double dt) {
    double diffusionRate = dt * diffusion * (dataModel.size - 2) * (dataModel.size - 2);
    double cRecip = 1.0 / (1 + 4 * diffusionRate);

    lin_solve(dataModel, mode, current, previous, diffusionRate, cRecip);
}

void CUDAFluidStrategy::advect(const FluidDataModel &dataModel, const Axis mode, double *d, const double *d0, const double *vX, const double *vY, const double dt) const {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((dataModel.size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (dataModel.size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    advect_kernel<<<numBlocks, threadsPerBlock>>>(dataModel.size, d, d0, vX, vY, dt);
    gpuErrchk(cudaPeekAtLastError());

    set_bnd(dataModel, mode, d);
}

void CUDAFluidStrategy::project(const FluidDataModel &dataModel, double *vX, double *vY, double *p, double *div)  {
    const size_t size_bytes = dataModel.size * dataModel.size * sizeof(double);
    gpuErrchk(cudaMemset(p, 0, size_bytes));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((dataModel.size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (dataModel.size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    project_kernel<<<numBlocks, threadsPerBlock>>>(dataModel.size, vX, vY, div);
    gpuErrchk(cudaPeekAtLastError());

    set_bnd(dataModel, Axis::ZERO, div);
    set_bnd(dataModel, Axis::ZERO, p);

    double cRecip = 1.0 / 4;
    for (int k = 0; k < JACOBI_ITERATIONS; k++) {
        lin_solve_kernel<<<numBlocks, threadsPerBlock>>>(dataModel.size, p, div, d_newValue, 1.0, cRecip);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        std::swap(p, d_newValue);
        set_bnd(dataModel, Axis::ZERO, p);
    }

    update_velocity_kernel<<<numBlocks, threadsPerBlock>>>(dataModel.size, vX, vY, p);
    gpuErrchk(cudaPeekAtLastError());

    set_bnd(dataModel, Axis::X, vX);
    set_bnd(dataModel, Axis::Y, vY);
}

void CUDAFluidStrategy::set_bnd(const FluidDataModel &dataModel, const Axis mode, double *attr) const {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((dataModel.size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (dataModel.size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(mode, attr, dataModel.size);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

void CUDAFluidStrategy::lin_solve(const FluidDataModel &dataModel, const Axis mode, double *value, const double *oldValue, const double diffusionRate, const double cRecip) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((dataModel.size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (dataModel.size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    for (int k = 0; k < JACOBI_ITERATIONS; k++) {
        lin_solve_kernel<<<numBlocks, threadsPerBlock>>>(dataModel.size, value, oldValue, d_newValue,
                                                        diffusionRate, cRecip);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        std::swap(value, d_newValue);
        set_bnd(dataModel, mode, value);
    }
}

void CUDAFluidStrategy::fadeDensity(const FluidDataModel &dataModel, double *density) const {
    dim3 threadsPerBlock(256);
    dim3 numBlocks((dataModel.size * dataModel.size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    fade_density_kernel<<<numBlocks, threadsPerBlock>>>(dataModel.size, density);
    gpuErrchk(cudaPeekAtLastError());
}
