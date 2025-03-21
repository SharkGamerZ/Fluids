#include "fluid_matrix.hpp"
#include <sstream>

#define gpuErrchk(ans) gpuAssert((ans), __FILE__, __LINE__);

inline void gpuAssert(const cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error: " << cudaGetErrorString(code) << " at " << file << ":" << line;
        const std::string err_msg = oss.str();

        std::cerr << err_msg << std::endl;
        cudaDeviceReset();
        throw std::runtime_error(err_msg);
    }
}

void FluidMatrix::CUDA_init() {
    const size_t size_bytes = this->size * this->size * sizeof(double);
    gpuErrchk(cudaMalloc(&d_density, size_bytes));
    gpuErrchk(cudaMalloc(&d_density_prev, size_bytes));
    gpuErrchk(cudaMalloc(&d_vX, size_bytes));
    gpuErrchk(cudaMalloc(&d_vX_prev, size_bytes));
    gpuErrchk(cudaMalloc(&d_vY, size_bytes));
    gpuErrchk(cudaMalloc(&d_vY_prev, size_bytes));
    gpuErrchk(cudaMalloc(&d_newValue, size_bytes));
}

void FluidMatrix::CUDA_destroy() const {
    gpuErrchk(cudaFree(d_density));
    gpuErrchk(cudaFree(d_density_prev));
    gpuErrchk(cudaFree(d_vX));
    gpuErrchk(cudaFree(d_vX_prev));
    gpuErrchk(cudaFree(d_vY));
    gpuErrchk(cudaFree(d_vY_prev));
    gpuErrchk(cudaFree(d_newValue));
}

void FluidMatrix::copyToHost() {
    const size_t size_bytes = this->size * this->size * sizeof(double);
    gpuErrchk(cudaMemcpyAsync(density.data(), d_density, size_bytes, cudaMemcpyDeviceToHost));
    // gpuErrchk(cudaMemcpy(density_prev.data(), d_density_prev, size_bytes, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpyAsync(vX.data(), d_vX, size_bytes, cudaMemcpyDeviceToHost));
    // gpuErrchk(cudaMemcpy(vX_prev.data(), d_vX_prev, size_bytes, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpyAsync(vY.data(), d_vY, size_bytes, cudaMemcpyDeviceToHost));
    // gpuErrchk(cudaMemcpy(vY_prev.data(), d_vY_prev, size_bytes, cudaMemcpyDeviceToHost));
}

void FluidMatrix::copyToDevice() const {
    const size_t size_bytes = this->size * this->size * sizeof(double);
    gpuErrchk(cudaMemcpy(d_density, density.data(), size_bytes, cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(d_density_prev, density_prev.data(), size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_vX, vX.data(), size_bytes, cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(d_vX_prev, vX_prev.data(), size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_vY, vY.data(), size_bytes, cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(d_vY_prev, vY_prev.data(), size_bytes, cudaMemcpyHostToDevice));
}


__device__ int index(const int i, const int j, const int size) { return i * size + j; }

__global__ void addVelocity_kernel(double *d_vX, double *d_vY, const int x, const int y, const int size, const double amountX, const double amountY) {
    d_vX[index(y, x, size)] += amountY;
    d_vY[index(y, x, size)] += amountX;
}

void FluidMatrix::CUDA_addVelocity(const int x, const int y, const double amountX, const double amountY) const {
    if (x < 0 || x >= size || y < 0 || y >= size) return;
    addVelocity_kernel<<<1, 1>>>(d_vX, d_vY, x, y, size, amountX, amountY);
}

__global__ void addDensity_kernel(double *d_density, const int x, const int y, const int size, const double amount) { d_density[index(y, x, size)] += amount; }

void FluidMatrix::CUDA_addDensity(const int x, const int y, const double amount) const {
    if (x < 0 || x >= size || y < 0 || y >= size) return;
    addDensity_kernel<<<1, 1>>>(d_density, x, y, size, amount);
}

void FluidMatrix::CUDA_reset() const {
    const size_t size_bytes = this->size * this->size * sizeof(double);
    gpuErrchk(cudaMemset(d_density, 0, size_bytes));
    gpuErrchk(cudaMemset(d_density_prev, 0, size_bytes));
    gpuErrchk(cudaMemset(d_vX, 0, size_bytes));
    gpuErrchk(cudaMemset(d_vX_prev, 0, size_bytes));
    gpuErrchk(cudaMemset(d_vY, 0, size_bytes));
    gpuErrchk(cudaMemset(d_vY_prev, 0, size_bytes));
}

__global__ void advect_kernel(const int size, double *d, const double *d0, const double *vX, const double *vY, const double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < 1 || i >= size - 1 || j < 1 || j >= size - 1) return;
    int i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1, dt0;

    dt0 = dt * (size - 2);

    x = i - dt0 * vX[index(i, j, size)];
    y = j - dt0 * vY[index(i, j, size)];
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
    d[index(i, j, size)] = s0 * (t0 * d0[index(i0, j0, size)] + t1 * d0[index(i0, j1, size)]) + s1 * (t0 * d0[index(i1, j0, size)] + t1 * d0[index(i1, j1, size)]);
}

__global__ void project_kernel(const int size, const double *vX, const double *vY, double *vY_prev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 1 || i >= size - 1 || j < 1 || j >= size - 1) return;

    vY_prev[index(i, j, size)] = -0.5 * (vX[index(i + 1, j, size)] - vX[index(i - 1, j, size)] + vY[index(i, j + 1, size)] - vY[index(i, j - 1, size)]) * (size - 2);
}

__global__ void update_velocity_kernel(const int size, double *vX, double *vY, const double *vX_prev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 1 || i >= size - 1 || j < 1 || j >= size - 1) return;

    vX[index(i, j, size)] -= 0.5 * (vX_prev[index(i + 1, j, size)] - vX_prev[index(i - 1, j, size)]) / (size - 2);
    vY[index(i, j, size)] -= 0.5 * (vX_prev[index(i, j + 1, size)] - vX_prev[index(i, j - 1, size)]) / (size - 2);
}

__global__ void set_bnd_edges(const int size, const Axis mode, double *attr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i < size - 1) {
        attr[index(i, 0, size)] = mode == Y ? -attr[index(i, 1, size)] : attr[index(i, 1, size)];
        attr[index(i, size - 1, size)] = mode == Y ? -attr[index(i, size - 2, size)] : attr[index(i, size - 2, size)];
    }

    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= 1 && j < size - 1) {
        attr[index(0, j, size)] = mode == X ? -attr[index(1, j, size)] : attr[index(1, j, size)];
        attr[index(size - 1, j, size)] = mode == X ? -attr[index(size - 2, j, size)] : attr[index(size - 2, j, size)];
    }
}

__global__ void set_bnd_corners(const int size, double *attr) {
    attr[index(0, 0, size)] = 0.5f * (attr[index(1, 0, size)] + attr[index(0, 1, size)]);
    attr[index(0, size - 1, size)] = 0.5f * (attr[index(1, size - 1, size)] + attr[index(0, size - 2, size)]);
    attr[index(size - 1, 0, size)] = 0.5f * (attr[index(size - 2, 0, size)] + attr[index(size - 1, 1, size)]);
    attr[index(size - 1, size - 1, size)] = 0.5f * (attr[index(size - 2, size - 1, size)] + attr[index(size - 1, size - 2, size)]);
}

__global__ void lin_solve_kernel(const int size, const double *d_value, const double *d_oldValue, double *d_newValue, const double diffusionRate, const double cRecip) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < 1 || i >= size - 1 || j < 1 || j >= size - 1) return;

    d_newValue[index(i, j, size)] = (d_oldValue[index(i, j, size)] + diffusionRate * (d_value[index(i + 1, j, size)] + d_value[index(i - 1, j, size)] +
                                                                                      d_value[index(i, j + 1, size)] + d_value[index(i, j - 1, size)])) *
                                    cRecip;
}

__global__ void fade_density_kernel(const int size, double *density) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 0 || i >= size * size) return;

    double d = density[i];
    density[i] = (d - 0.005f < 0) ? 0 : d - 0.005f;
}

void FluidMatrix::CUDA_step() {
    // Velocity
    {
        std::swap(d_vX_prev, d_vX);
        CUDA_diffuse(X, d_vX, d_vX_prev, visc, dt);

        std::swap(d_vY_prev, d_vY);
        CUDA_diffuse(Y, d_vY, d_vY_prev, visc, dt);

        CUDA_project(d_vX, d_vY, d_vX_prev, d_vY_prev);

        std::swap(d_vX_prev, d_vX);
        std::swap(d_vY_prev, d_vY);
        CUDA_advect(X, d_vX, d_vX_prev, d_vX_prev, d_vY_prev, dt);
        CUDA_advect(Y, d_vY, d_vY_prev, d_vX_prev, d_vY_prev, dt);

        CUDA_project(d_vX, d_vY, d_vX_prev, d_vY_prev);
    }

    // Density
    {
        std::swap(d_density_prev, d_density);
        CUDA_diffuse(ZERO, d_density, d_density_prev, visc, dt);

        std::swap(d_density_prev, d_density);
        CUDA_advect(ZERO, d_density, d_density_prev, d_vX, d_vY, dt);
    }

    CUDA_fadeDensity(d_density);

    CalculateVorticity(vX, vY, vorticity);
}

void FluidMatrix::CUDA_diffuse(const Axis mode, double *current, const double *previous, const double diffusion, const double dt) {
    double diffusionRate = dt * diffusion * (this->size - 2) * (this->size - 2);
    double cRecip = 1.0 / (1 + 4 * diffusionRate);
    CUDA_lin_solve(mode, current, previous, diffusionRate, cRecip);
}

void FluidMatrix::CUDA_advect(const Axis mode, double *d_density, const double *d_density0, const double *d_vX, const double *d_vY, const double dt) const {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((this->size + threadsPerBlock.x - 1) / threadsPerBlock.x, (this->size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    advect_kernel<<<numBlocks, threadsPerBlock>>>(this->size, d_density, d_density0, d_vX, d_vY, dt);
    gpuErrchk(cudaPeekAtLastError());

    CUDA_set_bnd(mode, d_density);
}

void FluidMatrix::CUDA_project(double *d_vX, double *d_vY, double *d_vX_prev, double *d_vY_prev) {
    const size_t size_bytes = this->size * this->size * sizeof(double);

    gpuErrchk(cudaMemset(d_vX_prev, 0, size_bytes));
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((this->size + threadsPerBlock.x - 1) / threadsPerBlock.x, (this->size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    project_kernel<<<numBlocks, threadsPerBlock>>>(this->size, d_vX, d_vY, d_vY_prev);
    gpuErrchk(cudaPeekAtLastError());

    CUDA_set_bnd(ZERO, d_vY_prev);
    CUDA_set_bnd(ZERO, d_vX_prev);
    // CUDA_lin_solve(ZERO, vX_prev, vY_prev, 1, 1.0 / 4);


    for (int k = 0; k < JACOBI_ITERATIONS; k++) {
        lin_solve_kernel<<<numBlocks, threadsPerBlock>>>(this->size, d_vX_prev, d_vY_prev, d_newValue, 1.0, 1.0 / 4);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        std::swap(d_vX_prev, d_newValue);
        CUDA_set_bnd(ZERO, d_vX_prev);
    }


    update_velocity_kernel<<<numBlocks, threadsPerBlock>>>(this->size, d_vX, d_vY, d_vX_prev);
    gpuErrchk(cudaPeekAtLastError());

    CUDA_set_bnd(X, d_vX);
    CUDA_set_bnd(Y, d_vY);
}

__global__ void CUDA_set_bnd_kernel(const Axis mode, double *d_value, const int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= size || j >= size) return;

    // Top boundary (j == size-1) and bottom boundary (j == 0)
    if (j == 0 && i > 0 && i < size - 1) {
        // For vertical component (mode Y), reflect; otherwise copy
        d_value[index(i, 0, size)] = (mode == Axis::Y) ? -d_value[index(i, 1, size)] : d_value[index(i, 1, size)];
    }
    if (j == size - 1 && i > 0 && i < size - 1) {
        d_value[index(i, size - 1, size)] = (mode == Axis::Y) ? -d_value[index(i, size - 2, size)] : d_value[index(i, size - 2, size)];
    }

    // Left boundary (i == 0) and right boundary (i == size-1)
    if (i == 0 && j > 0 && j < size - 1) {
        d_value[index(0, j, size)] = (mode == Axis::X) ? -d_value[index(1, j, size)] : d_value[index(1, j, size)];
    }
    if (i == size - 1 && j > 0 && j < size - 1) {
        d_value[index(size - 1, j, size)] = (mode == Axis::X) ? -d_value[index(size - 2, j, size)] : d_value[index(size - 2, j, size)];
    }

    // Corners
    if (i == 0 && j == 0) {
        d_value[index(0, 0, size)] = 0.5 * (d_value[index(1, 0, size)] + d_value[index(0, 1, size)]);
    }
    if (i == 0 && j == size - 1) {
        d_value[index(0, size - 1, size)] = 0.5 * (d_value[index(1, size - 1, size)] + d_value[index(0, size - 2, size)]);
    }
    if (i == size - 1 && j == 0) {
        d_value[index(size - 1, 0, size)] = 0.5 * (d_value[index(size - 2, 0, size)] + d_value[index(size - 1, 1, size)]);
    }
    if (i == size - 1 && j == size - 1) {
        d_value[index(size - 1, size - 1, size)] = 0.5 * (d_value[index(size - 2, size - 1, size)] + d_value[index(size - 1, size - 2, size)]);
    }
}

void FluidMatrix::CUDA_set_bnd(const Axis mode, double *d_value) const {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x, (size + threadsPerBlock.y - 1) / threadsPerBlock.y);
    CUDA_set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(mode, d_value, this->size);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

void FluidMatrix::CUDA_lin_solve(const Axis mode, double *d_value, const double *d_oldValue, const double diffusionRate, const double cRecip) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((this->size + threadsPerBlock.x - 1) / threadsPerBlock.x, (this->size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    for (int k = 0; k < JACOBI_ITERATIONS; k++) {
        lin_solve_kernel<<<numBlocks, threadsPerBlock>>>(this->size, d_value, d_oldValue, d_newValue, diffusionRate, cRecip);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        std::swap(d_value, d_newValue);
        CUDA_set_bnd(mode, d_value);
    }
}

void FluidMatrix::CUDA_fadeDensity(double *d_density) const {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((this->size + threadsPerBlock.x - 1) / threadsPerBlock.x, (this->size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    fade_density_kernel<<<numBlocks, threadsPerBlock>>>(this->size, d_density);
    gpuErrchk(cudaPeekAtLastError());
}
