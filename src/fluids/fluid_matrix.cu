#include "fluid_matrix.hpp"

#define gpuErrchk(ans)                                                                                                                                                             \
    {                                                                                                                                                                              \
        gpuAssert((ans), __FILE__, __LINE__);                                                                                                                                      \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
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
}

void FluidMatrix::CUDA_destroy() const {
    gpuErrchk(cudaFree(d_density));
    gpuErrchk(cudaFree(d_density_prev));
    gpuErrchk(cudaFree(d_vX));
    gpuErrchk(cudaFree(d_vX_prev));
    gpuErrchk(cudaFree(d_vY));
    gpuErrchk(cudaFree(d_vY_prev));
}

__device__ int index(const int i, const int j, const int size) { return i * size + j; }

__global__ void advect_kernel(int size, double *d, const double *d0, const double *vX, const double *vY, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < 1 || i >= size - 1 || j < 1 || j >= size - 1) return;

    double dt0 = dt * (size - 2);
    double N_double = size - 2;

    double v1 = vX[index(i, j, size)];
    double v2 = vY[index(i, j, size)];

    double x = i - dt0 * v1;
    double y = j - dt0 * v2;

    if (x < 0.5) x = 0.5;
    if (x > N_double + 0.5) x = N_double + 0.5;
    if (y < 0.5) y = 0.5;
    if (y > N_double + 0.5) y = N_double + 0.5;

    int i0 = floor(x);
    int i1 = i0 + 1;
    int j0 = floor(y);
    int j1 = j0 + 1;

    double s1 = x - i0;
    double s0 = 1 - s1;
    double t1 = y - j0;
    double t0 = 1 - t1;

    d[index(i, j, size)] = s0 * (t0 * d0[index(i0, j0, size)] + t1 * d0[index(i0, j1, size)]) + s1 * (t0 * d0[index(i1, j0, size)] + t1 * d0[index(i1, j1, size)]);
}

__global__ void project_kernel(int size, double *vX, double *vY, double *div) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 1 || i >= size - 1 || j < 1 || j >= size - 1) return;

    div[index(i, j, size)] = -0.5 * (vX[index(i + 1, j, size)] - vX[index(i - 1, j, size)] + vY[index(i, j + 1, size)] - vY[index(i, j - 1, size)]) / size;
}

__global__ void update_velocity_kernel(int size, double *vX, double *vY, double *p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 1 || i >= size - 1 || j < 1 || j >= size - 1) return;

    vX[index(i, j, size)] -= 0.5 * (p[index(i + 1, j, size)] - p[index(i - 1, j, size)]) * size;
    vY[index(i, j, size)] -= 0.5 * (p[index(i, j + 1, size)] - p[index(i, j - 1, size)]) * size;
}

__global__ void set_bnd_edges(int size, Axis mode, double *attr) {
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

__global__ void set_bnd_corners(int size, double *attr) {
    attr[index(0, 0, size)] = 0.5f * (attr[index(1, 0, size)] + attr[index(0, 1, size)]);
    attr[index(0, size - 1, size)] = 0.5f * (attr[index(1, size - 1, size)] + attr[index(0, size - 2, size)]);
    attr[index(size - 1, 0, size)] = 0.5f * (attr[index(size - 2, 0, size)] + attr[index(size - 1, 1, size)]);
    attr[index(size - 1, size - 1, size)] = 0.5f * (attr[index(size - 2, size - 1, size)] + attr[index(size - 1, size - 2, size)]);
}

__global__ void lin_solve_kernel(int size, double *value, const double *oldValue, double diffusionRate, double cRecip) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < 1 || i >= size - 1 || j < 1 || j >= size - 1) return;

    value[index(i, j, size)] = (oldValue[index(i, j, size)] +
                                diffusionRate * (value[index(i - 1, j, size)] + value[index(i + 1, j, size)] + value[index(i, j - 1, size)] + value[index(i, j + 1, size)])) *
                               cRecip;
}

__global__ void fade_density_kernel(int size, double *density) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 0 || i >= size * size) return;

    double d = density[i];
    density[i] = (d - 0.005f < 0) ? 0 : d - 0.005f;
}

void FluidMatrix::CUDA_step() {
    // Velocity
    {
        SWAP(vX_prev, vX);
        CUDA_diffuse(X, vX, vX_prev, visc, dt);

        SWAP(vY_prev, vY);
        CUDA_diffuse(Y, vY, vY_prev, visc, dt);

        CUDA_project(vX, vY, vX_prev, vY_prev);

        SWAP(vX_prev, vX);
        SWAP(vY_prev, vY);
        CUDA_advect(X, vX, vX_prev, vX_prev, vY_prev, dt);
        CUDA_advect(Y, vY, vY_prev, vX_prev, vY_prev, dt);

        CUDA_project(vX, vY, vX_prev, vY_prev);
    }

    // Density
    {
        SWAP(density_prev, density);
        CUDA_diffuse(ZERO, density, density_prev, visc, dt);

        SWAP(density_prev, density);
        CUDA_advect(ZERO, density, density_prev, vX, vY, dt);
    }

    CUDA_fadeDensity(density);

    CalculateVorticity(vX, vY, vorticity);
}

void FluidMatrix::CUDA_diffuse(Axis mode, std::vector<double> &current, std::vector<double> &previous, double diffusion, double dt) const {
    double diffusionRate = dt * diffusion * (this->size - 2) * (this->size - 2);
    CUDA_lin_solve(mode, current, previous, diffusionRate);
}

void FluidMatrix::CUDA_advect(Axis mode, std::vector<double> &d, std::vector<double> &d0, std::vector<double> &vX, std::vector<double> &vY, double dt) const {
    const size_t size_bytes = this->size * this->size * sizeof(double);

    double *dp_d, *dp_d0, *dp_vX, *dp_vY;

    switch (mode) {
        case X:
            dp_d = d_vX;
            dp_d0 = d_vX_prev;
            dp_vX = d_vX_prev;
            dp_vY = d_vY_prev;
            break;
        case Y:
            dp_d = d_vY;
            dp_d0 = d_vY_prev;
            dp_vX = d_vX_prev;
            dp_vY = d_vY_prev;
            break;
        case ZERO:
            dp_d = d_density;
            dp_d0 = d_density_prev;
            dp_vX = d_vX;
            dp_vY = d_vY;
            break;
        default: printf("Invalid mode\n"); return;
    }

    gpuErrchk(cudaMemcpy(dp_d0, d0.data(), size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dp_vX, vX.data(), size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dp_vY, vY.data(), size_bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((this->size + threadsPerBlock.x - 1) / threadsPerBlock.x, (this->size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    advect_kernel<<<numBlocks, threadsPerBlock>>>(this->size, dp_d, dp_d0, dp_vX, dp_vY, dt);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaMemcpy(d.data(), dp_d, size_bytes, cudaMemcpyDeviceToHost));

    // CUDA_set_bnd(mode, d);
}

void FluidMatrix::CUDA_project(std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &p, std::vector<double> &div) const {
    const size_t size_bytes = this->size * this->size * sizeof(double);

    gpuErrchk(cudaMemcpy(d_vX, vX.data(), size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_vY, vY.data(), size_bytes, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemset(d_vX_prev, 0, size_bytes));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((this->size + threadsPerBlock.x - 1) / threadsPerBlock.x, (this->size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    project_kernel<<<numBlocks, threadsPerBlock>>>(this->size, d_vX, d_vY, d_vY_prev);
    gpuErrchk(cudaPeekAtLastError());

    // CUDA_set_bnd(ZERO, div);
    // CUDA_set_bnd(ZERO, p);
    CUDA_lin_solve(ZERO, p, div, 1);

    update_velocity_kernel<<<numBlocks, threadsPerBlock>>>(this->size, d_vX, d_vY, d_vX_prev);
    gpuErrchk(cudaPeekAtLastError());

    // CUDA_set_bnd(X, vX);
    // CUDA_set_bnd(Y, vY);

    gpuErrchk(cudaMemcpy(vX.data(), d_vX, size_bytes, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(vY.data(), d_vY, size_bytes, cudaMemcpyDeviceToHost));
}

void FluidMatrix::CUDA_set_bnd(Axis mode, std::vector<double> &attr) const {
    const size_t size_bytes = this->size * this->size * sizeof(double);

    gpuErrchk(cudaMemcpy(d_density, attr.data(), size_bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((this->size + threadsPerBlock.x - 1) / threadsPerBlock.x, (this->size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    set_bnd_edges<<<numBlocks, threadsPerBlock>>>(this->size, mode, d_density);
    gpuErrchk(cudaPeekAtLastError());
    set_bnd_corners<<<1, 1>>>(this->size, d_density);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaMemcpy(attr.data(), d_density, size_bytes, cudaMemcpyDeviceToHost));
}

void FluidMatrix::CUDA_lin_solve(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusionRate) const {

    const size_t size_bytes = this->size * this->size * sizeof(double);

    double c = diffusionRate;
    double cRecip = 1.0 / (1 + 4 * c);

    double *d_value, *d_oldValue;
    switch (mode) {
        case X:
            d_value = d_vX;
            d_oldValue = d_vX_prev;
            break;
        case Y:
            d_value = d_vY;
            d_oldValue = d_vY_prev;
            break;
        case ZERO:
            d_value = d_density;
            d_oldValue = d_density_prev;
            break;
        default: printf("Invalid mode\n"); return;
    }

    gpuErrchk(cudaMemcpy(d_value, value.data(), size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_oldValue, oldValue.data(), size_bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((this->size + threadsPerBlock.x - 1) / threadsPerBlock.x, (this->size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("CUDA diffusionRate: %f, cRecip: %f\n", diffusionRate, cRecip);
    printf("pre-CUDA Value[2][2]: %f\n", value[index(2, 2, this->size)]);
    printf("pre-CUDA Value[2][2-1]: %f\n", value[index(2, 1, this->size)]);
    printf("pre-CUDA Value[2][2+1]: %f\n", value[index(2, 3, this->size)]);
    printf("pre-CUDA Value[2-1][2]: %f\n", value[index(1, 2, this->size)]);
    printf("pre-CUDA Value[2+1][2]: %f\n", value[index(3, 2, this->size)]);

    for (int k = 0; k < ITERATIONS; k++) {
        lin_solve_kernel<<<numBlocks, threadsPerBlock>>>(this->size, d_value, d_oldValue, diffusionRate, cRecip);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        //CUDA_set_bnd(mode, value);
    }

    gpuErrchk(cudaMemcpy(value.data(), d_value, size_bytes, cudaMemcpyDeviceToHost));
    printf("post-CUDA Value[2][2]: %f\n", value[index(2, 2, this->size)]);
    printf("post-CUDA Value[2][2-1]: %f\n", value[index(2, 1, this->size)]);
    printf("post-CUDA Value[2][2+1]: %f\n", value[index(2, 3, this->size)]);
    printf("post-CUDA Value[2-1][2]: %f\n", value[index(1, 2, this->size)]);
    printf("post-CUDA Value[2+1][2]: %f\n", value[index(3, 2, this->size)]);
}

void FluidMatrix::CUDA_fadeDensity(std::vector<double> &density) const {
    const size_t size_bytes = this->size * this->size * sizeof(double);

    gpuErrchk(cudaMemcpy(d_density, density.data(), size_bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((this->size + threadsPerBlock.x - 1) / threadsPerBlock.x, (this->size + threadsPerBlock.y - 1) / threadsPerBlock.y);

    fade_density_kernel<<<numBlocks, threadsPerBlock>>>(this->size, d_density);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaMemcpy(density.data(), d_density, size_bytes, cudaMemcpyDeviceToHost));
}
