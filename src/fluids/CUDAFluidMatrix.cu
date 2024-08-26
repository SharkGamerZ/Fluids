#include "CUDAFluidMatrix.cuh"

__host__ void FluidMatrix_CUDA_step(FluidMatrix *matrix) {
    // Define grid and block dimensions
    dim3 BlockSize(16, 16, 1);
    dim3 GridSize((matrix->size + 15) / 16, (matrix->size + 15) / 16, 1);

    // velocity step
    {
        FluidMatrix_CUDA_diffuse(matrix, Axis::X, matrix->Vx0, matrix->Vx, matrix->visc, matrix->dt);
        FluidMatrix_CUDA_diffuse(matrix, Axis::Y, matrix->Vy0, matrix->Vy, matrix->visc, matrix->dt);

        FluidMatrix_CUDA_project<<<GridSize, BlockSize>>>(matrix, matrix->Vx0, matrix->Vy0, matrix->Vx, matrix->Vy);


        FluidMatrix_CUDA_advect<<<GridSize, BlockSize>>>(matrix, Axis::X, matrix->Vx, matrix->Vx0, matrix->Vx0, matrix->Vy0, matrix->dt);
        FluidMatrix_CUDA_advect<<<GridSize, BlockSize>>>(matrix, Axis::Y, matrix->Vy, matrix->Vy0, matrix->Vx0, matrix->Vy0, matrix->dt);

        FluidMatrix_CUDA_project<<<GridSize, BlockSize>>>(matrix, matrix->Vx, matrix->Vy, matrix->Vx0, matrix->Vy0);
    }

    // density step
    {
        FluidMatrix_CUDA_diffuse(matrix, Axis::ZERO, matrix->density0, matrix->density, matrix->diff, matrix->dt);

        FluidMatrix_CUDA_advect<<<GridSize, BlockSize>>>(matrix, Axis::ZERO, matrix->density, matrix->density0, matrix->Vx, matrix->Vy, matrix->dt);
    }

    FluidMatrix_CUDA_fade_density<<<GridSize, BlockSize>>>(matrix, matrix->density);
}

// Private methods

__host__ void FluidMatrix_CUDA_diffuse(FluidMatrix *matrix, enum Axis mode, double *value, double *oldValue, double diffusion, double dt) {
    double diffusionRate = dt * diffusion * matrix->size * matrix->size;

    dim3 BlockSize(16, 16, 1);
    dim3 GridSize((matrix->size + 15) / 16, (matrix->size + 15) / 16, 1);


    double *d_value;
    double *d_oldValue;

    cudaMalloc(&d_value, matrix->size * matrix->size * sizeof(double));
    cudaMalloc(&d_oldValue, matrix->size * matrix->size * sizeof(double));

    // TODO: memcpy requires __host__, check if correct
    cudaMemcpy(d_value, &value[0], matrix->size * matrix->size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldValue, &oldValue[0], matrix->size * matrix->size * sizeof(double), cudaMemcpyHostToDevice);


    for (int i = 0; i < ITERATIONS; i++) {
        FluidMatrix_CUDA_lin_solve<<<GridSize, BlockSize>>>(matrix, mode, &d_value[0], &d_oldValue[0], diffusionRate);
        FluidMatrix_CUDA_set_bnd<<<1, 1>>>(matrix, mode, &d_value[0]);
    }

    cudaMemcpy(&value[0], d_value, matrix->size * matrix->size * sizeof(double), cudaMemcpyDeviceToHost);


    cudaFree(d_value);
    cudaFree(d_oldValue);
}

__global__ void FluidMatrix_CUDA_advect(FluidMatrix *matrix, enum Axis mode, double *value, double *oldValue, double *Vx, double *Vy, double dt) {
    // TODO
}

__global__ void FluidMatrix_CUDA_project(FluidMatrix *matrix, double *Vx, double *Vy, double *p, double *div) {
    // TODO
}

__global__ void FluidMatrix_CUDA_set_bnd(FluidMatrix *matrix, enum Axis mode, double *value) {
    for (int i = 1; i < matrix->size - 1; i++) {
        value[FluidMatrix_CUDA_index(i, 0, matrix->size)] =
                mode == Axis::Y ? -value[FluidMatrix_CUDA_index(i, 1, matrix->size)] : value[FluidMatrix_CUDA_index(i, 1, matrix->size)];
        value[FluidMatrix_CUDA_index(i, matrix->size - 1, matrix->size)] =
                mode == Axis::Y ? -value[FluidMatrix_CUDA_index(i, matrix->size - 2, matrix->size)] : value[FluidMatrix_CUDA_index(i, matrix->size - 2, matrix->size)];
    }
    for (int j = 1; j < matrix->size - 1; j++) {
        value[FluidMatrix_CUDA_index(0, j, matrix->size)] =
                mode == Axis::X ? -value[FluidMatrix_CUDA_index(1, j, matrix->size)] : value[FluidMatrix_CUDA_index(1, j, matrix->size)];
        value[FluidMatrix_CUDA_index(matrix->size - 1, j, matrix->size)] =
                mode == Axis::X ? -value[FluidMatrix_CUDA_index(matrix->size - 2, j, matrix->size)] : value[FluidMatrix_CUDA_index(matrix->size - 2, j, matrix->size)];
    }


    value[FluidMatrix_CUDA_index(0, 0, matrix->size)] =
            0.5f * (value[FluidMatrix_CUDA_index(1, 0, matrix->size)] +
                    value[FluidMatrix_CUDA_index(0, 1, matrix->size)]);
    value[FluidMatrix_CUDA_index(0, matrix->size - 1, matrix->size)] =
            0.5f * (value[FluidMatrix_CUDA_index(1, matrix->size - 1, matrix->size)] +
                    value[FluidMatrix_CUDA_index(0, matrix->size - 2, matrix->size)]);

    value[FluidMatrix_CUDA_index(matrix->size - 1, 0, matrix->size)] =
            0.5f * (value[FluidMatrix_CUDA_index(matrix->size - 2, 0, matrix->size)] +
                    value[FluidMatrix_CUDA_index(matrix->size - 1, 1, matrix->size)]);
    value[FluidMatrix_CUDA_index(matrix->size - 1, matrix->size - 1, matrix->size)] =
            0.5f * (value[FluidMatrix_CUDA_index(matrix->size - 2, matrix->size - 1, matrix->size)] +
                    value[FluidMatrix_CUDA_index(matrix->size - 1, matrix->size - 2, matrix->size)]);
}

__global__ void FluidMatrix_CUDA_lin_solve(FluidMatrix *matrix, enum Axis mode, double *nextValue, double *value, double diffusionRate) {
    double c = 1 + 4 * diffusionRate;
    double cRecip = 1.0 / c;

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col == 0 || col >= matrix->size - 1 || row == 0 || row >= matrix->size - 1) return;


    nextValue[FluidMatrix_CUDA_index(row, col, matrix->size)] =
            (value[FluidMatrix_CUDA_index(row, col, matrix->size)] + diffusionRate * (
                                                                                              nextValue[FluidMatrix_CUDA_index(row + 1, col, matrix->size)] +
                                                                                              nextValue[FluidMatrix_CUDA_index(row - 1, col, matrix->size)] +
                                                                                              nextValue[FluidMatrix_CUDA_index(row, col + 1, matrix->size)] +
                                                                                              nextValue[FluidMatrix_CUDA_index(row, col - 1, matrix->size)])
            ) * cRecip;

    // __syncthreads();
    // if (col == 1 && row == 1)
    //     kernel_set_bnd(matrix->size, mode, nextValue);
    // __syncthreads();
}

__global__ void FluidMatrix_CUDA_fade_density(FluidMatrix *matrix, double *density) {

}
