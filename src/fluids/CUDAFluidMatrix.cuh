#pragma once

#include "FluidMatrix.h"

__host__ __device__ inline unsigned int FluidMatrix_CUDA_index(unsigned int i, unsigned int j, unsigned int matrix_size) {
    return CFluidMatrix_index(i, j, matrix_size); // Call the original index function
}

__host__ void FluidMatrix_CUDA_step(FluidMatrix *matrix);

// Private methods

__host__ void FluidMatrix_CUDA_diffuse(FluidMatrix *matrix, enum Axis mode, double *value, double *oldValue, double diffusion, double dt);

__global__ void FluidMatrix_CUDA_advect(FluidMatrix *matrix, enum Axis mode, double *value, double *oldValue, double *Vx, double *Vy, double dt);

__global__ void FluidMatrix_CUDA_project(FluidMatrix *matrix, double *Vx, double *Vy, double *p, double *div);

__global__ void FluidMatrix_CUDA_set_bnd(FluidMatrix *matrix, enum Axis mode, double *value);

__global__ void FluidMatrix_CUDA_lin_solve(FluidMatrix *matrix, enum Axis mode, double *nextValue, double *value, double diffusionRate);

__global__ void FluidMatrix_CUDA_fade_density(FluidMatrix *matrix, double *density);
