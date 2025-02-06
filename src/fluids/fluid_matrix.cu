#include "fluid_matrix.hpp"

__global__ void diffusionKernel(double *current, double *previous, double diffusion, double dt, int size) {
    // TODO
}

__global__ void advectionKernel(double *d, double *d0, double *vX, double *vY, double dt, int size) {
    // TODO
}

void FluidMatrix::CUDA_step() {}

void FluidMatrix::CUDA_diffuse(Axis mode, std::vector<double> &current, std::vector<double> &previous, double diffusion, double dt) const {}

void FluidMatrix::CUDA_advect(Axis mode, std::vector<double> &d, std::vector<double> &d0, std::vector<double> &vX, std::vector<double> &vY, double dt) const {}

void FluidMatrix::CUDA_project(std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &p, std::vector<double> &div) const {}

void FluidMatrix::CUDA_set_bnd(Axis mode, std::vector<double> &attr) const {}

void FluidMatrix::CUDA_lin_solve(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusionRate) const {}

void FluidMatrix::CUDA_fadeDensity(std::vector<double> &density) const {}
