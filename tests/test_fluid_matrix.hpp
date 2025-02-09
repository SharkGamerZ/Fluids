#pragma once

#include "../src/fluids/fluid_matrix.hpp"

/// TestFluidMatrix class to expose private members and methods for testing
class TestFluidMatrix : public FluidMatrix {
public:
    // Inherit constructors
    using FluidMatrix::FluidMatrix;

    // Expose private member
    using FluidMatrix::numMaxThreads;

    // Expose serial implementations
    using FluidMatrix::diffuse;
    using FluidMatrix::advect;
    using FluidMatrix::project;
    using FluidMatrix::set_bnd;
    using FluidMatrix::lin_solve;
    using FluidMatrix::fadeDensity;

    // Expose OpenMP implementations
    using FluidMatrix::OMP_diffuse;
    using FluidMatrix::OMP_advect;
    using FluidMatrix::OMP_project;
    using FluidMatrix::OMP_set_bnd;
    using FluidMatrix::OMP_lin_solve;
    using FluidMatrix::OMP_fadeDensity;

#ifdef __CUDACC__
    // Expose CUDA implementations
    using FluidMatrix::CUDA_diffuse;
    using FluidMatrix::CUDA_advect;
    using FluidMatrix::CUDA_project;
    using FluidMatrix::CUDA_set_bnd;
    using FluidMatrix::CUDA_lin_solve;
    using FluidMatrix::CUDA_fadeDensity;
#endif
};
