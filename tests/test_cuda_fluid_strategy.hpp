#pragma once

#include "../src/fluids/cuda_fluid_strategy.cuh"

class TestCUDAFluidStrategy {
public:
    static void test_gauss_lin_solve(CUDAFluidStrategy &strategy, FluidDataModel &dataModel) {
        strategy.lin_solve(dataModel, Axis::X, strategy.d_density, strategy.d_density_prev, dataModel.visc, 1.0 / (1 + 4 * dataModel.visc));
    }

    // TODO: Implement test_jacobi_lin_solve
    static void test_jacobi_lin_solve(CUDAFluidStrategy &strategy, FluidDataModel &dataModel) {
        strategy.lin_solve(dataModel, Axis::X, strategy.d_density, strategy.d_density_prev, dataModel.visc, 1.0 / (1 + 4 * dataModel.visc));
    }
};
