#pragma once

#include "../src/fluids/openmp_fluid_strategy.hpp"

class TestOpenMPFluidStrategy {
public:
    static void test_gauss_lin_solve(const OpenMPFluidStrategy &strategy, FluidDataModel &dataModel) {
        strategy.gauss_lin_solve(dataModel, Axis::X, dataModel.density, dataModel.density_prev, dataModel.visc);
    }

    static void test_jacobi_lin_solve(const OpenMPFluidStrategy &strategy, FluidDataModel &dataModel) {
        strategy.jacobi_lin_solve(dataModel, Axis::X, dataModel.density, dataModel.density_prev, dataModel.visc);
    }
};
