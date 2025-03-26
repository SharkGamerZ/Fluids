#pragma once

#include "../src/fluids/serial_fluid_strategy.hpp"

class TestSerialFluidStrategy {
public:
    static void test_gauss_lin_solve(const SerialFluidStrategy &strategy, FluidDataModel &dataModel) {
        strategy.gauss_lin_solve(dataModel, Axis::X, dataModel.density, dataModel.density_prev, dataModel.visc);
    }

    static void test_jacobi_lin_solve(const SerialFluidStrategy &strategy, FluidDataModel &dataModel) {
        strategy.jacobi_lin_solve(dataModel, Axis::X, dataModel.density, dataModel.density_prev, dataModel.visc);
    }
};
