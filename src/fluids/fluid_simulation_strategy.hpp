#pragma once

#include <cstdint>

class FluidDataModel;

enum class  Axis { X, Y, ZERO };

// Abstract strategy interface
class FluidSimulationStrategy {
public:
    virtual ~FluidSimulationStrategy() = default;

#ifndef ENABLE_TESTING
    static constexpr int GAUSS_ITERATIONS = 15;
    static constexpr int JACOBI_ITERATIONS = 20;
#else
    inline static int GAUSS_ITERATIONS = 15;
    inline static int JACOBI_ITERATIONS = 20;
#endif

    // Core simulation steps
    virtual void step(FluidDataModel &dataModel) = 0;
    virtual void addDensity(FluidDataModel &dataModel, uint32_t x, uint32_t y, double amount) = 0;
    virtual void addVelocity(FluidDataModel &dataModel, uint32_t x, uint32_t y, double amountX, double amountY) = 0;
    virtual void reset(FluidDataModel &dataModel) = 0;
};
