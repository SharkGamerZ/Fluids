#pragma once

#include "fluid_data_model.hpp"
#include "fluid_simulation_strategy.hpp"
#include <memory>

// Fluid Simulation Context - manages the current strategy and data
class FluidSimulation {
    std::unique_ptr<FluidDataModel> dataModel;
    std::unique_ptr<FluidSimulationStrategy> strategy;

public:
    FluidSimulation(uint32_t size, double diffusion, double viscosity, double dt, std::unique_ptr<FluidSimulationStrategy> initialStrategy);

    // Switch strategy method
    void setStrategy(std::unique_ptr<FluidSimulationStrategy> newStrategy);

    // Delegate methods to current strategy
    void step() const;

    void addDensity(uint32_t x, uint32_t y, double amount) const;

    void addVelocity(uint32_t x, uint32_t y, double amountX, double amountY) const;

    void reset() const;

    // Accessor for the underlying data model (if needed)
    [[nodiscard]] FluidDataModel &getDataModel() const;
};
