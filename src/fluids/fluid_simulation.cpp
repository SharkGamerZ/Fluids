#include "fluid_simulation.hpp"

FluidSimulation::FluidSimulation(uint32_t size, double diffusion, double viscosity, double dt, std::unique_ptr<FluidSimulationStrategy> initialStrategy)
    : dataModel(std::make_unique<FluidDataModel>(size, diffusion, viscosity, dt)), strategy(std::move(initialStrategy)) {}

void FluidSimulation::setStrategy(std::unique_ptr<FluidSimulationStrategy> newStrategy) { strategy = std::move(newStrategy); }

void FluidSimulation::step() const { strategy->step(*dataModel); }

void FluidSimulation::addDensity(const uint32_t x, const uint32_t y, const double amount) const { strategy->addDensity(*dataModel, x, y, amount); }

void FluidSimulation::addVelocity(const uint32_t x, const uint32_t y, const double amountX, const double amountY) const { strategy->addVelocity(*dataModel, x, y, amountX, amountY); }

void FluidSimulation::reset() const { strategy->reset(*dataModel); }

FluidDataModel &FluidSimulation::getDataModel() const { return *dataModel; }
