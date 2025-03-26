#pragma once

#include "fluid_simulation_strategy.hpp"
#include <vector>

#ifdef ENABLE_TESTING
class TestSerialFluidStrategy;
#endif

class SerialFluidStrategy final : public FluidSimulationStrategy {
public:
    SerialFluidStrategy() = default;
    ~SerialFluidStrategy() override = default;

    void step(FluidDataModel &dataModel) override;
    void addDensity(FluidDataModel &dataModel, uint32_t x, uint32_t y, double amount) override;
    void addVelocity(FluidDataModel &dataModel, uint32_t x, uint32_t y, double amountX, double amountY) override;
    void reset(FluidDataModel &dataModel) override;

private:
#ifdef ENABLE_TESTING
    friend class TestSerialFluidStrategy;
#endif

    void diffuse(const FluidDataModel &dataModel, Axis mode, std::vector<double> &current, const std::vector<double> &previous, double diffusion, double dt) const;
    void advect(const FluidDataModel &dataModel, Axis mode, std::vector<double> &d, const std::vector<double> &d0, const std::vector<double> &vX, const std::vector<double> &vY, double dt) const;
    void project(const FluidDataModel &dataModel, std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &p, std::vector<double> &div) const;
    void set_bnd(const FluidDataModel &dataModel, Axis mode, std::vector<double> &attr) const;
    void gauss_lin_solve(const FluidDataModel &dataModel, Axis mode, std::vector<double> &value, const std::vector<double> &oldValue, double diffusionRate) const;
    void jacobi_lin_solve(const FluidDataModel &dataModel, Axis mode, std::vector<double> &value, const std::vector<double> &oldValue, double diffusionRate) const;
    void fadeDensity(const FluidDataModel &dataModel, std::vector<double> &density) const;
    void CalculateVorticity(const FluidDataModel &dataModel, const std::vector<double> &vX, const std::vector<double> &vY, std::vector<double> &vorticity) const;
};
