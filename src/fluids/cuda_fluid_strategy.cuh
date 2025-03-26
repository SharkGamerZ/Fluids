#pragma once

#include "fluid_simulation_strategy.hpp"

#ifdef ENABLE_TESTING
class TestCUDAFluidStrategy;
#endif

class CUDAFluidStrategy final : public FluidSimulationStrategy {
public:
    CUDAFluidStrategy();
    ~CUDAFluidStrategy() override;

    void step(FluidDataModel &dataModel) override;
    void addDensity(FluidDataModel &dataModel, uint32_t x, uint32_t y, double amount) override;
    void addVelocity(FluidDataModel &dataModel, uint32_t x, uint32_t y, double amountX, double amountY) override;
    void reset(FluidDataModel &dataModel) override;

    // CUDA device pointers
    double *d_density = nullptr;
    double *d_density_prev = nullptr;
    double *d_vX = nullptr;
    double *d_vX_prev = nullptr;
    double *d_vY = nullptr;
    double *d_vY_prev = nullptr;
    double *d_newValue = nullptr;

private:
#ifdef ENABLE_TESTING
    friend class TestCUDAFluidStrategy;
#endif

    void init(const FluidDataModel &dataModel);
    void destroy();
    void copyToDevice(const FluidDataModel &dataModel) const;
    void copyToHost(FluidDataModel &dataModel) const;

    void diffuse(const FluidDataModel &dataModel, Axis mode, double *current, const double *previous, double diffusion, double dt);
    void advect(const FluidDataModel &dataModel, Axis mode, double *d, const double *d0, const double *vX, const double *vY, double dt) const;
    void project(const FluidDataModel &dataModel, double *vX, double *vY, double *p, double *div);
    void set_bnd(const FluidDataModel &dataModel, Axis mode, double *attr) const;
    void lin_solve(const FluidDataModel &dataModel, Axis mode, double *value, const double *oldValue, double diffusionRate, double cRecip);
    void fadeDensity(const FluidDataModel &dataModel, double *density) const;
};
