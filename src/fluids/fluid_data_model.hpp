#pragma once

#include <cstdint>
#include <vector>

// Shared data model to be used across all strategies
class FluidDataModel {
public:
    int size;                         ///< Size of the fluid matrix
    double dt;                        ///< Delta time
    double diff;                      ///< Diffusion
    double visc;                      ///< Viscosity
    std::vector<double> density;      ///< Density of the fluid
    std::vector<double> density_prev; ///< Density of the fluid in the previous step
    std::vector<double> vX;           ///< Velocity in the x-axis
    std::vector<double> vY;           ///< Velocity in the y-axis
    std::vector<double> vX_prev;      ///< Velocity in the x-axis in the previous step
    std::vector<double> vY_prev;      ///< Velocity in the y-axis in the previous step
    std::vector<double> vorticity;    ///< Vorticity of the fluid

    // Utility method
    static uint32_t index(const uint32_t i, const uint32_t j, const uint32_t matrix_size) { return j + i * matrix_size; }

    // Constructor
    FluidDataModel(const uint32_t size, const double diffusion, const double viscosity, const double dt)
        : size(size), dt(dt), diff(diffusion), visc(viscosity), density(size * size), density_prev(size * size), vX(size * size), vY(size * size), vX_prev(size * size), vY_prev(size * size),
          vorticity(size * size) {}
};
