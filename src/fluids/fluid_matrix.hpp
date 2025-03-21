#pragma once

#include "../utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <omp.h>
#include <vector>

extern int GAUSS_ITERATIONS;  ///< Number of iterations for the Gauss-Siedel
extern int JACOBI_ITERATIONS; ///< Number of iterations for the Jacobi
enum Axis { X, Y, ZERO };

/// FluidMatrix class for serial, OpenMP, and CUDA-based simulation
class FluidMatrix {
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

    std::vector<double> vorticity; ///< Vorticity of the fluid

#ifdef CUDA_SUPPORT
    double *d_density, *d_density_prev;
    double *d_vX, *d_vX_prev;
    double *d_vY, *d_vY_prev;
    double *d_newValue;
#endif

    FluidMatrix(uint32_t size, double diffusion, double viscosity, double dt);
    ~FluidMatrix();

    [[nodiscard]] static uint32_t index(const uint32_t i, const uint32_t j, const uint32_t matrix_size) { return j + i * matrix_size; }
    void reset();

    void step();
    void OMP_step();
#ifdef CUDA_SUPPORT
    void CUDA_step();

    void CUDA_reset() const;
    void copyToHost();
    void copyToDevice() const;
#endif

    void addDensity(uint32_t x, uint32_t y, double amount);
    void addVelocity(uint32_t x, uint32_t y, double amountX, double amountY);

#ifdef CUDA_SUPPORT
    void CUDA_addVelocity(int x, int y, double amountX, double amountY) const;
    void CUDA_addDensity(int x, int y, double amount) const;
#endif

protected:
    int numMaxThreads; ///< Number of threads used by OpenMP

#ifdef CUDA_SUPPORT
    /// Preallocate CUDA memory on matrix creation
    void CUDA_init();
    /// Free all CUDA memory on matrix destruction
    void CUDA_destroy() const;
#endif

    // Serial implementations

    void diffuse(Axis mode, std::vector<double> &current, const std::vector<double> &previous, double diffusion, double dt) const;
    void advect(Axis mode, std::vector<double> &d, const std::vector<double> &d0, const std::vector<double> &vX, const std::vector<double> &vY, double dt) const;
    void project(std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &p, std::vector<double> &div) const;
    void set_bnd(Axis mode, std::vector<double> &attr) const;
    void gauss_lin_solve(Axis mode, std::vector<double> &value, const std::vector<double> &oldValue, double diffusionRate) const;
    void jacobi_lin_solve(Axis mode, std::vector<double> &value, const std::vector<double> &oldValue, double diffusionRate) const;
    void fadeDensity(std::vector<double> &density) const;
    void CalculateVorticity(const std::vector<double> &vX, const std::vector<double> &vY, std::vector<double> &vorticity) const;

    // OpenMP implementations

    void OMP_diffuse(Axis mode, std::vector<double> &current, const std::vector<double> &previous, double diffusion, double dt) const;
    void OMP_advect(Axis mode, std::vector<double> &d, const std::vector<double> &d0, const std::vector<double> &vX, const std::vector<double> &vY, double dt) const;
    void OMP_project(std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &p, std::vector<double> &div) const;
    void OMP_set_bnd(Axis mode, std::vector<double> &attr) const;
    void OMP_gauss_lin_solve(Axis mode, std::vector<double> &value, const std::vector<double> &oldValue, double diffusionRate) const;
    void OMP_jacobi_lin_solve(Axis mode, std::vector<double> &value, const std::vector<double> &oldValue, double diffusionRate) const;
    void OMP_fadeDensity(std::vector<double> &density) const;
    void OMP_CalculateVorticity(const std::vector<double> &vX, const std::vector<double> &vY, std::vector<double> &vorticity) const;

    // CUDA implementations
#ifdef CUDA_SUPPORT
    void CUDA_diffuse(Axis mode, double *current, const double *previous, double diffusion, double dt);
    void CUDA_advect(Axis mode, double *d_density, const double *d_density0, const double *d_vX, const double *d_vY, double dt) const;
    void CUDA_project(double *d_vX, double *d_vY, double *d_vX_prev, double *d_vY_prev);
    void CUDA_set_bnd(Axis mode, double *d_value) const;
    void CUDA_lin_solve(Axis mode, double *d_value, const double *d_oldValue, double diffusionRate, double cRecip);
    void CUDA_fadeDensity(double *density) const;
#endif
};
