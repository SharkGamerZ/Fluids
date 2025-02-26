#pragma once

#include "../utils.hpp"
#include <cmath>
#include <cstdint>
#include <omp.h>
#include <vector>

constexpr int ITERATIONS = 15; ///< Number of iterations
enum Axis { X, Y, ZERO };

#define SWAP(x0, x) \
    {               \
        auto tmp = x0; \
        x0 = x;     \
        x = tmp;    \
    }


/// FluidMatrix class for serial, OpenMP, and CUDA-based simulation
class FluidMatrix {
public:
    uint32_t size;                    ///< Size of the fluid matrix
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

#ifdef CUDA_SUPPORT
    double *d_density, *d_density_prev;
    double *d_vX, *d_vX_prev;
    double *d_vY, *d_vY_prev;
#endif

    FluidMatrix(uint32_t size, double diffusion, double viscosity, double dt);
    ~FluidMatrix();

    [[nodiscard]] static uint32_t index(const uint32_t i, const uint32_t j, const uint32_t matrix_size) { return j + i * matrix_size; }
    void reset();

    void step();
    void OMP_step();
#ifdef CUDA_SUPPORT
    void CUDA_step();
#endif

    void addDensity(uint32_t x, uint32_t y, double amount);
    void addVelocity(uint32_t x, uint32_t y, double amountX, double amountY);

protected:
    int numMaxThreads; ///< Number of threads used by OpenMP

#ifdef CUDA_SUPPORT
    /// Preallocate CUDA memory on matrix creation
    void CUDA_init();
    /// Free all CUDA memory on matrix destruction
    void CUDA_destroy() const;
#endif

    void diffuse(Axis mode, std::vector<double> &current, std::vector<double> &previous, double diffusion, double dt) const;
    void OMP_diffuse(Axis mode, std::vector<double> &current, std::vector<double> &previous, double diffusion, double dt) const;
#ifdef CUDA_SUPPORT
    void CUDA_diffuse(Axis mode, std::vector<double> &current, std::vector<double> &previous, double diffusion, double dt) const;
#endif

    void advect(Axis mode, std::vector<double> &d, std::vector<double> &d0, std::vector<double> &vX, std::vector<double> &vY, double dt) const;
    void OMP_advect(Axis mode, std::vector<double> &d, std::vector<double> &d0, std::vector<double> &vX, std::vector<double> &vY, double dt) const;
#ifdef CUDA_SUPPORT
    void CUDA_advect(Axis mode, std::vector<double> &d, std::vector<double> &d0, std::vector<double> &vX, std::vector<double> &vY, double dt) const;
#endif

    void project(std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &p, std::vector<double> &div) const;
    void OMP_project(std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &p, std::vector<double> &div) const;
#ifdef CUDA_SUPPORT
    void CUDA_project(std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &p, std::vector<double> &div) const;
#endif

    void set_bnd(Axis mode, std::vector<double> &attr) const;
    void OMP_set_bnd(Axis mode, std::vector<double> &attr) const;
#ifdef CUDA_SUPPORT
    void CUDA_set_bnd(Axis mode, std::vector<double> &attr) const;
#endif

    void lin_solve(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusionRate) const;
    void OMP_lin_solve(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusionRate) const;
#ifdef CUDA_SUPPORT
    void CUDA_lin_solve(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusionRate) const;
#endif

    void fadeDensity(std::vector<double> &density) const;
    void OMP_fadeDensity(std::vector<double> &density) const;
#ifdef CUDA_SUPPORT
    void CUDA_fadeDensity(std::vector<double> &density) const;
#endif


    void CalculateVorticity(std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &vorticity);
    void OMP_CalculateVorticity(std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &vorticity);

};
