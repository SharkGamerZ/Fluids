#pragma once

#include "../utils.hpp"
#include <cmath>
#include <cstdint>
#include <omp.h>
#include <vector>

extern int GAUSS_ITERATIONS;    ///< Number of iterations for the Gauss-Siedel
extern int JACOBI_ITERATIONS;  ///< Number of iterations for the Jacobi
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
    int size;                    ///< Size of the fluid matrix
    double dt;                        ///< Delta time
    double diff;                      ///< Diffusion
    double visc;                      ///< Viscosity
    double gravity = -0.00981; // Gravity acceleration
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

    void CUDA_reset();
    void copyToHost();
    void copyToDevice();
#endif



    void addDensity(uint32_t x, uint32_t y, double amount);
    void addVelocity(uint32_t x, uint32_t y, double amountX, double amountY);

#ifdef CUDA_SUPPORT
    void CUDA_addVelocity(int x, int y, double amountX, double amountY);
    void CUDA_addDensity(int x, int y, double amount);
#endif

    void applyGravity();

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
    void CUDA_diffuse(Axis mode, double *current, double *previous, double diffusion, double dt);
#endif

    void advect(Axis mode, std::vector<double> &d, std::vector<double> &d0, std::vector<double> &vX, std::vector<double> &vY, double dt) const;
    void OMP_advect(Axis mode, std::vector<double> &d, std::vector<double> &d0, std::vector<double> &vX, std::vector<double> &vY, double dt) const;
#ifdef CUDA_SUPPORT
    void CUDA_advect(Axis mode, double *d_density, double *d_density0, double *d_vX, double *d_vY, double dt);
#endif

    void project(std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &p, std::vector<double> &div) const;
    void OMP_project(std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &p, std::vector<double> &div) const;
#ifdef CUDA_SUPPORT
    void CUDA_project(double *d_vX, double *d_vY, double *d_vX_prev, double *d_vY_prev);
#endif

    void set_bnd(Axis mode, std::vector<double> &attr) const;
    void OMP_set_bnd(Axis mode, std::vector<double> &attr) const;
#ifdef CUDA_SUPPORT
    void CUDA_set_bnd(Axis mode, double *d_value) const;
#endif

    void gauss_lin_solve(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusionRate) const;
    void jacobi_lin_solve(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusionRate) const;
    void OMP_gauss_lin_solve(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusionRate) const;
    void OMP_jacobi_lin_solve(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusionRate) const;
#ifdef CUDA_SUPPORT
    void CUDA_lin_solve(Axis mode, double *d_value, double *d_oldValue, double diffusionRate, double cRecip);
#endif

    void fadeDensity(std::vector<double> &density) const;
    void OMP_fadeDensity(std::vector<double> &density) const;
#ifdef CUDA_SUPPORT
    void CUDA_fadeDensity(double *density) const;
#endif


    void CalculateVorticity(std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &vorticity);
    void OMP_CalculateVorticity(std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &vorticity);

};
