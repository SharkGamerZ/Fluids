#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "../globals.hpp"

const int ITERATIONS = 15; ///< number of iterations
enum Axis { X, Y, ZERO }; ///< axis of the matrix

typedef struct FluidMatrix {
    unsigned int size; ///< size of the matrix
    double dt;     ///< delta time
    double diff;   ///< diffusion
    double visc;   ///< viscosity

    double *density;  ///< density (vector)
    double *density0; ///< density at previous step (vector)

    double *Vx; ///< velocity on x axis (vector)
    double *Vy; ///< velocity on y axis (vector)

    double *Vx0; ///< velocity on x axis at previous step (vector)
    double *Vy0; ///< velocity on y axis at previous step (vector)
} FluidMatrix;

FluidMatrix *CFluidMatrix_new(unsigned int size, double diffusion, double viscosity, double dt);

void CFluidMatrix_delete(FluidMatrix *matrix);

static inline unsigned int CFluidMatrix_index(unsigned int i, unsigned int j, unsigned int matrix_size) { return j + i * matrix_size; }

void CFluidMatrix_reset(FluidMatrix *matrix);

void CFluidMatrix_step(FluidMatrix *matrix);

void CFluidMatrix_OMP_step(FluidMatrix *matrix);

void CFluidMatrix_add_density(FluidMatrix *matrix, unsigned int x, unsigned int y, double amount);

void CFluidMatrix_add_velocity(FluidMatrix *matrix, unsigned int x, unsigned int y, double amountX, double amountY);

// Private methods

void CFluidMatrix_diffuse(FluidMatrix *matrix, enum Axis mode, double *value, double *oldValue, double diffusion, double dt);

void CFluidMatrix_OMP_diffuse(FluidMatrix *matrix, enum Axis mode, double *value, double *oldValue, double diffusion, double dt);

void CFluidMatrix_advect(FluidMatrix *matrix, enum Axis mode, double *value, double *oldValue, double *Vx, double *Vy, double dt);

void CFluidMatrix_OMP_advect(FluidMatrix *matrix, enum Axis mode, double *value, double *oldValue, double *Vx, double *Vy, double dt);

void CFluidMatrix_project(FluidMatrix *matrix, double *Vx, double *Vy, double *p, double *div);

void CFluidMatrix_OMP_project(FluidMatrix *matrix, double *Vx, double *Vy, double *p, double *div);

void CFluidMatrix_set_bnd(FluidMatrix *matrix, enum Axis mode, double *value);

void CFluidMatrix_OMP_set_bnd(FluidMatrix *matrix, enum Axis mode, double *value);

void
CFluidMatrix_linearSolve(FluidMatrix *matrix, enum Axis mode, double *nextValue, double *value, double diffusionRate);

void CFluidMatrix_OMP_lin_solve(FluidMatrix *matrix, enum Axis mode, double *nextValue, double *value, double diffusionRate);

void CFluidMatrix_fade_density(FluidMatrix *matrix, double *density);

void CFluidMatrix_OMP_fade_density(FluidMatrix *matrix, double *density);
