#include "FluidMatrix.h"

FluidMatrix *CFluidMatrix_new(unsigned int size, double diffusion, double viscosity, double dt) {
    FluidMatrix *matrix = (FluidMatrix *) malloc(sizeof(FluidMatrix));
    matrix->size = size;
    matrix->dt = dt;
    matrix->diff = diffusion;
    matrix->visc = viscosity;

    matrix->density = (double *) malloc(size * size * sizeof(double));
    matrix->density0 = (double *) malloc(size * size * sizeof(double));

    matrix->Vx = (double *) malloc(size * size * sizeof(double));
    matrix->Vy = (double *) malloc(size * size * sizeof(double));

    matrix->Vx0 = (double *) malloc(size * size * sizeof(double));
    matrix->Vy0 = (double *) malloc(size * size * sizeof(double));

    return matrix;
}

void CFluidMatrix_delete(FluidMatrix *matrix) {
    if (matrix != NULL) {
        free(matrix->density);
        matrix->density = NULL;
        free(matrix->density0);
        matrix->density0 = NULL;
        free(matrix->Vx);
        matrix->Vx = NULL;
        free(matrix->Vy);
        matrix->Vy = NULL;
        free(matrix->Vx0);
        matrix->Vx0 = NULL;
        free(matrix->Vy0);
        matrix->Vy0 = NULL;
        free(matrix);
    }
}

void CFluidMatrix_reset(FluidMatrix *matrix) {
    unsigned int totalSize = matrix->size * matrix->size;
    for (unsigned int i = 0; i < totalSize; i++) {
        matrix->density[i] = 0;
        matrix->density0[i] = 0;
        matrix->Vx[i] = 0;
        matrix->Vy[i] = 0;
        matrix->Vx0[i] = 0;
        matrix->Vy0[i] = 0;
    }
}

void CFluidMatrix_step(FluidMatrix *matrix) {
    // Velocity step
    {
        CFluidMatrix_diffuse(matrix, X, matrix->Vx0, matrix->Vx, matrix->visc, matrix->dt);
        CFluidMatrix_diffuse(matrix, Y, matrix->Vy0, matrix->Vy, matrix->visc, matrix->dt);

        CFluidMatrix_project(matrix, matrix->Vx0, matrix->Vy0, matrix->Vx, matrix->Vy);

        CFluidMatrix_advect(matrix, X, matrix->Vx, matrix->Vx0, matrix->Vx0, matrix->Vy0, matrix->dt);
        CFluidMatrix_advect(matrix, Y, matrix->Vy, matrix->Vy0, matrix->Vx0, matrix->Vy0, matrix->dt);

        CFluidMatrix_project(matrix, matrix->Vx, matrix->Vy, matrix->Vx0, matrix->Vy0);
    }

    // Density step
    {
        CFluidMatrix_diffuse(matrix, ZERO, matrix->density0, matrix->density, matrix->diff, matrix->dt);

        CFluidMatrix_advect(matrix, ZERO, matrix->density, matrix->density0, matrix->Vx, matrix->Vy, matrix->dt);
    }

    CFluidMatrix_fade_density(matrix, matrix->density);
}

void CFluidMatrix_OMP_step(FluidMatrix *matrix) {
    // Velocity step
    {
        CFluidMatrix_OMP_diffuse(matrix, X, matrix->Vx0, matrix->Vx, matrix->visc, matrix->dt);
        CFluidMatrix_OMP_diffuse(matrix, Y, matrix->Vy0, matrix->Vy, matrix->visc, matrix->dt);

        CFluidMatrix_OMP_project(matrix, matrix->Vx0, matrix->Vy0, matrix->Vx, matrix->Vy);

        CFluidMatrix_OMP_advect(matrix, X, matrix->Vx, matrix->Vx0, matrix->Vx0, matrix->Vy0, matrix->dt);
        CFluidMatrix_OMP_advect(matrix, Y, matrix->Vy, matrix->Vy0, matrix->Vx0, matrix->Vy0, matrix->dt);

        CFluidMatrix_OMP_project(matrix, matrix->Vx, matrix->Vy, matrix->Vx0, matrix->Vy0);
    }

    // Density step
    {
        CFluidMatrix_OMP_diffuse(matrix, ZERO, matrix->density0, matrix->density, matrix->diff, matrix->dt);

        CFluidMatrix_OMP_advect(matrix, ZERO, matrix->density, matrix->density0, matrix->Vx, matrix->Vy, matrix->dt);
    }

    CFluidMatrix_OMP_fade_density(matrix, matrix->density);
}

void CFluidMatrix_add_density(FluidMatrix *matrix, unsigned int x, unsigned int y, double amount) {
    matrix->density[CFluidMatrix_index(y, x, matrix->size)] += amount;
}

void CFluidMatrix_add_velocity(FluidMatrix *matrix, unsigned int x, unsigned int y, double amountX, double amountY) {
    unsigned int idx = CFluidMatrix_index(y, x, matrix->size);

    matrix->Vx[idx] += amountX;
    matrix->Vy[idx] += amountY;
}

// Private methods

void CFluidMatrix_diffuse(FluidMatrix *matrix, enum Axis mode, double *value, double *oldValue, double diffusion, double dt) {
    double diffusionRate = dt * diffusion * (matrix->size - 2) * (matrix->size - 2);

    CFluidMatrix_linearSolve(matrix, mode, value, oldValue, diffusionRate);
}

void CFluidMatrix_OMP_diffuse(FluidMatrix *matrix, enum Axis mode, double *value, double *oldValue, double diffusion, double dt) {
    double diffusionRate = dt * diffusion * matrix->size * matrix->size;

    CFluidMatrix_OMP_lin_solve(matrix, mode, value, oldValue, diffusionRate);
}

void CFluidMatrix_advect(FluidMatrix *matrix, enum Axis mode, double *value, double *oldValue, double *Vx, double *Vy, double dt) {
    double i0, i1, j0, j1;

    double dt0 = dt * (matrix->size - 2);

    double s0, s1, t0, t1;
    double tmp1, tmp2, x, y;

    double N_double = matrix->size - 2;

    for (unsigned int i = 1; i < matrix->size - 1; i++) {
        for (unsigned int j = 1; j < matrix->size - 1; j++) {
            double v1 = Vx[CFluidMatrix_index(i, j, matrix->size)];
            double v2 = Vy[CFluidMatrix_index(i, j, matrix->size)];
            tmp1 = dt0 * v1;
            tmp2 = dt0 * v2;
            x = (double) i - tmp1;
            y = (double) j - tmp2;

            if (x < 0.5f) x = 0.5f;
            if (x > N_double + 0.5f) x = N_double + 0.5f;
            i0 = floor(x);
            i1 = i0 + 1.0f;
            if (y < 0.5f) y = 0.5f;
            if (y > N_double + 0.5f) y = N_double + 0.5f;
            j0 = floor(y);
            j1 = j0 + 1.0f;

            s1 = x - i0;
            s0 = 1.0f - s1;
            t1 = y - j0;
            t0 = 1.0f - t1;

            int i0i = i0;
            int i1i = i1;
            int j0i = j0;
            int j1i = j1;

            value[CFluidMatrix_index(j, i, matrix->size)] =
                    s0 * (t0 * oldValue[CFluidMatrix_index(i0i, j0i, matrix->size)] + t1 * oldValue[CFluidMatrix_index(i0i, j1i, matrix->size)]) +
                    s1 * (t0 * oldValue[CFluidMatrix_index(i1i, j0i, matrix->size)] + t1 * oldValue[CFluidMatrix_index(i1i, j1i, matrix->size)]);
        }
    }

    CFluidMatrix_set_bnd(matrix, mode, value);
}

void CFluidMatrix_OMP_advect(FluidMatrix *matrix, enum Axis mode, double *value, double *oldValue, double *Vx, double *Vy, double dt) {
    double dt0 = dt * (matrix->size - 2);
    double N_double = matrix->size - 2;


#pragma omp parallel num_threads(numThreads)
    {
        double i0, i1, j0, j1;
        double s0, s1, t0, t1;
        double tmp1, tmp2, x, y;

#pragma omp for
        for (int i = 1; i < matrix->size - 1; i++) {
            for (int j = 1; j < matrix->size - 1; j++) {
                double v1 = Vx[CFluidMatrix_index(i, j, matrix->size)];
                double v2 = Vy[CFluidMatrix_index(i, j, matrix->size)];
                tmp1 = dt0 * v1;
                tmp2 = dt0 * v2;
                x = (double) i - tmp1;
                y = (double) j - tmp2;

                if (x < 0.5f) x = 0.5f;
                if (x > N_double + 0.5f) x = N_double + 0.5f;
                i0 = floor(x);
                i1 = i0 + 1.0f;
                if (y < 0.5f) y = 0.5f;
                if (y > N_double + 0.5f) y = N_double + 0.5f;
                j0 = floor(y);
                j1 = j0 + 1.0f;

                s1 = x - i0;
                s0 = 1.0f - s1;
                t1 = y - j0;
                t0 = 1.0f - t1;

                int i0i = i0;
                int i1i = i1;
                int j0i = j0;
                int j1i = j1;

                value[CFluidMatrix_index(i, j, matrix->size)] =
                        s0 * (t0 * oldValue[CFluidMatrix_index(i0i, j0i, matrix->size)] + t1 * oldValue[CFluidMatrix_index(i0i, j1i, matrix->size)]) +
                        s1 * (t0 * oldValue[CFluidMatrix_index(i1i, j0i, matrix->size)] + t1 * oldValue[CFluidMatrix_index(i1i, j1i, matrix->size)]);
            }
        }
        CFluidMatrix_OMP_set_bnd(matrix, mode, value);
    }
}

void CFluidMatrix_project(FluidMatrix *matrix, double *Vx, double *Vy, double *p, double *div) {
    for (unsigned int i = 1; i < matrix->size - 1; i++) {
        for (unsigned int j = 1; j < matrix->size - 1; j++) {
            div[CFluidMatrix_index(i, j, matrix->size)] =
                    -0.5f * (Vx[CFluidMatrix_index(i + 1, j, matrix->size)] - Vx[CFluidMatrix_index(i - 1, j, matrix->size)] +
                             Vy[CFluidMatrix_index(i, j + 1, matrix->size)] - Vy[CFluidMatrix_index(i, j - 1, matrix->size)]) / matrix->size;
            p[CFluidMatrix_index(i, j, matrix->size)] = 0;
        }
    }

    CFluidMatrix_set_bnd(matrix, ZERO, div);
    CFluidMatrix_set_bnd(matrix, ZERO, p);
    CFluidMatrix_linearSolve(matrix, ZERO, p, div, 1);

    for (unsigned int i = 1; i < matrix->size - 1; i++) {
        for (unsigned int j = 1; j < matrix->size - 1; j++) {
            Vx[CFluidMatrix_index(i, j, matrix->size)] -=
                    0.5f * (p[CFluidMatrix_index(i + 1, j, matrix->size)] - p[CFluidMatrix_index(i - 1, j, matrix->size)]) * matrix->size;
            Vy[CFluidMatrix_index(i, j, matrix->size)] -=
                    0.5f * (p[CFluidMatrix_index(i, j + 1, matrix->size)] - p[CFluidMatrix_index(i, j - 1, matrix->size)]) * matrix->size;
        }
    }

    CFluidMatrix_set_bnd(matrix, X, Vx);
    CFluidMatrix_set_bnd(matrix, Y, Vy);

}

void CFluidMatrix_OMP_project(FluidMatrix *matrix, double *Vx, double *Vy, double *p, double *div) {
#pragma omp parallel default(shared) num_threads(numThreads)
    {
#pragma omp for schedule(guided) collapse(2)
        for (uint32_t i = 1; i < matrix->size - 1; i++) {
            for (uint32_t j = 1; j < matrix->size - 1; j++) {
                div[CFluidMatrix_index(i, j, matrix->size)] =
                        -0.5f * (Vx[CFluidMatrix_index(i + 1, j, matrix->size)] - Vx[CFluidMatrix_index(i - 1, j, matrix->size)] +
                                 Vy[CFluidMatrix_index(i, j + 1, matrix->size)] - Vy[CFluidMatrix_index(i, j - 1, matrix->size)]) /
                        matrix->size;
                p[CFluidMatrix_index(i, j, matrix->size)] = 0;
            }
        }
        CFluidMatrix_set_bnd(matrix, ZERO, div);
        CFluidMatrix_set_bnd(matrix, ZERO, p);
#pragma omp single
        {
            omp_set_nested(1);
            CFluidMatrix_OMP_lin_solve(matrix, ZERO, p, div, 1);
        }

#pragma omp for schedule(guided) collapse(2)
        for (uint32_t i = 1; i < matrix->size - 1; i++) {
            for (uint32_t j = 1; j < matrix->size - 1; j++) {
                Vx[CFluidMatrix_index(i, j, matrix->size)] -=
                        0.5f * (p[CFluidMatrix_index(i + 1, j, matrix->size)] - p[CFluidMatrix_index(i - 1, j, matrix->size)]) * matrix->size;
                Vy[CFluidMatrix_index(i, j, matrix->size)] -=
                        0.5f * (p[CFluidMatrix_index(i, j + 1, matrix->size)] - p[CFluidMatrix_index(i, j - 1, matrix->size)]) * matrix->size;
            }
        }
        CFluidMatrix_set_bnd(matrix, X, Vx);
        CFluidMatrix_set_bnd(matrix, Y, Vy);
    }
}

void CFluidMatrix_set_bnd(FluidMatrix *matrix, enum Axis mode, double *value) {
    for (uint32_t i = 1; i < matrix->size - 1; i++) {
        value[CFluidMatrix_index(i, 0, matrix->size)] =
                mode == Y ? -value[CFluidMatrix_index(i, 1, matrix->size)] : value[CFluidMatrix_index(i, 1, matrix->size)];
        value[CFluidMatrix_index(i, matrix->size - 1, matrix->size)] =
                mode == Y ? -value[CFluidMatrix_index(i, matrix->size - 2, matrix->size)] : value[CFluidMatrix_index(i, matrix->size - 2, matrix->size)];
    }
    for (uint32_t j = 1; j < matrix->size - 1; j++) {
        value[CFluidMatrix_index(0, j, matrix->size)] =
                mode == X ? -value[CFluidMatrix_index(1, j, matrix->size)] : value[CFluidMatrix_index(1, j, matrix->size)];
        value[CFluidMatrix_index(matrix->size - 1, j, matrix->size)] =
                mode == X ? -value[CFluidMatrix_index(matrix->size - 2, j, matrix->size)] : value[CFluidMatrix_index(matrix->size - 2, j, matrix->size)];
    }


    value[CFluidMatrix_index(0, 0, matrix->size)] =
            0.5f * (value[CFluidMatrix_index(1, 0, matrix->size)] +
                    value[CFluidMatrix_index(0, 1, matrix->size)]);
    value[CFluidMatrix_index(0, matrix->size - 1, matrix->size)] =
            0.5f * (value[CFluidMatrix_index(1, matrix->size - 1, matrix->size)] +
                    value[CFluidMatrix_index(0, matrix->size - 2, matrix->size)]);

    value[CFluidMatrix_index(matrix->size - 1, 0, matrix->size)] =
            0.5f * (value[CFluidMatrix_index(matrix->size - 2, 0, matrix->size)] +
                    value[CFluidMatrix_index(matrix->size - 1, 1, matrix->size)]);
    value[CFluidMatrix_index(matrix->size - 1, matrix->size - 1, matrix->size)] =
            0.5f * (value[CFluidMatrix_index(matrix->size - 2, matrix->size - 1, matrix->size)] +
                    value[CFluidMatrix_index(matrix->size - 1, matrix->size - 2, matrix->size)]);
}

void CFluidMatrix_OMP_set_bnd(FluidMatrix *matrix, enum Axis mode, double *value) {
#pragma omp for
    for (uint32_t i = 1; i < matrix->size - 1; i++) {
        value[CFluidMatrix_index(i, 0, matrix->size)] =
                mode == Y ? -value[CFluidMatrix_index(i, 1, matrix->size)] : value[CFluidMatrix_index(i, 1, matrix->size)];
        value[CFluidMatrix_index(i, matrix->size - 1, matrix->size)] =
                mode == Y ? -value[CFluidMatrix_index(i, matrix->size - 2, matrix->size)] : value[CFluidMatrix_index(i, matrix->size - 2, matrix->size)];
    }
#pragma omp for
    for (uint32_t j = 1; j < matrix->size - 1; j++) {
        value[CFluidMatrix_index(0, j, matrix->size)] =
                mode == X ? -value[CFluidMatrix_index(1, j, matrix->size)] : value[CFluidMatrix_index(1, j, matrix->size)];
        value[CFluidMatrix_index(matrix->size - 1, j, matrix->size)] =
                mode == X ? -value[CFluidMatrix_index(matrix->size - 2, j, matrix->size)] : value[CFluidMatrix_index(matrix->size - 2, j, matrix->size)];
    }

#pragma omp single
    {
        value[CFluidMatrix_index(0, 0, matrix->size)] =
                0.5f * (value[CFluidMatrix_index(1, 0, matrix->size)] +
                        value[CFluidMatrix_index(0, 1, matrix->size)]);
        value[CFluidMatrix_index(0, matrix->size - 1, matrix->size)] =
                0.5f * (value[CFluidMatrix_index(1, matrix->size - 1, matrix->size)] +
                        value[CFluidMatrix_index(0, matrix->size - 2, matrix->size)]);

        value[CFluidMatrix_index(matrix->size - 1, 0, matrix->size)] =
                0.5f * (value[CFluidMatrix_index(matrix->size - 2, 0, matrix->size)] +
                        value[CFluidMatrix_index(matrix->size - 1, 1, matrix->size)]);
        value[CFluidMatrix_index(matrix->size - 1, matrix->size - 1, matrix->size)] =
                0.5f * (value[CFluidMatrix_index(matrix->size - 2, matrix->size - 1, matrix->size)] +
                        value[CFluidMatrix_index(matrix->size - 1, matrix->size - 2, matrix->size)]);
    }
}

void CFluidMatrix_linearSolve(FluidMatrix *matrix, enum Axis mode, double *nextValue, double *value, double diffusionRate) {
    double c = 1 + 6 * diffusionRate;
    double cRecip = 1.0 / c;

    for (int k = 0; k < ITERATIONS; k++) {
        for (uint32_t i = 1; i < matrix->size - 1; i++) {
            for (uint32_t j = 1; j < matrix->size - 1; j++) {
                nextValue[CFluidMatrix_index(i, j, matrix->size)] =
                        (value[CFluidMatrix_index(i, j, matrix->size)] + diffusionRate * (
                                nextValue[CFluidMatrix_index(i + 1, j, matrix->size)] +
                                nextValue[CFluidMatrix_index(i - 1, j, matrix->size)] +
                                nextValue[CFluidMatrix_index(i, j + 1, matrix->size)] +
                                nextValue[CFluidMatrix_index(i, j - 1, matrix->size)])
                        ) * cRecip;
            }
        }
        CFluidMatrix_set_bnd(matrix, mode, nextValue);
    }
}

void CFluidMatrix_OMP_lin_solve(FluidMatrix *matrix, enum Axis mode, double *nextValue, double *value, double diffusionRate) {
    double c = 1 + 4 * diffusionRate;
    double cRecip = 1.0 / c;

    for (int k = 0; k < ITERATIONS; k++) {
#pragma omp parallel default(shared) num_threads(numThreads)
        {
#pragma omp for schedule(guided) collapse(2)
            for (uint32_t i = 1; i < matrix->size - 1; i++) {
                for (uint32_t j = 1; j < matrix->size - 1; j++) {
                    nextValue[CFluidMatrix_index(i, j, matrix->size)] =
                            (value[CFluidMatrix_index(i, j, matrix->size)] + diffusionRate * (
                                    nextValue[CFluidMatrix_index(i + 1, j, matrix->size)] +
                                    nextValue[CFluidMatrix_index(i - 1, j, matrix->size)] +
                                    nextValue[CFluidMatrix_index(i, j + 1, matrix->size)] +
                                    nextValue[CFluidMatrix_index(i, j - 1, matrix->size)])
                            ) * cRecip;
                }
            }
            CFluidMatrix_OMP_set_bnd(matrix, mode, nextValue);
        }
    }
}

void CFluidMatrix_fade_density(FluidMatrix *matrix, double *density) {
    for (uint32_t i = 0; i < matrix->size * matrix->size; i++) {
        double d = density[i];
        density[i] = (d - 0.005f < 0) ? 0 : d - 0.005f;
    }
}

void CFluidMatrix_OMP_fade_density(FluidMatrix *matrix, double *density) {
#pragma omp parallel for num_threads(numThreads) default(shared)
    for (uint32_t i = 0; i < matrix->size * matrix->size; i++) {
        double d = matrix->density[i];
        density[i] = (d - 0.005f < 0) ? 0 : d - 0.005f;
    }
}
