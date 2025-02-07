#include "fluid_matrix.hpp"

#include "../utils.hpp"

#include <cmath>


FluidMatrix::FluidMatrix(uint32_t size, double diffusion, double viscosity, double dt)
    : size(size), dt(dt), diff(diffusion), visc(viscosity), density(std::vector<double>(size * size)), density_prev(std::vector<double>(size * size)),
      Vx(std::vector<double>(size * size)), Vy(std::vector<double>(size * size)), Vx_prev(std::vector<double>(size * size)), Vy_prev(std::vector<double>(size * size)),
      numMaxThreads(omp_get_max_threads()) {
    log(Utils::LogLevel::DEBUG, std::cout, std::format("FluidMatrix created with {} threads", numMaxThreads));
}

FluidMatrix::~FluidMatrix() { log(Utils::LogLevel::DEBUG, std::cout, "FluidMatrix deleted"); }

void FluidMatrix::reset() {
    std::ranges::fill(density, 0);
    std::ranges::fill(density_prev, 0);
    std::ranges::fill(Vx, 0);
    std::ranges::fill(Vy, 0);
    std::ranges::fill(Vx_prev, 0);
    std::ranges::fill(Vy_prev, 0);
}

void FluidMatrix::step() {
    // Velocity
    {
        diffuse(X, Vx_prev, Vx, visc, dt);
        diffuse(Y, Vy_prev, Vy, visc, dt);

        project(Vx_prev, Vy_prev, Vx, Vy);

        advect(X, Vx, Vx_prev, Vx_prev, Vy_prev, dt);
        advect(Y, Vy, Vy_prev, Vx_prev, Vy_prev, dt);
    }

    // Density
    {
        diffuse(ZERO, density_prev, density, diff, dt);
        advect(ZERO, density, density_prev, Vx, Vy, dt);
    }

    fadeDensity(density);
}

void FluidMatrix::OMP_step() {
    // Velocity
    {
        OMP_diffuse(X, Vx_prev, Vx, visc, dt);
        OMP_diffuse(Y, Vy_prev, Vy, visc, dt);

        OMP_project(Vx_prev, Vy_prev, Vx, Vy);

        OMP_advect(X, Vx, Vx_prev, Vx_prev, Vy_prev, dt);
        OMP_advect(Y, Vy, Vy_prev, Vx_prev, Vy_prev, dt);
    }

    // Density
    {
        OMP_diffuse(ZERO, density_prev, density, diff, dt);
        OMP_advect(ZERO, density, density_prev, Vx, Vy, dt);
    }
    OMP_fadeDensity(density);
}

void FluidMatrix::addDensity(uint32_t x, uint32_t y, double amount) { this->density[index(y, x, this->size)] += amount; }

void FluidMatrix::addVelocity(uint32_t x, uint32_t y, double amountX, double amountY) {
    uint32_t idx = index(y, x, this->size);

    this->Vx[idx] += amountX;
    this->Vy[idx] += amountY;
}

void FluidMatrix::diffuse(Axis mode, std::vector<double> &current, std::vector<double> &previous, double diffusion, double dt) const {
    double diffusionRate = dt * diffusion * (this->size - 2) * (this->size - 2);
    lin_solve(mode, current, previous, diffusionRate);
}

void FluidMatrix::OMP_diffuse(Axis mode, std::vector<double> &current, std::vector<double> &previous, double diffusion, double dt) const {
    double diffusionRate = dt * diffusion * (this->size - 2) * (this->size - 2);
    OMP_lin_solve(mode, current, previous, diffusionRate);
}

void FluidMatrix::advect(Axis mode, std::vector<double> &d, std::vector<double> &d0, std::vector<double> &vX, std::vector<double> &vY, double dt) const {
    double i0, i1, j0, j1;
    double dt0 = dt * (this->size - 2);
    double s0, s1, t0, t1;
    double tmp1, tmp2, x, y;
    double N_double = this->size - 2;

    for (int i = 1; i < this->size - 1; i++) {
        for (int j = 1; j < this->size - 1; j++) {
            double v1 = vX[index(i, j, this->size)];
            double v2 = vY[index(i, j, this->size)];
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

            d[index(i, j, this->size)] = s0 * (t0 * d0[index(i0i, j0i, this->size)] + t1 * d0[index(i0i, j1i, this->size)]) +
                                         s1 * (t0 * d0[index(i1i, j0i, this->size)] + t1 * d0[index(i1i, j1i, this->size)]);
        }
    }

    set_bnd(mode, d);
}

void FluidMatrix::OMP_advect(Axis mode, std::vector<double> &d, std::vector<double> &d0, std::vector<double> &vX, std::vector<double> &vY, double dt) const {
    double dt0 = dt * (this->size - 2);
    double N_double = this->size - 2;

#pragma omp parallel num_threads(this->numMaxThreads)
    {
        double i0, i1, j0, j1;
        double s0, s1, t0, t1;
        double tmp1, tmp2, x, y;

#pragma omp for
        for (int i = 1; i < this->size; i++) {
            for (int j = 1; j < this->size; j++) {
                double v1 = vX[index(i, j, this->size)];
                double v2 = vY[index(i, j, this->size)];
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

                d[index(i, j, this->size)] = s0 * (t0 * d0[index(i0i, j0i, this->size)] + t1 * d0[index(i0i, j1i, this->size)]) +
                                             s1 * (t0 * d0[index(i1i, j0i, this->size)] + t1 * d0[index(i1i, j1i, this->size)]);
            }
        }
        OMP_set_bnd(mode, d);
    }
}

void FluidMatrix::project(std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &p, std::vector<double> &div) const {
    for (int i = 1; i < this->size - 1; i++) {
        for (int j = 1; j < this->size - 1; j++) {
            div[index(i, j, this->size)] =
                    -0.5f * (vX[index(i + 1, j, this->size)] - vX[index(i - 1, j, this->size)] + vY[index(i, j + 1, this->size)] - vY[index(i, j - 1, this->size)]) / this->size;
            p[index(i, j, this->size)] = 0;
        }
    }

    set_bnd(ZERO, div);
    set_bnd(ZERO, p);
    lin_solve(ZERO, p, div, 1);

    for (int i = 1; i < this->size - 1; i++) {
        for (int j = 1; j < this->size - 1; j++) {
            vX[index(i, j, this->size)] -= 0.5f * (p[index(i + 1, j, this->size)] - p[index(i - 1, j, this->size)]) * this->size;
            vY[index(i, j, this->size)] -= 0.5f * (p[index(i, j + 1, this->size)] - p[index(i, j - 1, this->size)]) * this->size;
        }
    }

    set_bnd(X, vX);
    set_bnd(Y, vY);
}

void FluidMatrix::OMP_project(std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &p, std::vector<double> &div) const {
#pragma omp parallel default(shared) num_threads(this->numMaxThreads)
    {
#pragma omp for schedule(guided) collapse(2)
        for (int i = 1; i < this->size - 1; i++) {
            for (int j = 1; j < this->size - 1; j++) {
                div[index(i, j, this->size)] =
                        -0.5f * (vX[index(i + 1, j, this->size)] - vX[index(i - 1, j, this->size)] + vY[index(i, j + 1, this->size)] - vY[index(i, j - 1, this->size)]) /
                        this->size;
                p[index(i, j, this->size)] = 0;
            }
        }

        OMP_set_bnd(ZERO, div);
        OMP_set_bnd(ZERO, p);
#pragma omp single
        {
            omp_set_nested(1);
            OMP_lin_solve(ZERO, p, div, 1);
        }

#pragma omp for schedule(guided) collapse(2)
        for (int i = 1; i < this->size - 1; i++) {
            for (int j = 1; j < this->size - 1; j++) {
                vX[index(i, j, this->size)] -= 0.5f * (p[index(i + 1, j, this->size)] - p[index(i - 1, j, this->size)]) * this->size;
                vY[index(i, j, this->size)] -= 0.5f * (p[index(i, j + 1, this->size)] - p[index(i, j - 1, this->size)]) * this->size;
            }
        }

        OMP_set_bnd(X, vX);
        OMP_set_bnd(Y, vY);
    }
}

void FluidMatrix::set_bnd(Axis mode, std::vector<double> &attr) const {
    for (int i = 1; i < this->size - 1; i++) {
        attr[index(i, 0, this->size)] = mode == Y ? -attr[index(i, 1, this->size)] : attr[index(i, 1, this->size)];
        attr[index(i, this->size - 1, this->size)] = mode == Y ? -attr[index(i, this->size - 2, this->size)] : attr[index(i, this->size - 2, this->size)];
    }

    for (int j = 1; j < this->size - 1; j++) {
        attr[index(0, j, this->size)] = mode == X ? -attr[index(1, j, this->size)] : attr[index(1, j, this->size)];
        attr[index(this->size - 1, j, this->size)] = mode == X ? -attr[index(this->size - 2, j, this->size)] : attr[index(this->size - 2, j, this->size)];
    }

    attr[index(0, 0, this->size)] = 0.5f * (attr[index(1, 0, this->size)] + attr[index(0, 1, this->size)]);
    attr[index(0, this->size - 1, this->size)] = 0.5f * (attr[index(1, this->size - 1, this->size)] + attr[index(0, this->size - 2, this->size)]);
    attr[index(this->size - 1, 0, this->size)] = 0.5f * (attr[index(this->size - 2, 0, this->size)] + attr[index(this->size - 1, 1, this->size)]);
    attr[index(this->size - 1, this->size - 1, this->size)] =
            0.5f * (attr[index(this->size - 2, this->size - 1, this->size)] + attr[index(this->size - 1, this->size - 2, this->size)]);
}

void FluidMatrix::OMP_set_bnd(Axis mode, std::vector<double> &attr) const {
#pragma omp for
    for (int i = 1; i < this->size - 1; i++) {
        attr[index(i, 0, this->size)] = mode == Y ? -attr[index(i, 1, this->size)] : attr[index(i, 1, this->size)];
        attr[index(i, this->size - 1, this->size)] = mode == Y ? -attr[index(i, this->size - 2, this->size)] : attr[index(i, this->size - 2, this->size)];
    }

#pragma omp for
    for (int j = 1; j < this->size - 1; j++) {
        attr[index(0, j, this->size)] = mode == X ? -attr[index(1, j, this->size)] : attr[index(1, j, this->size)];
        attr[index(this->size - 1, j, this->size)] = mode == X ? -attr[index(this->size - 2, j, this->size)] : attr[index(this->size - 2, j, this->size)];
    }

#pragma omp single
    {
        attr[index(0, 0, this->size)] = 0.5f * (attr[index(1, 0, this->size)] + attr[index(0, 1, this->size)]);
        attr[index(0, this->size - 1, this->size)] = 0.5f * (attr[index(1, this->size - 1, this->size)] + attr[index(0, this->size - 2, this->size)]);
        attr[index(this->size - 1, 0, this->size)] = 0.5f * (attr[index(this->size - 2, 0, this->size)] + attr[index(this->size - 1, 1, this->size)]);
        attr[index(this->size - 1, this->size - 1, this->size)] =
                0.5f * (attr[index(this->size - 2, this->size - 1, this->size)] + attr[index(this->size - 1, this->size - 2, this->size)]);
    }
}

void FluidMatrix::lin_solve(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusionRate) const {
    double c = 1 + 6 * diffusionRate;
    double cRecip = 1.0 / c;

    for (int k = 0; k < ITERATIONS; k++) {
        for (int i = 1; i < this->size - 1; i++) {
            for (int j = 1; j < this->size - 1; j++) {
                value[index(i, j, this->size)] = (oldValue[index(i, j, this->size)] + diffusionRate * (value[index(i + 1, j, this->size)] + value[index(i - 1, j, this->size)] +
                                                                                                       value[index(i, j + 1, this->size)] + value[index(i, j - 1, this->size)])) *
                                                 cRecip;
            }
        }
        set_bnd(mode, value);
    }
}

void FluidMatrix::OMP_lin_solve(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusionRate) const {
    double c = 1 + 4 * diffusionRate;
    double cRecip = 1.0 / c;

    for (int k = 0; k < ITERATIONS; k++) {
#pragma omp parallel default(shared) num_threads(this->numMaxThreads)
        {
#pragma omp for schedule(guided) collapse(2)
            for (int i = 1; i < this->size - 1; i++) {
                for (int j = 1; j < this->size - 1; j++) {
                    value[index(i, j, this->size)] =
                            (oldValue[index(i, j, this->size)] + diffusionRate * (value[index(i + 1, j, this->size)] + value[index(i - 1, j, this->size)] +
                                                                                  value[index(i, j + 1, this->size)] + value[index(i, j - 1, this->size)])) *
                            cRecip;
                }
            }
            OMP_set_bnd(mode, value);
        }
    }
}

void FluidMatrix::fadeDensity(std::vector<double> &density) const {
    for (int i = 0; i < this->size * this->size; i++) {
        double d = density[i];
        density[i] = (d - 0.005f < 0) ? 0 : d - 0.005f;
    }
}

void FluidMatrix::OMP_fadeDensity(std::vector<double> &density) const {
#pragma omp parallel for num_threads(this->numMaxThreads) default(shared)
    for (int i = 0; i < this->size * this->size; i++) {
        double d = density[i];
        density[i] = (d - 0.005f < 0) ? 0 : d - 0.005f;
    }
}
