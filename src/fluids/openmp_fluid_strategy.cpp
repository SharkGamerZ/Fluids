#include "openmp_fluid_strategy.hpp"
#include "fluid_data_model.hpp"
#include <algorithm>
#include <omp.h>

OpenMPFluidStrategy::OpenMPFluidStrategy() : numMaxThreads(omp_get_max_threads()) {}

void OpenMPFluidStrategy::step(FluidDataModel &dataModel) {
    // Velocity
    {
        std::swap(dataModel.vX_prev, dataModel.vX);
        diffuse(dataModel, Axis::X, dataModel.vX, dataModel.vX_prev, dataModel.visc, dataModel.dt);

        std::swap(dataModel.vY_prev, dataModel.vY);
        diffuse(dataModel, Axis::Y, dataModel.vY, dataModel.vY_prev, dataModel.visc, dataModel.dt);

        project(dataModel, dataModel.vX, dataModel.vY, dataModel.vX_prev, dataModel.vY_prev);

        std::swap(dataModel.vX_prev, dataModel.vX);
        std::swap(dataModel.vY_prev, dataModel.vY);

        advect(dataModel, Axis::X, dataModel.vX, dataModel.vX_prev, dataModel.vX_prev, dataModel.vY_prev, dataModel.dt);
        advect(dataModel, Axis::Y, dataModel.vY, dataModel.vY_prev, dataModel.vX_prev, dataModel.vY_prev, dataModel.dt);

        project(dataModel, dataModel.vX, dataModel.vY, dataModel.vX_prev, dataModel.vY_prev);
    }

    // Density
    {
        std::swap(dataModel.density_prev, dataModel.density);
        diffuse(dataModel, Axis::ZERO, dataModel.density, dataModel.density_prev, dataModel.visc, dataModel.dt);

        std::swap(dataModel.density_prev, dataModel.density);
        advect(dataModel, Axis::ZERO, dataModel.density, dataModel.density_prev, dataModel.vX, dataModel.vY, dataModel.dt);
    }

    fadeDensity(dataModel, dataModel.density);
    calculateVorticity(dataModel, dataModel.vX, dataModel.vY, dataModel.vorticity);
}

void OpenMPFluidStrategy::addDensity(FluidDataModel &dataModel, const uint32_t x, const uint32_t y, const double amount) {
    dataModel.density[FluidDataModel::index(y, x, dataModel.size)] += amount;
}

void OpenMPFluidStrategy::addVelocity(FluidDataModel &dataModel, const uint32_t x, const uint32_t y, const double amountX, const double amountY) {
    uint32_t idx = FluidDataModel::index(y, x, dataModel.size);
    dataModel.vX[idx] += amountY;
    dataModel.vY[idx] += amountX;
}

void OpenMPFluidStrategy::reset(FluidDataModel &dataModel) {
    std::ranges::fill(dataModel.density, 0);
    std::ranges::fill(dataModel.density_prev, 0);
    std::ranges::fill(dataModel.vX, 0);
    std::ranges::fill(dataModel.vY, 0);
    std::ranges::fill(dataModel.vX_prev, 0);
    std::ranges::fill(dataModel.vY_prev, 0);
    std::ranges::fill(dataModel.vorticity, 0);
}

void OpenMPFluidStrategy::diffuse(const FluidDataModel &dataModel, const Axis mode,  std::vector<double> &current, const std::vector<double> &previous, const double diffusion, const double dt) const {
    double diffusionRate = dt * diffusion * (dataModel.size - 2) * (dataModel.size - 2);
    jacobi_lin_solve(dataModel, mode, current, previous, diffusionRate);
}

void OpenMPFluidStrategy::advect(const FluidDataModel &dataModel, const Axis mode, std::vector<double> &d, const std::vector<double> &d0, const std::vector<double> &vX, const std::vector<double> &vY, const double dt) const {
    #pragma omp parallel default(shared) num_threads(this->numMaxThreads)
    {
        int i0, j0, i1, j1;
        float x, y, s0, t0, s1, t1, dt0;

        dt0 = dt * (dataModel.size - 2);

        #pragma omp for
        for (int i = 1; i < dataModel.size - 1; i++) {
            for (int j = 1; j < dataModel.size - 1; j++) {
                x = i - dt0 * vX[FluidDataModel::index(i, j, dataModel.size)];
                y = j - dt0 * vY[FluidDataModel::index(i, j, dataModel.size)];

                if (x < 0.5f) x = 0.5f;
                if (x > dataModel.size - 2 + 0.5f) x = dataModel.size - 2 + 0.5f;
                i0 = (int) x; i1 = i0 + 1;

                if (y < 0.5f) y = 0.5f;
                if (y > dataModel.size - 2 + 0.5f) y = dataModel.size - 2 + 0.5f;
                j0 = (int) y; j1 = j0 + 1;

                s1 = x - i0; s0 = 1 - s1; t1 = y - j0; t0 = 1 - t1;
                d[FluidDataModel::index(i, j, dataModel.size)] =
                    s0 * (t0 * d0[FluidDataModel::index(i0, j0, dataModel.size)] +
                          t1 * d0[FluidDataModel::index(i0, j1, dataModel.size)]) +
                    s1 * (t0 * d0[FluidDataModel::index(i1, j0, dataModel.size)] +
                          t1 * d0[FluidDataModel::index(i1, j1, dataModel.size)]);
            }
        }
    }
    set_bnd(dataModel, mode, d);
}

void OpenMPFluidStrategy::project(const FluidDataModel &dataModel, std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &p, std::vector<double> &div) const {
    #pragma omp parallel default(shared) num_threads(this->numMaxThreads)
    {
        #pragma omp for schedule(guided) collapse(2)
        for (int i = 1; i < dataModel.size - 1; i++) {
            for (int j = 1; j < dataModel.size - 1; j++) {
                div[FluidDataModel::index(i, j, dataModel.size)] =
                    -0.5f * (vX[FluidDataModel::index(i + 1, j, dataModel.size)] -
                             vX[FluidDataModel::index(i - 1, j, dataModel.size)] +
                             vY[FluidDataModel::index(i, j + 1, dataModel.size)] -
                             vY[FluidDataModel::index(i, j - 1, dataModel.size)]) / dataModel.size;
                p[FluidDataModel::index(i, j, dataModel.size)] = 0;
            }
        }
    }

    set_bnd(dataModel, Axis::ZERO, div);
    set_bnd(dataModel, Axis::ZERO, p);

    double cRecip = 1.0 / 4;

    for (int k = 0; k < JACOBI_ITERATIONS; k++) {
        #pragma omp parallel default(shared) num_threads(this->numMaxThreads)
        {
            #pragma omp for schedule(guided) collapse(2)
            for (int i = 1; i < dataModel.size - 1; i++) {
                for (int j = 1; j < dataModel.size - 1; j++) {
                    p[FluidDataModel::index(i, j, dataModel.size)] =
                        (div[FluidDataModel::index(i, j, dataModel.size)] +
                         (p[FluidDataModel::index(i + 1, j, dataModel.size)] +
                          p[FluidDataModel::index(i - 1, j, dataModel.size)] +
                          p[FluidDataModel::index(i, j + 1, dataModel.size)] +
                          p[FluidDataModel::index(i, j - 1, dataModel.size)])) * cRecip;
                }
            }
        }
        set_bnd(dataModel, Axis::ZERO, p);
    }

    #pragma omp parallel default(shared) num_threads(this->numMaxThreads)
    {
        #pragma omp for schedule(guided) collapse(2)
        for (int i = 1; i < dataModel.size - 1; i++) {
            for (int j = 1; j < dataModel.size - 1; j++) {
                vX[FluidDataModel::index(i, j, dataModel.size)] -=
                    0.5f * (p[FluidDataModel::index(i + 1, j, dataModel.size)] -
                            p[FluidDataModel::index(i - 1, j, dataModel.size)]) * dataModel.size;
                vY[FluidDataModel::index(i, j, dataModel.size)] -=
                    0.5f * (p[FluidDataModel::index(i, j + 1, dataModel.size)] -
                            p[FluidDataModel::index(i, j - 1, dataModel.size)]) * dataModel.size;
            }
        }
    }

    set_bnd(dataModel, Axis::X, vX);
    set_bnd(dataModel, Axis::Y, vY);
}

void OpenMPFluidStrategy::set_bnd(const FluidDataModel &dataModel, const Axis mode, std::vector<double> &attr) const {
    #pragma omp parallel default(shared) num_threads(this->numMaxThreads)
    {
        #pragma omp for
        for (int i = 1; i < dataModel.size - 1; i++) {
            attr[FluidDataModel::index(i, 0, dataModel.size)] =
                mode == Axis::Y ? -attr[FluidDataModel::index(i, 1, dataModel.size)] :
                            attr[FluidDataModel::index(i, 1, dataModel.size)];
            attr[FluidDataModel::index(i, dataModel.size - 1, dataModel.size)] =
                mode == Axis::Y ? -attr[FluidDataModel::index(i, dataModel.size - 2, dataModel.size)] :
                            attr[FluidDataModel::index(i, dataModel.size - 2, dataModel.size)];
        }

        #pragma omp for
        for (int j = 1; j < dataModel.size - 1; j++) {
            attr[FluidDataModel::index(0, j, dataModel.size)] =
                mode == Axis::X ? -attr[FluidDataModel::index(1, j, dataModel.size)] :
                            attr[FluidDataModel::index(1, j, dataModel.size)];
            attr[FluidDataModel::index(dataModel.size - 1, j, dataModel.size)] =
                mode == Axis::X ? -attr[FluidDataModel::index(dataModel.size - 2, j, dataModel.size)] :
                            attr[FluidDataModel::index(dataModel.size - 2, j, dataModel.size)];
        }

        #pragma omp single
        {
            attr[FluidDataModel::index(0, 0, dataModel.size)] =
                0.5f * (attr[FluidDataModel::index(1, 0, dataModel.size)] +
                        attr[FluidDataModel::index(0, 1, dataModel.size)]);
            attr[FluidDataModel::index(0, dataModel.size - 1, dataModel.size)] =
                0.5f * (attr[FluidDataModel::index(1, dataModel.size - 1, dataModel.size)] +
                        attr[FluidDataModel::index(0, dataModel.size - 2, dataModel.size)]);
            attr[FluidDataModel::index(dataModel.size - 1, 0, dataModel.size)] =
                0.5f * (attr[FluidDataModel::index(dataModel.size - 2, 0, dataModel.size)] +
                        attr[FluidDataModel::index(dataModel.size - 1, 1, dataModel.size)]);
            attr[FluidDataModel::index(dataModel.size - 1, dataModel.size - 1, dataModel.size)] =
                0.5f * (attr[FluidDataModel::index(dataModel.size - 2, dataModel.size - 1, dataModel.size)] +
                        attr[FluidDataModel::index(dataModel.size - 1, dataModel.size - 2, dataModel.size)]);
        }
    }
}

void OpenMPFluidStrategy::gauss_lin_solve(const FluidDataModel &dataModel, const Axis mode, std::vector<double> &value, const std::vector<double> &oldValue, const double diffusionRate) const {
    double c = diffusionRate;
    double cRecip = 1.0 / (1 + 4 * c);

    for (int k = 0; k < GAUSS_ITERATIONS; k++) {
        #pragma omp parallel default(shared) num_threads(this->numMaxThreads)
        {
            #pragma omp for schedule(guided) collapse(2)
            for (int i = 1; i < dataModel.size - 1; i++) {
                for (int j = 1; j < dataModel.size - 1; j++) {
                    value[FluidDataModel::index(i, j, dataModel.size)] =
                        (oldValue[FluidDataModel::index(i, j, dataModel.size)] +
                         diffusionRate * (value[FluidDataModel::index(i + 1, j, dataModel.size)] +
                                         value[FluidDataModel::index(i - 1, j, dataModel.size)] +
                                         value[FluidDataModel::index(i, j + 1, dataModel.size)] +
                                         value[FluidDataModel::index(i, j - 1, dataModel.size)])) * cRecip;
                }
            }
        }
        set_bnd(dataModel, mode, value);
    }
}

void OpenMPFluidStrategy::jacobi_lin_solve(const FluidDataModel &dataModel, const Axis mode, std::vector<double> &value, const std::vector<double> &oldValue, const double diffusionRate) const {
    double c = diffusionRate;
    double cRecip = 1.0 / (1 + 4 * c);

    // Create a temporary array to store the new values
    std::vector<double> newValue(dataModel.size * dataModel.size, 0.0);

    for (int k = 0; k < JACOBI_ITERATIONS; k++) {
        #pragma omp parallel default(shared) num_threads(this->numMaxThreads)
        {
            #pragma omp for schedule(guided) collapse(2)
            for (int i = 1; i < dataModel.size - 1; i++) {
                for (int j = 1; j < dataModel.size - 1; j++) {
                    newValue[FluidDataModel::index(i, j, dataModel.size)] =
                        (oldValue[FluidDataModel::index(i, j, dataModel.size)] +
                         diffusionRate * (value[FluidDataModel::index(i + 1, j, dataModel.size)] +
                                         value[FluidDataModel::index(i - 1, j, dataModel.size)] +
                                         value[FluidDataModel::index(i, j + 1, dataModel.size)] +
                                         value[FluidDataModel::index(i, j - 1, dataModel.size)])) * cRecip;
                }
            }
        }
        // Swap the new values into the main array
        std::swap(value, newValue);
        set_bnd(dataModel, mode, value);
    }
}

void OpenMPFluidStrategy::fadeDensity(const FluidDataModel &dataModel, std::vector<double> &density) const {
    #pragma omp parallel for num_threads(this->numMaxThreads) default(shared)
    for (int i = 0; i < dataModel.size * dataModel.size; i++) {
        double d = density[i];
        density[i] = (d - 0.005f < 0) ? 0 : d - 0.005f;
    }
}

void OpenMPFluidStrategy::calculateVorticity(const FluidDataModel &dataModel, const std::vector<double> &vX, const std::vector<double> &vY, std::vector<double> &vorticity) const {
    const double h = 1.0 / (dataModel.size - 2); // assuming unit length domain

    #pragma omp parallel for num_threads(this->numMaxThreads) default(shared)
    for (int i = 1; i < dataModel.size - 1; i++) {
        for (int j = 1; j < dataModel.size - 1; j++) {
            int idx = FluidDataModel::index(i, j, dataModel.size);
            double dv_dx = (vY[FluidDataModel::index(i + 1, j, dataModel.size)] -
                           vY[FluidDataModel::index(i - 1, j, dataModel.size)]) / (2 * h);
            double du_dy = (vX[FluidDataModel::index(i, j + 1, dataModel.size)] -
                           vX[FluidDataModel::index(i, j - 1, dataModel.size)]) / (2 * h);
            vorticity[idx] = dv_dx - du_dy;
        }
    }
}
