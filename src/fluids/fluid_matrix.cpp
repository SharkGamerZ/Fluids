#include "fluid_matrix.hpp"

#define SWAP(x0, x) \
    {               \
        auto tmp = x0; \
        x0 = x;     \
        x = tmp;    \
    }

FluidMatrix::FluidMatrix(uint32_t size, double diffusion, double viscosity, double dt)
    : size(size), dt(dt), diff(diffusion), visc(viscosity), density(std::vector<double>(size * size)), density_prev(std::vector<double>(size * size)),
        vX(std::vector<double>(size * size)), vY(std::vector<double>(size * size)), vX_prev(std::vector<double>(size * size)), vY_prev(std::vector<double>(size * size)), vorticity(std::vector<double>(size * size)),
      numMaxThreads(omp_get_max_threads()) {
#ifdef CUDA_SUPPORT
    CUDA_init();
#endif
    log(Utils::LogLevel::DEBUG, std::cout, std::format("FluidMatrix created with {} threads", numMaxThreads));
}

FluidMatrix::~FluidMatrix() {
#ifdef CUDA_SUPPORT
    CUDA_destroy();
#endif
    log(Utils::LogLevel::DEBUG, std::cout, "FluidMatrix deleted");
}

void FluidMatrix::reset() {
    std::ranges::fill(density, 0);
    std::ranges::fill(density_prev, 0);
    std::ranges::fill(vX, 0);
    std::ranges::fill(vY, 0);
    std::ranges::fill(vX_prev, 0);
    std::ranges::fill(vY_prev, 0);
    std::ranges::fill(vorticity, 0);
}

void FluidMatrix::step() {
    // Velocity
    {
        SWAP(vX_prev, vX); diffuse(X, vX, vX_prev, visc, dt);
        SWAP(vY_prev, vY); diffuse(Y, vY, vY_prev, visc, dt);
        project(vX, vY, vX_prev, vY_prev);

        SWAP(vX_prev, vX); SWAP(vY_prev, vY);
        advect(X, vX, vX_prev, vX_prev, vY_prev, dt);
        advect(Y, vY, vY_prev, vX_prev, vY_prev, dt);

        project(vX, vY, vX_prev, vY_prev);
    }

    // Density
    {
        SWAP(density_prev, density); diffuse(ZERO, density, density_prev, diff, dt);
        SWAP(density_prev, density); advect(ZERO, density, density_prev, vX, vY, dt);
    }

    fadeDensity(density);

    CalculateVorticity(vX, vY, vorticity);
}

void FluidMatrix::OMP_step() {
    /*// Velocity*/
    /*{*/
    /*    OMP_diffuse(X, vX_prev, vX, visc, dt);*/
    /*    OMP_diffuse(Y, vY_prev, vY, visc, dt);*/
    /**/
    /*    OMP_project(vX_prev, vY_prev, vX, vY);*/
    /**/
    /*    OMP_advect(X, vX, vX_prev, vX_prev, vY_prev, dt);*/
    /*    OMP_advect(Y, vY, vY_prev, vX_prev, vY_prev, dt);*/
    /*}*/
    /**/
    /*// Density*/
    /*{*/
    /*    OMP_diffuse(ZERO, density_prev, density, diff, dt);*/
    /*    OMP_advect(ZERO, density, density_prev, vX, vY, dt);*/
    /*}*/
    /*OMP_fadeDensity(density);*/

    // Velocity
    {
        SWAP(vX_prev, vX); OMP_diffuse(X, vX, vX_prev, visc, dt);
        SWAP(vY_prev, vY); OMP_diffuse(Y, vY, vY_prev, visc, dt);
        OMP_project(vX, vY, vX_prev, vY_prev);

        SWAP(vX_prev, vX); SWAP(vY_prev, vY);
        OMP_advect(X, vX, vX_prev, vX_prev, vY_prev, dt);
        OMP_advect(Y, vY, vY_prev, vX_prev, vY_prev, dt);

        OMP_project(vX, vY, vX_prev, vY_prev);
    }

    // Density
    {
        SWAP(density_prev, density); OMP_diffuse(ZERO, density, density_prev, diff, dt);
        SWAP(density_prev, density); OMP_advect(ZERO, density, density_prev, vX, vY, dt);
    }

    fadeDensity(density);
}

void FluidMatrix::addDensity(uint32_t x, uint32_t y, double amount) { this->density[index(y, x, this->size)] += amount; }

void FluidMatrix::addVelocity(uint32_t x, uint32_t y, double amountX, double amountY) {
    uint32_t idx = index(y, x, this->size);

    this->vX[idx] += amountY;
    this->vY[idx] += amountX;
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
    int i, j, i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1, dt0;

    dt0 = dt*(this->size-2);

    for (int i = 1; i < this->size - 1; i++) {
        for (int j = 1; j < this->size - 1; j++) {
                    x = i-dt0*vX[index(i,j, this->size)]; y = j-dt0*vY[index(i,j, this->size)];
                    if (x<0.5f) x=0.5f; if (x>this->size-2+0.5f) x=this->size-2+0.5f; i0=(int)x; i1=i0+1;
                    if (y<0.5f) y=0.5f; if (y>this->size-2+0.5f) y=this->size-2+0.5f; j0=(int)y; j1=j0+1;
                    s1 = x-i0; s0 = 1-s1; t1 = y-j0; t0 = 1-t1;
                    d[index(i,j, this->size)] = s0*(t0*d0[index(i0,j0, this->size)]+t1*d0[index(i0,j1, this->size)])+
                                             s1*(t0*d0[index(i1,j0, this->size)]+t1*d0[index(i1,j1, this->size)]);
        }
    }
    set_bnd (mode, d);
    
}

void FluidMatrix::OMP_advect(Axis mode, std::vector<double> &d, std::vector<double> &d0, std::vector<double> &vX, std::vector<double> &vY, double dt) const {
#pragma omp parallel num_threads(this->numMaxThreads)
    {
        int i, j, i0, j0, i1, j1;
        float x, y, s0, t0, s1, t1, dt0;

        dt0 = dt*(this->size-2);

#pragma omp for
        for (int i = 1; i < this->size - 1; i++) {
            for (int j = 1; j < this->size - 1; j++) {
                        x = i-dt0*vX[index(i,j, this->size)]; y = j-dt0*vY[index(i,j, this->size)];
                        if (x<0.5f) x=0.5f; if (x>this->size-2+0.5f) x=this->size-2+0.5f; i0=(int)x; i1=i0+1;
                        if (y<0.5f) y=0.5f; if (y>this->size-2+0.5f) y=this->size-2+0.5f; j0=(int)y; j1=j0+1;
                        s1 = x-i0; s0 = 1-s1; t1 = y-j0; t0 = 1-t1;
                        d[index(i,j, this->size)] = s0*(t0*d0[index(i0,j0, this->size)]+t1*d0[index(i0,j1, this->size)])+
                                                 s1*(t0*d0[index(i1,j0, this->size)]+t1*d0[index(i1,j1, this->size)]);
            }
        }
        OMP_set_bnd(mode, d);
    }
}

void FluidMatrix::project(std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &p, std::vector<double> &div) const {
    for (int i = 1; i < this->size - 1; i++) {
        for (int j = 1; j < this->size - 1; j++) {
            div[index(i, j, this->size)] =
                    -0.5f * (vX[index(i + 1, j, this->size)] - vX[index(i - 1, j, this->size)] + vY[index(i, j + 1, this->size)] - vY[index(i, j - 1, this->size)]) * (this->size - 2);
            p[index(i, j, this->size)] = 0;
        }
    }

    set_bnd(ZERO, div);
    set_bnd(ZERO, p);
    /*lin_solve(ZERO, p, div, 1);*/

    double cRecip = 1.0 / 4;

    for (int k = 0; k < ITERATIONS; k++) {
        for (int i = 1; i < this->size - 1; i++) {
            for (int j = 1; j < this->size - 1; j++) {
                p[index(i, j, this->size)] = (div[index(i, j, this->size)] + (p[index(i + 1, j, this->size)] + p[index(i - 1, j, this->size)] +
                                                                            p[index(i, j + 1, this->size)] + p[index(i, j - 1, this->size)])) *
                                                 cRecip;
            }
        }
        set_bnd(ZERO, p);
    }

    for (int i = 1; i < this->size - 1; i++) {
        for (int j = 1; j < this->size - 1; j++) {
            vX[index(i, j, this->size)] -= 0.5f * (p[index(i + 1, j, this->size)] - p[index(i - 1, j, this->size)]) / (this->size - 2);
            vY[index(i, j, this->size)] -= 0.5f * (p[index(i, j + 1, this->size)] - p[index(i, j - 1, this->size)]) / (this->size - 2);
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
/*#pragma omp single*/
/*        {*/
/*            omp_set_nested(1);*/
/*            OMP_lin_solve(ZERO, p, div, 1);*/
/*        }*/


    double cRecip = 1.0 / 4;

    for (int k = 0; k < ITERATIONS; k++) {
        {
#pragma omp for schedule(guided) collapse(2)
            for (int i = 1; i < this->size - 1; i++) {
                for (int j = 1; j < this->size - 1; j++) {
                    p[index(i, j, this->size)] =
                            (div[index(i, j, this->size)] + (p[index(i + 1, j, this->size)] + p[index(i - 1, j, this->size)] +
                                                                                  p[index(i, j + 1, this->size)] + p[index(i, j - 1, this->size)])) *
                            cRecip;
                }
            }
            OMP_set_bnd(ZERO, p);
        }
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
    double c = diffusionRate;
    double cRecip = 1.0 / (1 + 4*c);

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
    double c = diffusionRate;
    double cRecip = 1.0 / (1 + 4 * c);

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

void FluidMatrix::CalculateVorticity(std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &vorticity) {
    const double h = 1.0 / (this->size - 2); // assuming unit length domain
    for (int i = 1; i < this->size - 1; i++) {
        for (int j = 1; j < this->size - 1; j++) {
            int idx = index(i, j, this->size);
            double dv_dx = (vY[index(i+1, j, this->size)] - vY[index(i-1, j, this->size)]) / (2 * h);
            double du_dy = (vX[index(i, j+1, this->size)] - vX[index(i, j-1, this->size)]) / (2 * h);
            vorticity[idx] = dv_dx - du_dy;
        }
    }
}
