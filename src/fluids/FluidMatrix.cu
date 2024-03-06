#include "FluidMatrix.hu"


FluidMatrix::FluidMatrix(uint32_t size, double diffusion, double viscosity, double dt)
    : size(size), dt(dt), diff(diffusion), visc(viscosity), density(std::vector<double>(size * size)), density0(std::vector<double>(size * size)),
      Vx(std::vector<double>(size * size)), Vy(std::vector<double>(size * size)), Vx0(std::vector<double>(size * size)), Vy0(std::vector<double>(size * size)) {
    Utils::log(Utils::LogLevel::DEBUG, std::cout, "FluidMatrix created");
}

FluidMatrix::~FluidMatrix() { Utils::log(Utils::LogLevel::DEBUG, std::cout, "FluidMatrix destroyed"); }

void FluidMatrix::reset() {
    // Imposta tutti i vettori della matrice a 0
    std::fill(density.begin(), density.end(), 0);
    std::fill(density0.begin(), density0.end(), 0);
    std::fill(Vx.begin(), Vx.end(), 0);
    std::fill(Vy.begin(), Vy.end(), 0);
    std::fill(Vx0.begin(), Vx0.end(), 0);
    std::fill(Vy0.begin(), Vy0.end(), 0);
}

void FluidMatrix::step() {

    auto begin = std::chrono::high_resolution_clock::now();

    // velocity step 22ms
    {
        // 15 ms
        diffuse(Axis::X, Vx0, Vx, visc, dt);
        diffuse(Axis::Y, Vy0, Vy, visc, dt);

        project(Vx0, Vy0, Vx, Vy);


        // 7 ms
        advect(Axis::X, Vx, Vx0, Vx0, Vy0, dt);
        advect(Axis::Y, Vy, Vy0, Vx0, Vy0, dt);

        project(Vx, Vy, Vx0, Vy0);
    }


    // density step 7ms
    {
        diffuse(Axis::ZERO, density0, density, diff, dt);

        advect(Axis::ZERO, density, density0, Vx, Vy, dt);
    }

    fadeDensity(density);

    auto end = std::chrono::high_resolution_clock::now();
    Utils::log(Utils::LogLevel::DEBUG, std::cout, "Time: ", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count(), " micros");
}


void FluidMatrix::OMPstep() {
    auto begin = std::chrono::high_resolution_clock::now();

    // velocity step
    {
        omp_diffuse(Axis::X, Vx0, Vx, visc, dt);
        omp_diffuse(Axis::Y, Vy0, Vy, visc, dt);

        omp_project(Vx0, Vy0, Vx, Vy);


        omp_advect(Axis::X, Vx, Vx0, Vx0, Vy0, dt);
        omp_advect(Axis::Y, Vy, Vy0, Vx0, Vy0, dt);

        omp_project(Vx, Vy, Vx0, Vy0);
    }

    // density step
    {
        omp_diffuse(Axis::ZERO, density0, density, diff, dt);

        omp_advect(Axis::ZERO, density, density0, Vx, Vy, dt);
    }

    omp_fadeDensity(density);

    auto end = std::chrono::high_resolution_clock::now();
    Utils::log(Utils::LogLevel::DEBUG, std::cout, "Time: ", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count(), " micros");
}


void FluidMatrix::CUDAstep() {
    auto begin = std::chrono::high_resolution_clock::now();

    // velocity step
    {
        cuda_diffuse(Axis::X, Vx0, Vx, visc, dt);
        cuda_diffuse(Axis::Y, Vy0, Vy, visc, dt);

        project(Vx0, Vy0, Vx, Vy);


        advect(Axis::X, Vx, Vx0, Vx0, Vy0, dt);
        advect(Axis::Y, Vy, Vy0, Vx0, Vy0, dt);

        project(Vx, Vy, Vx0, Vy0);
    }

    // density step
    {
        cuda_diffuse(Axis::ZERO, density0, density, diff, dt);

        advect(Axis::ZERO, density, density0, Vx, Vy, dt);
    }

    fadeDensity(density);

    auto end = std::chrono::high_resolution_clock::now();
    Utils::log(Utils::LogLevel::DEBUG, std::cout, "Time: ", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count(), " micros");
}


void FluidMatrix::addDensity(uint32_t x, uint32_t y, double amount) { this->density[index(y, x, this->size)] += amount; }

void FluidMatrix::addVelocity(uint32_t x, uint32_t y, double amountX, double amountY) {
    uint32_t idx = index(y, x, this->size);

    this->Vx[idx] += amountX;
    this->Vy[idx] += amountY;
}

void FluidMatrix::diffuse(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusion, double dt) const {
    double diffusionRate = dt * diffusion * (this->size - 2) * (this->size - 2);

    lin_solve(mode, value, oldValue, diffusionRate);
}

void FluidMatrix::omp_diffuse(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusion, double dt) const {
    double diffusionRate = dt * diffusion * this->size * this->size;

    omp_lin_solve(mode, value, oldValue, diffusionRate);
}

void cuda_diffuse(int N, Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusion, double dt) {
    double diffusionRate = dt * diffusion * N * N;

    dim3 BlockSize(16, 16, 1);
    dim3 GridSize((N + 15) / 16, (N + 15) / 16, 1);


    double *d_value;
    double *d_oldValue;

    cudaMalloc(&d_value, N * N * sizeof(double));
    cudaMalloc(&d_oldValue, N * N * sizeof(double));

    cudaMemcpy(d_value, &value[0], N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldValue, &oldValue[0], N * N * sizeof(double), cudaMemcpyHostToDevice);


    for (int i = 0; i < ITERATIONS; i++) 
    {
        kernel_lin_solve<<<GridSize, BlockSize>>>(N, mode, &d_value[0], &d_oldValue[0], diffusionRate);
        kernel_set_bnd<<<1,1>>>(N, mode, &d_value[0]);
    }

    cudaMemcpy(&value[0], d_value, N * N * sizeof(double), cudaMemcpyDeviceToHost);




    cudaFree(d_value);
    cudaFree(d_oldValue);
}

/*
// DA controllare come stanno i double (se sono tutti double o se ci sono anche int)
void FluidMatrix::advect(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, std::vector<double> &vX, std::vector<double> &vY, double dt) const {
    int N = this->size;

    // indexes at the previous step
    int i0, j0, i1, j1;

    double dt0 = dt * (N - 2);

    double s0, s1, t0, t1;
    double x, y;

//  Per ogni cella vediamo da dove deve arrivare il value, tramite la prima formula. Poi visto che potrebbe arrivare da un punto
//  non esattamente su una cella, effettuiamo l'iterpolazione lineare tra le coordinate più vicine per capire quella del punto preciso.

    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            x = i - (dt0 * vX[index(i, j)]);
            y = j - (dt0 * vY[index(i, j)]);

            if (x < 0.5f) x = 0.5f;
            if (x > N - 2 + 0.5f) x = N - 2 + 0.5f;
            i0 = (int) x;
            i1 = i0 + 1;

            if (y < 0.5f) y = 0.5f;
            if (y > N - 2 + 0.5f) y = N - 2 + 0.5f;
            j0 = (int) y;
            j1 = j0 + 1;

            s1 = x - i0;
            s0 = 1 - s1;
            t1 = y - j0;
            t0 = 1 - t1;

            // printf(BOLD BLUE "vX:" RESET " %f, " BOLD BLUE "vY:" RESET " %f, ", vX[index(i, j)], vY[index(i, j)]);

            // printf(BOLD RED "x:" RESET " %f, " BOLD RED "y:" RESET " %f, ", x, y);
            // printf(BOLD YELLOW "i0:" RESET " %d, " BOLD YELLOW "i1:" RESET " %d, " BOLD YELLOW "j0:" RESET " %d, " BOLD YELLOW "j1:" RESET " %d\n", i0, i1, j0, j1);

            value[index(i, j)] =   s0 * (t0 * oldValue[index(i0, j0)] + t1 * oldValue[index(i0, j1)]) + s1 * (t0 * oldValue[index(i1, j0)] + t1 * oldValue[index(i1, j1)]);
        }
    }

    set_bnd(mode, value);
}
*/

void FluidMatrix::advect(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, std::vector<double> &vX, std::vector<double> &vY, double dt) const {
    double i0, i1, j0, j1;

    double dt0 = dt * (this->size - 2);

    double s0, s1, t0, t1;
    double tmp1, tmp2, x, y;

    double Ndouble = this->size - 2;

    for (uint32_t i = 1; i < this->size - 1; i++) {
        for (uint32_t j = 1; j < this->size - 1; j++) {
            double v1 = vX[index(i, j, this->size)];
            double v2 = vY[index(i, j, this->size)];
            tmp1 = dt0 * v1;
            tmp2 = dt0 * v2;
            x = (double) i - tmp1;
            y = (double) j - tmp2;

            if (x < 0.5f) x = 0.5f;
            if (x > Ndouble + 0.5f) x = Ndouble + 0.5f;
            i0 = floor(x);
            i1 = i0 + 1.0f;
            if (y < 0.5f) y = 0.5f;
            if (y > Ndouble + 0.5f) y = Ndouble + 0.5f;
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

            value[index(i, j, this->size)] = s0 * (t0 * oldValue[index(i0i, j0i, this->size)] + t1 * oldValue[index(i0i, j1i, this->size)]) +
                                             s1 * (t0 * oldValue[index(i1i, j0i, this->size)] + t1 * oldValue[index(i1i, j1i, this->size)]);
        }
    }
    set_bnd(mode, value);
}

void FluidMatrix::omp_advect(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, std::vector<double> &vX, std::vector<double> &vY, double dt) const {
    double dt0 = dt * (this->size - 2);
    double Ndouble = this->size - 2;


#pragma omp parallel num_threads(numThreads)
    {
        double i0, i1, j0, j1;
        double s0, s1, t0, t1;
        double tmp1, tmp2, x, y;

#pragma omp for
        for (int i = 1; i < this->size - 1; i++) {
            for (int j = 1; j < this->size - 1; j++) {
                double v1 = vX[index(i, j, this->size)];
                double v2 = vY[index(i, j, this->size)];
                tmp1 = dt0 * v1;
                tmp2 = dt0 * v2;
                x = (double) i - tmp1;
                y = (double) j - tmp2;

                if (x < 0.5f) x = 0.5f;
                if (x > Ndouble + 0.5f) x = Ndouble + 0.5f;
                i0 = floor(x);
                i1 = i0 + 1.0f;
                if (y < 0.5f) y = 0.5f;
                if (y > Ndouble + 0.5f) y = Ndouble + 0.5f;
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

                value[index(i, j, this->size)] = s0 * (t0 * oldValue[index(i0i, j0i, this->size)] + t1 * oldValue[index(i0i, j1i, this->size)]) +
                                                 s1 * (t0 * oldValue[index(i1i, j0i, this->size)] + t1 * oldValue[index(i1i, j1i, this->size)]);
            }
        }
        omp_set_bnd(mode, value);
    }
}


void FluidMatrix::project(std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &p, std::vector<double> &div) const {
    for (uint32_t i = 1; i < this->size - 1; i++) {
        for (uint32_t j = 1; j < this->size - 1; j++) {
            div[index(i, j, this->size)] =
                    -0.5f * (vX[index(i + 1, j, this->size)] - vX[index(i - 1, j, this->size)] + vY[index(i, j + 1, this->size)] - vY[index(i, j - 1, this->size)]) / this->size;
            p[index(i, j, this->size)] = 0;
        }
    }
    set_bnd(Axis::ZERO, div);
    set_bnd(Axis::ZERO, p);
    lin_solve(Axis::ZERO, p, div, 1);

    for (uint32_t i = 1; i < this->size - 1; i++) {
        for (uint32_t j = 1; j < this->size - 1; j++) {
            vX[index(i, j, this->size)] -= 0.5f * (p[index(i + 1, j, this->size)] - p[index(i - 1, j, this->size)]) * this->size;
            vY[index(i, j, this->size)] -= 0.5f * (p[index(i, j + 1, this->size)] - p[index(i, j - 1, this->size)]) * this->size;
        }
    }
    set_bnd(Axis::X, vX);
    set_bnd(Axis::Y, vY);
}

void FluidMatrix::omp_project(std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &p, std::vector<double> &div) const {
#pragma omp parallel default(shared) num_threads(numThreads)
    {
#pragma omp for schedule(guided) collapse(2)
        for (uint32_t i = 1; i < this->size - 1; i++) {
            for (uint32_t j = 1; j < this->size - 1; j++) {
                div[index(i, j, this->size)] =
                        -0.5f * (vX[index(i + 1, j, this->size)] - vX[index(i - 1, j, this->size)] + vY[index(i, j + 1, this->size)] - vY[index(i, j - 1, this->size)]) /
                        this->size;
                p[index(i, j, this->size)] = 0;
            }
        }
        omp_set_bnd(Axis::ZERO, div);
        omp_set_bnd(Axis::ZERO, p);
#pragma omp single
        {
            omp_set_nested(1);
            omp_lin_solve(Axis::ZERO, p, div, 1);
        }

#pragma omp for schedule(guided) collapse(2)
        for (uint32_t i = 1; i < this->size - 1; i++) {
            for (uint32_t j = 1; j < this->size - 1; j++) {
                vX[index(i, j, this->size)] -= 0.5f * (p[index(i + 1, j, this->size)] - p[index(i - 1, j, this->size)]) * this->size;
                vY[index(i, j, this->size)] -= 0.5f * (p[index(i, j + 1, this->size)] - p[index(i, j - 1, this->size)]) * this->size;
            }
        }
        omp_set_bnd(Axis::X, vX);
        omp_set_bnd(Axis::Y, vY);
    }
}


// IN TEORIA GIUSTA
void FluidMatrix::set_bnd(Axis mode, std::vector<double> &attr) const {
    for (uint32_t i = 1; i < this->size - 1; i++) {
        attr[index(i, 0, this->size)] = mode == Axis::Y ? -attr[index(i, 1, this->size)] : attr[index(i, 1, this->size)];
        attr[index(i, this->size - 1, this->size)] = mode == Axis::Y ? -attr[index(i, this->size - 2, this->size)] : attr[index(i, this->size - 2, this->size)];
    }
    for (uint32_t j = 1; j < this->size - 1; j++) {
        attr[index(0, j, this->size)] = mode == Axis::X ? -attr[index(1, j, this->size)] : attr[index(1, j, this->size)];
        attr[index(this->size - 1, j, this->size)] = mode == Axis::X ? -attr[index(this->size - 2, j, this->size)] : attr[index(this->size - 2, j, this->size)];
    }


    attr[index(0, 0, this->size)] = 0.5f * (attr[index(1, 0, this->size)] + attr[index(0, 1, this->size)]);
    attr[index(0, this->size - 1, this->size)] = 0.5f * (attr[index(1, this->size - 1, this->size)] + attr[index(0, this->size - 2, this->size)]);

    attr[index(this->size - 1, 0, this->size)] = 0.5f * (attr[index(this->size - 2, 0, this->size)] + attr[index(this->size - 1, 1, this->size)]);
    attr[index(this->size - 1, this->size - 1, this->size)] =
            0.5f * (attr[index(this->size - 2, this->size - 1, this->size)] + attr[index(this->size - 1, this->size - 2, this->size)]);
}

void FluidMatrix::omp_set_bnd(Axis mode, std::vector<double> &attr) const {
#pragma omp for
    for (uint32_t i = 1; i < this->size - 1; i++) {
        attr[index(i, 0, this->size)] = mode == Axis::Y ? -attr[index(i, 1, this->size)] : attr[index(i, 1, this->size)];
        attr[index(i, this->size - 1, this->size)] = mode == Axis::Y ? -attr[index(i, this->size - 2, this->size)] : attr[index(i, this->size - 2, this->size)];
    }
#pragma omp for
    for (uint32_t j = 1; j < this->size - 1; j++) {
        attr[index(0, j, this->size)] = mode == Axis::X ? -attr[index(1, j, this->size)] : attr[index(1, j, this->size)];
        attr[index(this->size - 1, j, this->size)] = mode == Axis::X ? -attr[index(this->size - 2, j, this->size)] : attr[index(this->size - 2, j, this->size)];
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

__global__ void FluidMatrix::kernel_set_bnd(int N, Axis mode, double *attr) {
    for (int i = 1; i < N - 1; i++) {
        attr[index(i, 0, N)] = mode == Axis::Y ? -attr[index(i, 1, N)] : attr[index(i, 1, N)];
        attr[index(i, N - 1, N)] = mode == Axis::Y ? -attr[index(i, N - 2, N)] : attr[index(i, N - 2, N)];
    }
    for (int j = 1; j < N - 1; j++) {
        attr[index(0, j, N)] = mode == Axis::X ? -attr[index(1, j, N)] : attr[index(1, j, N)];
        attr[index(N - 1, j, N)] = mode == Axis::X ? -attr[index(N - 2, j, N)] : attr[index(N - 2, j, N)];
    }


    attr[index(0, 0, N)] = 0.5f * (attr[index(1, 0, N)] + attr[index(0, 1, N)]);
    attr[index(0, N - 1, N)] = 0.5f * (attr[index(1, N - 1, N)] + attr[index(0, N - 2, N)]);

    attr[index(N - 1, 0, N)] = 0.5f * (attr[index(N - 2, 0, N)] + attr[index(N - 1, 1, N)]);
    attr[index(N - 1, N - 1, N)] = 0.5f * (attr[index(N - 2, N - 1, N)] + attr[index(N - 1, N - 2, N)]);
}



// GIUSTA
void FluidMatrix::lin_solve(Axis mode, std::vector<double> &nextValue, std::vector<double> &value, double diffusionRate) const {
    double c = 1 + 6 * diffusionRate;
    double cRecip = 1.0 / c;
    for (int k = 0; k < ITERATIONS; k++) {
        for (uint32_t i = 1; i < this->size - 1; i++) {
            for (uint32_t j = 1; j < this->size - 1; j++) {
                nextValue[index(i, j, this->size)] =
                        (value[index(i, j, this->size)] + diffusionRate * (nextValue[index(i + 1, j, this->size)] + nextValue[index(i - 1, j, this->size)] +
                                                                           nextValue[index(i, j + 1, this->size)] + nextValue[index(i, j - 1, this->size)])) *
                        cRecip;
            }
        }
        set_bnd(mode, nextValue);
    }
}

void FluidMatrix::omp_lin_solve(Axis mode, std::vector<double> &nextValue, std::vector<double> &value, double diffusionRate) const {
    double c = 1 + 4 * diffusionRate;
    double cRecip = 1.0 / c;

    for (int k = 0; k < ITERATIONS; k++) {

#pragma omp parallel default(shared) num_threads(numThreads)
        {
#pragma omp for schedule(guided) collapse(2)
            for (uint32_t i = 1; i < size - 1; i++) {
                for (uint32_t j = 1; j < size - 1; j++) {
                    nextValue[index(i, j, size)] = (value[index(i, j, size)] + diffusionRate * (nextValue[index(i + 1, j, size)] + nextValue[index(i - 1, j, size)] +
                                                                                                nextValue[index(i, j + 1, size)] + nextValue[index(i, j - 1, size)])) *
                                                   cRecip;
                }
            }
            omp_set_bnd(mode, nextValue);
        }
    }
}

__global__ void FluidMatrix::kernel_lin_solve(int N, Axis mode, double *nextValue, double *value, double diffusionRate) {


    double c = 1 + 4 * diffusionRate;
    double cRecip = 1.0 / c;

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col == 0 || col >= N - 1 || row == 0 || row >= N - 1) return;


    nextValue[index(row, col, N)] =
            (value[index(row, col, N)] + diffusionRate * (nextValue[index(row + 1, col, N)] + nextValue[index(row - 1, col, N)] + nextValue[index(row, col + 1, N)] + nextValue[index(row, col - 1, N])) *
            cRecip;

    // __syncthreads();
    // if (col == 1 && row == 1)
    //     kernel_set_bnd(N, mode, nextValue);
    // __syncthreads();
}


void FluidMatrix::fadeDensity(std::vector<double> &density) const {
    for (uint32_t i = 0; i < size * size; i++) {
        double d = density[i];
        density[i] = (d - 0.005f < 0) ? 0 : d - 0.005f;
    }
}


void FluidMatrix::omp_fadeDensity(std::vector<double> &density) const {
#pragma omp parallel for num_threads(numThreads) default(shared)
    for (uint32_t i = 0; i < this->size * this->size; i++) {
        double d = this->density[i];
        density[i] = (d - 0.0005f < 0) ? 0 : d - 0.0005f;
    }
}
