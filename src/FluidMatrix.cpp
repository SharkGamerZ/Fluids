#include "FluidMatrix.h"

FluidMatrix::FluidMatrix(int size, float diffusion, float viscosity, float dt) :
        size(size),
        dt(dt),
        diff(diffusion),
        visc(viscosity),
        density(std::vector<float>(size * size)),
        density0(std::vector<float>(size * size)),
        Vx(std::vector<float>(size * size)),
        Vy(std::vector<float>(size * size)),
        Vx0(std::vector<float>(size * size)),
        Vy0(std::vector<float>(size * size)) {
    std::cout << "DEBUG: FluidMatrix created" << std::endl;
}

FluidMatrix::~FluidMatrix() {
    std::cout << "DEBUG: FluidMatrix destroyed" << std::endl;
}

std::ostream &operator<<(std::ostream &os, const FluidMatrix &matrix) {
    os << "FluidMatrix {\n"
       << "\tsize: " << matrix.size
       << ",\n\tdt: " << matrix.dt
       << ",\n\tdiff: " << matrix.diff
       << ",\n\tvisc: " << matrix.visc
       << ",\n\tdensity: " << matrix.density.size()
       << ",\n\tdensity0: " << matrix.density0.size()
       << ",\n\tVx: " << matrix.Vx.size()
       << ",\n\tVy: " << matrix.Vy.size()
       << ",\n\tVx0: " << matrix.Vx0.size()
       << ",\n\tVy0: " << matrix.Vy0.size()
       << "\n}\n";
    return os;
}


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

    // velocity step
    {
        std::swap(Vx0, Vx);
        diffuse(xAxis, Vx, Vx0, visc, dt);

        std::swap(Vy0, Vy);
        diffuse(yAxis, Vy, Vy0, visc, dt);
// project va fatto solo sui valori aggiornati (questo creava il buco nero)
//           project(Vx0, Vy0, Vx, Vy);
         project(Vx, Vy, Vx0, Vy0);

        std::swap(Vx0, Vx);
        advect(xAxis, Vx, Vx0, Vx0, Vy0, dt);
        std::swap(Vy0, Vy);
        advect(yAxis, Vy, Vy0, Vx0, Vy0, dt);

         project(Vx, Vy, Vx0, Vy0);
    }

    // density step
    {
        std::swap(density0, density);
        diffuse(0, density, density0, diff, dt);

        std::swap(density0, density);
        advect(0, density, density0, Vx, Vy, dt);
    }
}




void FluidMatrix::addDensity(int x, int y, float amount) {
    int N = this->size;
    this->density[IX(x, y)] += amount;
}

void FluidMatrix::addVelocity(int x, int y, float amountX, float amountY) {
    int N = this->size;
    int index = IX(x, y);

    this->Vx[index] += amountX;
    this->Vy[index] += amountY;
}

// GIUSTA
void FluidMatrix::diffuse(int mode, std::vector<float> &value, std::vector<float> &oldValue, float diffusion, float dt) const {
    // std::vector<float> tmp = value;
    int N = this->size;
    float diffusionRate = dt * diffusion * N * N;
    lin_solve(mode, value, oldValue, diffusionRate);
    // std::swap(oldValue, value);
    // value = std::move(tmp);
}

// DA controllare come stanno i float (se sono tutti float o se ci sono anche int)
void FluidMatrix::advect(int mode, std::vector<float> &value, std::vector<float> &oldValue, std::vector<float> &vX, std::vector<float> &vY, float dt) const {
    int N = this->size;

    // indexes at the previous step
    int i0, j0, i1, j1;
    //tolto tutti gli (N-2 con solo N come in diffuse)
    float dt0 = dt * N;

    float s0, s1, t0, t1;
    float tmp1, tmp2, x, y;

    int i, j;

    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            x = i - (dt0 * vX[IX(i, j)]);
            y = j - (dt0 * vY[IX(i, j)]);

            if (x < 0.5f) x = 0.5f;
            if (x > N  + 0.5f) x = N + 0.5f;
            i0 = (int) x;
            i1 = i0 + 1;

            if (y < 0.5f) y = 0.5f;
            if (y > N+ 0.5f) y = N + 0.5f;
            j0 = (int) y;
            j1 = j0 + 1;

            s1 = x - i0;
            s0 = 1 - s1;
            t1 = y - j0;
            t0 = 1 - t1;

            value[IX(i, j)] =   s0 * (t0 * oldValue[IX(i0, j0)] + t1 * oldValue[IX(i0, j1)]) +
                                s1 * (t0 * oldValue[IX(i1, j0)] + t1 * oldValue[IX(i1, j1)]);
        }
    }

    set_bnd(mode, value);
}




void FluidMatrix::project(std::vector<float> &vX, std::vector<float> &vY, std::vector<float> &p, std::vector<float> &div) const {
    int N = this->size;
    for (int j = 1; j < N - 1; j++) {
        for (int i = 1; i < N - 1; i++) {
            div[IX(i, j)] = -0.5f * (
                    vX[IX(i + 1, j)]
                    - vX[IX(i - 1, j)]
                    + vY[IX(i, j + 1)]
                    - vY[IX(i, j - 1)]
            ) / N;
            p[IX(i, j)] = 0;
        }
    }
    set_bnd(0, div);
    set_bnd(0, p);
    lin_solve(0, p, div, 1);

    for (int j = 1; j < N - 1; j++) {
        for (int i = 1; i < N - 1; i++) {
            vX[IX(i, j)] -= 0.5f * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) * N;
            vY[IX(i, j)] -= 0.5f * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) * N;
        }
    }
    set_bnd(xAxis, vX);
    set_bnd(yAxis, vY);
}


// IN TEORIA GIUSTA
void FluidMatrix::set_bnd(int mode, std::vector<float> &attr) const {
    int N = this->size;
    for (int i = 1; i < N - 1; i++) {
        attr[IX(i, 0    )] = mode == yAxis ? -attr[IX(i, 1)] : attr[IX(i, 1)];
        attr[IX(i, N - 1)] = mode == yAxis ? -attr[IX(i, N - 2)] : attr[IX(i, N - 2)];
    }
    for (int j = 1; j < N - 1; j++) {
        attr[IX(0, j    )] = mode == xAxis ? -attr[IX(1, j)] : attr[IX(1, j)];
        attr[IX(N - 1, j)] = mode == xAxis ? -attr[IX(N - 2, j)] : attr[IX(N - 2, j)];
    }

    attr[IX(0    , 0    )] = 0.5f * (attr[IX(1, 0)] + attr[IX(0, 1)]);
    attr[IX(0    , N - 1)] = 0.5f * (attr[IX(1, N - 1)] + attr[IX(0, N - 2)]);

    attr[IX(N - 1, 0    )] = 0.5f * (attr[IX(N - 2, 0)] + attr[IX(N - 1, 1)]);
    attr[IX(N - 1, N - 1)] = 0.5f * (attr[IX(N - 2, N - 1)] + attr[IX(N - 1, N - 2)]);
}

// GIUSTA
void FluidMatrix::lin_solve(int mode, std::vector<float> &nextValue, std::vector<float> &value, float diffusionRate) const {
    int N = this->size;
    float c = 1 + 4 * diffusionRate;
    float cRecip = 1.0 / c;
    for (int k = 0; k < ITERATIONS; k++) {
        for (int j = 1; j < N - 1; j++) {
            for (int i = 1; i < N - 1; i++) {
                nextValue[IX(i, j)] = (value[IX(i, j)]
                                   + diffusionRate * (
                        nextValue[IX(i + 1, j)]
                        + nextValue[IX(i - 1, j)]
                        + nextValue[IX(i, j + 1)]
                        + nextValue[IX(i, j - 1)]
                )) * cRecip;
            }
        }
        set_bnd(mode, nextValue);
    }
}


