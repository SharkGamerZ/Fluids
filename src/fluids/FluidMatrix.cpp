#include "FluidMatrix.h"

FluidMatrix::FluidMatrix(int size, double diffusion, double viscosity, double dt) :
        size(size),
        dt(dt),
        diff(diffusion),
        visc(viscosity),
        density(std::vector<double>(size * size)),
        density0(std::vector<double>(size * size)),
        Vx(std::vector<double>(size * size)),
        Vy(std::vector<double>(size * size)),
        Vx0(std::vector<double>(size * size)),
        Vy0(std::vector<double>(size * size)) {
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
    auto begin = std::chrono::high_resolution_clock::now();

    // velocity step
    {
        diffuse(Axis::X, Vx0, Vx, visc, dt);
        diffuse(Axis::Y, Vy0, Vy, visc, dt);
        
        project(Vx0, Vy0, Vx, Vy);
        

        advect(Axis::X, Vx, Vx0, Vx0, Vy0, dt);
        advect(Axis::Y, Vy, Vy0, Vx0, Vy0, dt);

        project(Vx, Vy, Vx0, Vy0);
    }


        // density step
    {
        diffuse(Axis::ZERO, density0, density, diff, dt);

        advect(Axis::ZERO, density, density0, Vx, Vy, dt);
    }

    fadeDensity(density);

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << BOLD YELLOW "Total time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << RESET " micros" << std::endl;
}


void FluidMatrix::OMPstep() {
    auto begin = std::chrono::high_resolution_clock::now();

    // velocity step
    {
        omp_diffuse(Axis::X, Vx0, Vx, visc, dt);
        omp_diffuse(Axis::Y, Vy0, Vy, visc, dt);
        
        project(Vx0, Vy0, Vx, Vy);
        

        advect(Axis::X, Vx, Vx0, Vx0, Vy0, dt);
        advect(Axis::Y, Vy, Vy0, Vx0, Vy0, dt);

        project(Vx, Vy, Vx0, Vy0);
    }

        // density step
    {
        omp_diffuse(Axis::ZERO, density0, density, diff, dt);

        advect(Axis::ZERO, density, density0, Vx, Vy, dt);
    }

    fadeDensity(density);

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << BOLD YELLOW "Total time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << RESET " micros" << std::endl;

}


void FluidMatrix::addDensity(int x, int y, double amount) {
    int N = this->size;
    this->density[IX(y, x)] += amount;
}

void FluidMatrix::addVelocity(int x, int y, double amountX, double amountY) {
    int N = this->size;
    int index = IX(y, x);

    this->Vx[index] += amountX;
    this->Vy[index] += amountY;
}

void FluidMatrix::diffuse(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusion, double dt) const {
    int N = this->size;
    double diffusionRate = dt * diffusion * (N - 2) * (N - 2);

    lin_solve(mode, value, oldValue, diffusionRate);
}

void FluidMatrix::omp_diffuse(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusion, double dt) const {
    int N = this->size;
    double diffusionRate = dt * diffusion * N * N;

    omp_lin_solve(mode, value, oldValue, diffusionRate);
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
            x = i - (dt0 * vX[IX(i, j)]);
            y = j - (dt0 * vY[IX(i, j)]);

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

            // printf(BOLD BLUE "vX:" RESET " %f, " BOLD BLUE "vY:" RESET " %f, ", vX[IX(i, j)], vY[IX(i, j)]);
            
            // printf(BOLD RED "x:" RESET " %f, " BOLD RED "y:" RESET " %f, ", x, y);
            // printf(BOLD YELLOW "i0:" RESET " %d, " BOLD YELLOW "i1:" RESET " %d, " BOLD YELLOW "j0:" RESET " %d, " BOLD YELLOW "j1:" RESET " %d\n", i0, i1, j0, j1);

            value[IX(i, j)] =   s0 * (t0 * oldValue[IX(i0, j0)] + t1 * oldValue[IX(i0, j1)]) + s1 * (t0 * oldValue[IX(i1, j0)] + t1 * oldValue[IX(i1, j1)]);
        }
    }

    set_bnd(mode, value);
}
*/

void FluidMatrix::advect(Axis mode, std::vector<double> &value, std::vector<double> &oldValue, std::vector<double> &vX, std::vector<double> &vY, double dt) const {
    int N =this->size;
    double i0, i1, j0, j1;
    
    double dt0 = dt * (N - 2);

	double s0, s1, t0, t1;
	double tmp1, tmp2, x, y;

	double Ndouble = N - 2;
	double idouble, jdouble;

	int i, j;
    
	for(j = 1, jdouble = 1; j < N - 1; j++, jdouble++) { 
		for(i = 1, idouble = 1; i < N - 1; i++, idouble++) {
            double v1 = vX[IX(i, j)];
            double v2 = vY[IX(i, j)];
            tmp1 = dt0 * v1;
            tmp2 = dt0 * v2;
            x = idouble - tmp1; 
            y = jdouble - tmp2;
        
            if(x < 0.5f) x = 0.5f; 
            if(x > Ndouble + 0.5f) x = Ndouble + 0.5f; 
            i0 = floor(x); 
            i1 = i0 + 1.0f;
            if(y < 0.5f) y = 0.5f; 
            if(y > Ndouble + 0.5f) y = Ndouble + 0.5f; 
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
                
			value[IX(i, j)] = 
				s0 * (t0 * oldValue[IX(i0i, j0i)] + t1 * oldValue[IX(i0i, j1i)]) +
				s1 * (t0 * oldValue[IX(i1i, j0i)] + t1 * oldValue[IX(i1i, j1i)]);
            	}
        }
	set_bnd(mode, value);

}





void FluidMatrix::project(std::vector<double> &vX, std::vector<double> &vY, std::vector<double> &p, std::vector<double> &div) const {
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
    set_bnd(Axis::ZERO, div);
    set_bnd(Axis::ZERO, p);
    lin_solve(Axis::ZERO, p, div, 1);

    for (int j = 1; j < N - 1; j++) {
        for (int i = 1; i < N - 1; i++) {
            vX[IX(i, j)] -= 0.5f * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) * N;
            vY[IX(i, j)] -= 0.5f * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) * N;
        }
    }
    set_bnd(Axis::X, vX);
    set_bnd(Axis::Y, vY);
}


// IN TEORIA GIUSTA
void FluidMatrix::set_bnd(Axis mode, std::vector<double> &attr) const {
    int N = this->size;
    for (int i = 1; i < N - 1; i++) {
        attr[IX(i, 0    )] = mode == Axis::Y ? -attr[IX(i, 1)] : attr[IX(i, 1)];
        attr[IX(i, N - 1)] = mode == Axis::Y ? -attr[IX(i, N - 2)] : attr[IX(i, N - 2)];
    }
    for (int j = 1; j < N - 1; j++) {
        attr[IX(0, j    )] = mode == Axis::X ? -attr[IX(1, j)] : attr[IX(1, j)];
        attr[IX(N - 1, j)] = mode == Axis::X ? -attr[IX(N - 2, j)] : attr[IX(N - 2, j)];
    }


    attr[IX(0    , 0    )] = 0.5f * (attr[IX(1, 0)] + attr[IX(0, 1)]);
    attr[IX(0    , N - 1)] = 0.5f * (attr[IX(1, N - 1)] + attr[IX(0, N - 2)]);

    attr[IX(N - 1, 0    )] = 0.5f * (attr[IX(N - 2, 0)] + attr[IX(N - 1, 1)]);
    attr[IX(N - 1, N - 1)] = 0.5f * (attr[IX(N - 2, N - 1)] + attr[IX(N - 1, N - 2)]);
}

// GIUSTA
void FluidMatrix::lin_solve(Axis mode, std::vector<double> &nextValue, std::vector<double> &value, double diffusionRate) const {
    int N = this->size;
    double c = 1 + 6 * diffusionRate;
    double cRecip = 1.0 / c;
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

void FluidMatrix::omp_lin_solve(Axis mode, std::vector<double> &nextValue, std::vector<double> &value, double diffusionRate) const {
    int N = this->size;
    double c = 1 + 4 * diffusionRate;
    double cRecip = 1.0 / c;
    for (int k = 0; k < ITERATIONS; k++)
    {
        #pragma omp parallel for collapse(2) default(shared) schedule(static,1)
        for (int j = 1; j < N - 1; j++)
        {
            for (int i = 1; i < N - 1; i++)
            {
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



void FluidMatrix::fadeDensity(std::vector<double> &density) const {
    int N = this->size;
    for (int i = 0; i < N * N; i++) {
        double d = this->density[i];
        density[i] = (d - 0.005f < 0) ? 0 : d - 0.005f; 
    }
}

