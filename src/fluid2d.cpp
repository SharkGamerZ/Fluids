#include "fluid2d.hpp"



// Funzione per creare una matrice di fluidi
// @param size La dimensione della matrice
// @param diffusion La diffusione
// @param viscosity La viscosità
// @param dt Il delta time
// @return Il puntatore alla matrice
FluidMatrix *FluidMatrixCreate(int size, int diffusion, int viscosity, float dt)
{
    FluidMatrix *matrix = (FluidMatrix*) malloc(sizeof(*matrix));
    int N = size;
    
    matrix->size = size;
    matrix->dt = dt;
    matrix->diff = diffusion;
    matrix->visc = viscosity;
    
    matrix->s = (float*) calloc(N * N, sizeof(float));
    matrix->density = (float*) calloc(N * N, sizeof(float));
    
    // Velocities
    matrix->Vx = (float*) calloc(N * N, sizeof(float));
    matrix->Vy = (float*) calloc(N * N, sizeof(float));
    
    // Velocities at previous step
    matrix->Vx0 = (float*) calloc(N * N, sizeof(float));
    matrix->Vy0 = (float*) calloc(N * N, sizeof(float));
    
    return matrix;
}

// Funzione per liberare la memoria della matrice
// @param matrix La matrice da liberare
void FluidMatrixFree(FluidMatrix *matrix)
{
    free(matrix->s);
    free(matrix->density);
    
    free(matrix->Vx);
    free(matrix->Vy);
    
    free(matrix->Vx0);
    free(matrix->Vy0);
    
    free(matrix);
}


// Funzione per simulare un timestep
// @param matrix La matrice da simulare
void FluidMatrixStep(FluidMatrix *matrix)
{
    int N          = matrix->size;
    float visc     = matrix->visc;
    float diff     = matrix->diff;
    float dt       = matrix->dt;
    float *Vx      = matrix->Vx;
    float *Vy      = matrix->Vy;
    float *Vx0     = matrix->Vx0;
    float *Vy0     = matrix->Vy0;
    float *s       = matrix->s;
    float *density = matrix->density;
    
    // SWAP(Vx, Vx0); diffuse(xAxis, Vx, Vx0, visc, dt, N);
    // SWAP(Vy, Vy0); diffuse(yAxis, Vy, Vy0, visc, dt, N);
    
    // project(Vx0, Vy0, Vx, Vy, N);
    
    // advect(xAxis, Vx, Vx0, Vx0, Vy0, dt, N);
    // advect(yAxis, Vy, Vy0, Vx0, Vy0, dt, N);
    
    // project(Vx, Vy, Vx0, Vy0, N);
    
    SWAP (s, density); diffuse(0, s, density, diff, dt, N);
    // advect(0, density, s, Vx, Vy, dt, N);
}

// Funzione per aggiungere densità in un punto
void FluidMatrixAddDensity(FluidMatrix *matrix, int x, int y, float amount)
{
    int N = matrix->size;
    matrix->density[IX(x, y)] += amount;
}

// Funzione per aggiungere velocità in un punto
void FluidMatrixAddVelocity(FluidMatrix *matrix, int x, int y, float amountX, float amountY)
{
    int N = matrix->size;
    int index = IX(x, y);
    
    matrix->Vx[index] += amountX;
    matrix->Vy[index] += amountY;
}


// Funzione per simulare la diffusione in un timestep
// @param mode Se è xAxis o yAxis
// @param value Il valore da diffondere
// @param oldValue Il valore al timestep precedente
// @param diffusion La diffusione
static void diffuse (int mode, float *value, float *oldValue, float diffusion, float dt, int N)
{
    float diffusionRate = dt * diffusion * N * N;
    lin_solve(mode, value, oldValue, diffusionRate, N);
}

static void advect(int mode, float *d, float *d0,  float *velocX, float *velocY, float dt, int N)
{
    // Indici al timestep precedente
    float i0, i1, j0, j1;
    

    float dtx = dt * (N - 2);
    float dty = dt * (N - 2);
    
    float s0, s1, t0, t1;
    float tmp1, tmp2, x, y;
    
    float Nfloat = (float) N;
    float ifloat, jfloat;
    int i, j;
    
    for(j = 1, jfloat = 1; j < N - 1; j++, jfloat++) { 
        for(i = 1, ifloat = 1; i < N - 1; i++, ifloat++) {
            tmp1 = dtx * velocX[IX(i, j)];
            tmp2 = dty * velocY[IX(i, j)];
            x    = ifloat - tmp1; 
            y    = jfloat - tmp2;
            
            if(x < 0.5f) x = 0.5f; 
            if(x > Nfloat + 0.5f) x = Nfloat + 0.5f; 
            i0 = floorf(x); 
            i1 = i0 + 1.0f;
            if(y < 0.5f) y = 0.5f; 
            if(y > Nfloat + 0.5f) y = Nfloat + 0.5f; 
            j0 = floorf(y);
            j1 = j0 + 1.0f; 
            
            s1 = x - i0; 
            s0 = 1.0f - s1; 
            t1 = y - j0; 
            t0 = 1.0f - t1;
            
            int i0i = (int) i0;
            int i1i = (int) i1;
            int j0i = (int) j0;
            int j1i = (int) j1;
            
            d[IX(i, j)] = 
                s0 * ( t0 *  d0[IX(i0i, j0i)] +  t1 *  d0[IX(i0i, j1i)]) +
                s1 * ( t0 *  d0[IX(i1i, j0i)] +  t1 *  d0[IX(i1i, j1i)]);
        }
    }
    set_bnd(mode, d, N);
}

static void project(float *velocX, float *velocY, float *p, float *div, int N)
{
    for (int j = 1; j < N - 1; j++) {
        for (int i = 1; i < N - 1; i++) {
            div[IX(i, j)] = -0.5f*(
                         velocX[IX(i+1, j  )]
                        -velocX[IX(i-1, j  )]
                        +velocY[IX(i  , j+1)]
                        -velocY[IX(i  , j-1)]
                )/N;
            p[IX(i, j)] = 0;
        }
    }
    set_bnd(0, div, N); 
    set_bnd(0, p, N);
    lin_solve(0, p, div, 1, N);
    
    for (int j = 1; j < N - 1; j++) {
        for (int i = 1; i < N - 1; i++) {
            velocX[IX(i, j)] -= 0.5f * (  p[IX(i+1, j)]
                                         -p[IX(i-1, j)]) * N;
            velocY[IX(i, j)] -= 0.5f * (  p[IX(i, j+1)]
                                         -p[IX(i, j-1)]) * N;
        }
    }
    set_bnd(xAxis, velocX, N);
    set_bnd(yAxis, velocY, N);
}


// Questa funzione gestisce gli "edge case", in caso siamo su un bordo
// la velocità viene invertita, e il fluido "rimbalza"
// @param mode Se è xAxis o yAxis
// @param attr L'attributo da gestire
// @param N La dimensione della matrice
static void set_bnd(int mode, float *attr, int N)
{

    for(int i = 1; i < N - 1; i++) {
        attr[IX(i, 0  )] = mode == yAxis ? -attr[IX(i, 1  )] : attr[IX(i, 1  )];
        attr[IX(i, N-1)] = mode == yAxis ? -attr[IX(i, N-2)] : attr[IX(i, N-2)];
    }
    for(int j = 1; j < N - 1; j++) {
        attr[IX(0  , j)] = mode == xAxis ? -attr[IX(1  , j)] : attr[IX(1  , j)];
        attr[IX(N-1, j)] = mode == xAxis ? -attr[IX(N-2, j)] : attr[IX(N-2, j)];
    }


    attr[IX(0, 0)]       = 0.5f * (attr[IX(1, 0)    ] + attr[IX(0, 1)    ]);
    attr[IX(0, N-1)]     = 0.5f * (attr[IX(1, N-1)  ] + attr[IX(0, N-2)  ]);

    attr[IX(N-1, 0)]     = 0.5f * (attr[IX(N-2, 0)  ] + attr[IX(N-1, 1)  ]);
    attr[IX(N-1, N-1)]   = 0.5f * (attr[IX(N-2, N-1)] + attr[IX(N-1, N-2)]);
}

// Funzione per risolvere un sistema lineare
// @param mode Se è xAxis o yAxis
// @param value Il valore da risolvere
// @param oldValue Il valore al timestep precedente
// @param diffusionRate Il rate a cui avviene la diffusione
static void lin_solve(int mode, float *value, float *oldValue, float diffusionRate, int N)
{
    float c = 1 + 4 * diffusionRate;
    float cRecip = 1.0 / c;
    for (int k = 0; k < iterations; k++) {
        for (int j = 1; j < N - 1; j++) {
            for (int i = 1; i < N - 1; i++) {
                value[IX(i, j)] =(oldValue[IX(i, j)] + diffusionRate*(   value[IX(i+1, j  )] + value[IX(i-1, j  )]
                                                                        +value[IX(i  , j+1)] + value[IX(i  , j-1)])) * cRecip; }
            }
        set_bnd(mode, value, N);
    }
}

