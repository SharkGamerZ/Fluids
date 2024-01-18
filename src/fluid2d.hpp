#include <math.h>

#define IX(i , j) ((j) + (i) * N)
#define xAxis 1
#define yAxis 2
#define iterations 20


struct FluidMatrix {
    int size;
    float dt;
    float diff;
    float visc;
    
    float *s;
    float *density;
    
    float *Vx;
    float *Vy;

    float *Vx0;
    float *Vy0;
};


FluidMatrix *FluidMatrixCreate(int size, int diffusion, int viscosity, float dt);
void FluidMatrixFree(FluidMatrix *matrix);

void FluidMatrixStep(FluidMatrix *matrix);
void FluidMatrixAddDensity(FluidMatrix *matrix, int x, int y, float amount);
void FluidMatrixAddVelocity(FluidMatrix *matrix, int x, int y, float amountX, float amountY);

static void diffuse (int mode, float *value, float *oldValue, float diffusion, float dt, int N);
static void advect(int mode, float *d, float *d0,  float *velocX, float *velocY, float dt, int N);
static void project(float *velocX, float *velocY, float *p, float *div, int N);


static void set_bnd(int mode, float *attr, int N);
static void lin_solve(int mode, float *value, float *oldValue, float diffusionRate, int N);