#ifndef TEST_HPP
#define TEST_HPP

#include "../src/utils.hpp"

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <chrono>
#include <omp.h>

#define IX(i, j) ((j) + (i) * N) ///< index of the matrix

uint32_t index(uint32_t i, uint32_t j, uint32_t matrix_size) {
    return j + i * matrix_size;
}

enum Axis {
    X, Y, ZERO
}; ///< axis of the matrix

#define ITERATIONS 10 ///< number of iterations

void testDiffuse(int maxSize, int iterations);
void testAdvect(int maxSize, int iterations);


double randdouble();
bool double_equals(double a, double b, double epsilon = 0.0001);


void diffuse(int N, Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusion, double dt);
void omp_diffuse(int N, Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusion, double dt);
void cuda_diffuse(int N, Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusion, double dt);

void advect(int N, Axis mode, std::vector<double> &value, std::vector<double> &oldValue, std::vector<double> &vX, std::vector<double> &vY, double dt);
void omp_advect(int N, Axis mode, std::vector<double> &value, std::vector<double> &oldValue, std::vector<double> &vX, std::vector<double> &vY, double dt, int * trdN);



void lin_solve(int N, Axis mode, std::vector<double> &nextValue, std::vector<double> &value, double diffusionRate);
void omp_lin_solve(int N, Axis mode, std::vector<double> &nextValue, std::vector<double> &value, double diffusionRate);
__global__ void kernel_lin_solve(int N, Axis mode, double* nextValue, double* value, double diffusionRate);


void set_bnd(int N, Axis mode, std::vector<double> &attr);
__device__ void kernel_set_bnd(int N, Axis mode, double *attr);

void omp_set_bnd(int N, Axis mode, std::vector<double> &attr);
__device__ void kernel_set_bnd(int N, Axis mode, double *attr);

#endif
