#ifndef TEST_HPP
#define TEST_HPP

#include "../src/utils.hpp"

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <math.h>
#include <ctime>
#include <chrono>
#include <omp.h>

#define IX(i, j) ((j) + (i) * N) ///< index of the matrix

enum Axis {
    X, Y, ZERO
}; ///< axis of the matrix

#define ITERATIONS 10 ///< number of iterations


float randFloat();
bool float_equals(float a, float b, float epsilon = 0.0001);


void diffuse(int N, Axis mode, std::vector<float> &value, std::vector<float> &oldValue, float diffusion, float dt);
void omp_diffuse(int N, Axis mode, std::vector<float> &value, std::vector<float> &oldValue, float diffusion, float dt);
void cuda_diffuse(int N, Axis mode, std::vector<float> &value, std::vector<float> &oldValue, float diffusion, float dt);

void lin_solve(int N, Axis mode, std::vector<float> &nextValue, std::vector<float> &value, float diffusionRate);
void omp_lin_solve(int N, Axis mode, std::vector<float> &nextValue, std::vector<float> &value, float diffusionRate);
__global__ void kernel_lin_solve(int N, Axis mode, float* nextValue, float* value, float diffusionRate);


void set_bnd(int N, Axis mode, std::vector<float> &attr);
__device__ void kernel_set_bnd(int N, Axis mode, float *attr);

#endif
