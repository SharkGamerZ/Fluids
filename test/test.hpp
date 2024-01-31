#ifndef TEST_HPP
#define TEST_HPP

#include "../src/utils.hpp"

#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <chrono>
#include <omp.h>

#define IX(i, j) ((j) + (i) * N) ///< index of the matrix

enum Axis {
    X, Y, ZERO
}; ///< axis of the matrix



float randFloat();
bool float_equals(float a, float b, float epsilon = 0.0001);


void diffuse(int N, Axis mode, std::vector<float> &value, std::vector<float> &oldValue, float diffusion, float dt);
void omp_diffuse(int N, Axis mode, std::vector<float> &value, std::vector<float> &oldValue, float diffusion, float dt);

void lin_solve(int N, Axis mode, std::vector<float> &nextValue, std::vector<float> &value, float diffusionRate);
void omp_lin_solve(int N, Axis mode, std::vector<float> &nextValue, std::vector<float> &value, float diffusionRate);

void set_bnd(int N, Axis mode, std::vector<float> &attr);

#endif
