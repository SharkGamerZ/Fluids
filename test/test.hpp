#ifndef TEST_HPP
#define TEST_HPP

#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <omp.h>

#define IX(i, j) ((j) + (i) * N) ///< index of the matrix

enum Axis {
    X, Y, ZERO
}; ///< axis of the matrix



float randFloat();
void omp_lin_solve(int N, Axis mode, std::vector<float> &nextValue, std::vector<float> &value, float diffusionRate);
void set_bnd(int N, Axis mode, std::vector<float> &attr);

#endif
