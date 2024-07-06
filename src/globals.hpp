#pragma once

#include <omp.h>

#define SERIAL 0
#define OPENMP 1
#define CUDA 2
extern int numThreads;

void initGlobals();
