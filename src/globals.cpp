#include "globals.hpp"

int numThreads;

void initGlobals() {
    numThreads = omp_get_max_threads();
}
