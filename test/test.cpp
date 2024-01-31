#include "test.hpp"


int ITERATIONS = 20; ///< number of iterations




int main() {
    const int matrixSize = 2;


    std::srand(unsigned(std::time(nullptr)));
    std::vector<float> v(matrixSize * matrixSize);
    std::generate(v.begin(), v.end(), randFloat);

    for (auto &i : v)
        std::cout<<i<<std::endl;


    return 0;
}

float randFloat() {
    return ((double) rand() / (RAND_MAX));
}

void omp_lin_solve(int N, Axis mode, std::vector<float> &nextValue, std::vector<float> &value, float diffusionRate) {
    float c = 1 + 4 * diffusionRate;
    float cRecip = 1.0 / c;
    #pragma omp parallel
    {
        for (int k = 0; k < ITERATIONS; k++)
        {
            #pragma omp for collapse(2)
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
            set_bnd(N, mode, nextValue);
        }
    }
}

void set_bnd(int N, Axis mode, std::vector<float> &attr) {
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