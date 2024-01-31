#include "test.hpp"


int ITERATIONS = 1; ///< number of iterations

int num_threads = 4;


int main() {
    const int maxSize = 1500;
    
    for (int matrixSize = 5; matrixSize < maxSize; matrixSize+= 1)
    {
        omp_set_num_threads(num_threads);

        std::srand(unsigned(std::time(nullptr)));
        std::vector<float> value(matrixSize * matrixSize);
        std::generate(value.begin(), value.end(), randFloat);
        std::vector<float> valueCopy(matrixSize * matrixSize);
        std::copy(value.begin(), value.end(), valueCopy.begin());

        std::vector<float> oldValue(matrixSize * matrixSize);
        std::generate(oldValue.begin(), oldValue.end(), randFloat);
        std::vector<float> oldValueCopy(matrixSize * matrixSize);
        std::copy(oldValue.begin(), oldValue.end(), oldValueCopy.begin());

        double dt = 0.1;
        double diff = 0.1;

        int64_t serialTimeMean = 0;
        int64_t ompTimeMean = 0;

        std::cout << BOLD BLUE "Matrix size: " << RESET << matrixSize << std::endl;
        for (int i = 0; i < 10; i++) {
            auto serialBegin = std::chrono::high_resolution_clock::now();
            diffuse(matrixSize, Axis::ZERO, value, oldValue, diff, dt);
            auto serialEnd = std::chrono::high_resolution_clock::now();
            auto serialTime = std::chrono::duration_cast<std::chrono::microseconds>(serialEnd - serialBegin).count();

            serialTimeMean += serialTime;


            auto ompBegin = std::chrono::high_resolution_clock::now();
            omp_diffuse(matrixSize, Axis::ZERO, valueCopy, oldValueCopy, diff, dt);
            auto ompEnd = std::chrono::high_resolution_clock::now();
            auto ompTime = std::chrono::duration_cast<std::chrono::microseconds>(ompEnd - ompBegin).count();

            ompTimeMean += ompTime;

            for (int i = 0; i < value.size(); i++) {
                if (!float_equals(value[i],valueCopy[i])) {
                    std::cout << BOLD RED "ERROR: " << RESET << "Serial and OMP results are different at cell "<< i << std::endl;
                    std::cout << "Serial: " << value[i] << " OMP: " << valueCopy[i] << std::endl;
                    return 1;
                }
            }
        }

        serialTimeMean /= 10;
        ompTimeMean /= 10;

        std::cout << BOLD YELLOW "Diffuse: " << serialTimeMean << RESET " micros ";
        std::cout << BOLD RED "OMP Diffuse: " << ompTimeMean << RESET " micros" << std::endl;

        double speedUp = (double) serialTimeMean / (double) ompTimeMean;
        double efficiency = speedUp / num_threads;
        std::cout << BOLD BLUE "Speedup: " << RESET << speedUp << " ";
        std::cout << BOLD GREEN "Efficiency: " << RESET << efficiency << std::endl << std::endl;


        std::cout << std::endl << std::endl;
    }



    return 0;
}

float randFloat() {
    return ((double) rand() / (RAND_MAX));
}

bool float_equals(float a, float b, float epsilon)
{
    return std::abs(a - b) < epsilon;
}

void diffuse(int N, Axis mode, std::vector<float> &value, std::vector<float> &oldValue, float diffusion, float dt) {
    float diffusionRate = dt * diffusion * N * N;

    lin_solve(N, mode, value, oldValue, diffusionRate);
}

void omp_diffuse(int N, Axis mode, std::vector<float> &value, std::vector<float> &oldValue, float diffusion, float dt) {
    float diffusionRate = dt * diffusion * N * N;

    omp_lin_solve(N, mode, value, oldValue, diffusionRate);
}


void lin_solve(int N, Axis mode, std::vector<float> &nextValue, std::vector<float> &value, float diffusionRate) {
    float c = 1 + 4 * diffusionRate;
    float cRecip = 1.0 / c;
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
        set_bnd(N, mode, nextValue);
    }
}

void omp_lin_solve(int N, Axis mode, std::vector<float> &nextValue, std::vector<float> &value, float diffusionRate) {
    float c = 1 + 4 * diffusionRate;
    float cRecip = 1.0 / c;
    for (int k = 0; k < ITERATIONS; k++)
    {
        #pragma omp parallel for default(shared)
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