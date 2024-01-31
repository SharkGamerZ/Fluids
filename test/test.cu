#include "test.hpp"



int max_cells;

int main() {
    const int maxSize = 500;
    
    for (int matrixSize = 250; matrixSize <= maxSize; matrixSize+= 50)
    {
        // SETUP ----------------------------------------------------------------------------------------------
        std::cout<<"------------------------------------------------------------------"<<std::endl;
        double dt = 0.1;
        double diff = 0.1;

        double speedup, efficiency;

        std::srand(unsigned(std::time(nullptr)));

        std::vector<float> value(matrixSize * matrixSize);
        std::generate(value.begin(), value.end(), randFloat);

        std::vector<float> oldValue(matrixSize * matrixSize);
        std::generate(oldValue.begin(), oldValue.end(), randFloat);
        std::fill(oldValue.begin(), oldValue.end(), 0);


        std::vector<float> valueOmp(matrixSize * matrixSize);
        std::copy(value.begin(), value.end(), valueOmp.begin());

        std::vector<float> oldValueOmp(matrixSize * matrixSize);
        std::copy(oldValue.begin(), oldValue.end(), oldValueOmp.begin());

        // SERIAL ----------------------------------------------------------------------------------------------

        int64_t serialTimeMean = 0;

        std::cout << BOLD BLUE "Matrix size: " << RESET << matrixSize << std::endl;
        for (int i = 0; i < 10; i++) {
            auto serialBegin = std::chrono::high_resolution_clock::now();
            diffuse(matrixSize, Axis::ZERO, value, oldValue, diff, dt);
            auto serialEnd = std::chrono::high_resolution_clock::now();
            auto serialTime = std::chrono::duration_cast<std::chrono::microseconds>(serialEnd - serialBegin).count();

            serialTimeMean += serialTime;


        }

        serialTimeMean /= 10;
        std::cout << BOLD YELLOW "Diffuse: " << serialTimeMean << RESET " micros "<<std::endl<<std::endl;



        // OMP ----------------------------------------------------------------------------------------------
        int max_threads = omp_get_max_threads();   
        omp_set_num_threads(max_threads);
        // Calculate how many cells as maximum per thread

        const int max_rows = (int)(ceil((matrixSize-2) / max_threads) + 2);
        max_cells = max_rows * (matrixSize-2);        

        int64_t ompTimeMean = 0;

        for (int i = 0; i < 10; i++) {
            auto ompBegin = std::chrono::high_resolution_clock::now();
            omp_diffuse(matrixSize, Axis::ZERO, valueOmp, oldValueOmp, diff, dt);
            auto ompEnd = std::chrono::high_resolution_clock::now();
            auto ompTime = std::chrono::duration_cast<std::chrono::microseconds>(ompEnd - ompBegin).count();

            ompTimeMean += ompTime;
        }
        ompTimeMean /= 10;
        
        std::cout << BOLD RED "OMP Diffuse: " << ompTimeMean << RESET " micros" << std::endl;

        speedup = (double) serialTimeMean / (double) ompTimeMean;
        efficiency = speedup / max_threads;
        std::cout << BOLD BLUE "Speedup: " << RESET << speedup << " ";
        std::cout << BOLD GREEN "Efficiency: " << RESET << efficiency << std::endl << std::endl;


        // CUDA ----------------------------------------------------------------------------------------------
        int64_t cudaTimeMean = 0;




        for (int i = 0; i < 10; i++) {
            auto cudaBegin = std::chrono::high_resolution_clock::now();
            cuda_diffuse(matrixSize, Axis::ZERO, valueOmp, oldValueOmp, diff, dt);
            cudaDeviceSynchronize();
            auto cudaEnd = std::chrono::high_resolution_clock::now();
            auto cudaTime = std::chrono::duration_cast<std::chrono::microseconds>(cudaEnd - cudaBegin).count();

            cudaTimeMean += cudaTime;
        }

        cudaTimeMean /= 10;
        
        std::cout << BOLD GREEN "CUDA Diffuse: " << cudaTimeMean << RESET " micros" << std::endl;

        speedup = (double) serialTimeMean / (double) cudaTimeMean;
        std::cout << BOLD BLUE "Speedup: " << RESET << speedup << " "<<std::endl<<std::endl;




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

void cuda_diffuse(int N, Axis mode, std::vector<float> &value, std::vector<float> &oldValue, float diffusion, float dt) {
    float diffusionRate = dt * diffusion * N * N;
    
    dim3 BlockSize(16, 16, 1);
    dim3 GridSize((N+15)/16, (N+15)/16, 1);

    float* d_value;
    float* d_oldValue;

    cudaMalloc(&d_value, N * N * sizeof(float));
    cudaMalloc(&d_oldValue, N * N * sizeof(float));

    cudaMemcpy(d_value, &value[0], N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldValue, &oldValue[0], N * N * sizeof(float), cudaMemcpyHostToDevice);

    kernel_lin_solve<<<GridSize, BlockSize>>>(N, mode, &value[0], &oldValue[0], diffusionRate);

    cudaMemcpy(&value[0], d_value, N * N * sizeof(float), cudaMemcpyDeviceToHost);
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
        #pragma omp parallel for default(shared) schedule(static, max_cells) collapse(2)
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

__global__ void kernel_lin_solve(int N, Axis mode, float* nextValue, float* value, float diffusionRate) {

    float c = 1 + 4 * diffusionRate;
    float cRecip = 1.0 / c;
    
    int col = threadIdx.x+blockIdx.x*blockDim.x;
	int row = threadIdx.y+blockIdx.y*blockDim.y;

	if(col == 0 || col >= N - 1 || row == 0 || row >= N - 1) return;


    for (int k = 0; k < ITERATIONS; k++) {
        nextValue[IX(row, col)] = (value[IX(row,col)]
                            + diffusionRate * (
                nextValue[IX(row + 1, col)]
                + nextValue[IX(row - 1, col)]
                + nextValue[IX(row, col + 1)]
                + nextValue[IX(row, col - 1)]
        )) * cRecip;

        __syncthreads();
        if (col == 1 && row == 1)
            kernel_set_bnd<<<1,1>>>(N, mode, nextValue);
        __syncthreads();
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

__device__ void kernel_set_bnd(int N, Axis mode, float *attr) {
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