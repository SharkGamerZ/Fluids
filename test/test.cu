#include "test.hpp"



int max_cells;

int64_t serialTimeMean = 0, ompTimeMean = 0, cudaTimeMean = 0;

int main() {
    std::ofstream file;

    const int iterations = 15;
    const int maxSize = 800;

    testDiffuse(maxSize, iterations);



    return 0;
}

void testDiffuse(int maxSize, int iterations) {    
    std::ofstream file;
    file.open("diffuse_results.csv");
    file << "Matrix size,Serial,OMP,Speedup,Efficiency\n";

    for (int matrixSize = 100; matrixSize <= maxSize; matrixSize+= 50)
    {
        // SETUP ----------------------------------------------------------------------------------------------
        std::cout<<"------------------------------------------------------------------"<<std::endl;
        double dt = 0.2;
        double diff = 0.0;
        double visc =  0.0000001f;

        double speedup, efficiency;

        std::srand(unsigned(std::time(nullptr)));

        std::vector<double> value(matrixSize * matrixSize);
        std::generate(value.begin(), value.end(), randdouble);

        std::vector<double> oldValue(matrixSize * matrixSize);
        std::generate(oldValue.begin(), oldValue.end(), randdouble);
        std::fill(oldValue.begin(), oldValue.end(), 0);


        std::vector<double> valueOmp(matrixSize * matrixSize);
        std::copy(value.begin(), value.end(), valueOmp.begin());

        std::vector<double> oldValueOmp(matrixSize * matrixSize);
        std::copy(oldValue.begin(), oldValue.end(), oldValueOmp.begin());

        // SERIAL ----------------------------------------------------------------------------------------------
        std::cout << BOLD BLUE "Matrix size: " << RESET << matrixSize << std::endl;
        for (int i = 0; i < iterations; i++) {
            auto serialBegin = std::chrono::high_resolution_clock::now();
            diffuse(matrixSize, Axis::ZERO, value, oldValue, diff, dt);
            auto serialEnd = std::chrono::high_resolution_clock::now();
            auto serialTime = std::chrono::duration_cast<std::chrono::milliseconds>(serialEnd - serialBegin).count();

            serialTimeMean += serialTime;


        }

        serialTimeMean /= iterations;
        std::cout << BOLD YELLOW "Diffuse: " << serialTimeMean << RESET " millis "<<std::endl<<std::endl;

        file << matrixSize << "," << serialTimeMean << ",";


        // OMP ----------------------------------------------------------------------------------------------
        int num_threads = omp_get_max_threads();
        omp_set_num_threads(4);

        // Calculate how many cells as maximum per thread
        const int max_rows = (int)(ceil((matrixSize-2) / num_threads) + 2);
        max_cells = max_rows * (matrixSize-2);        

        int64_t ompTimeMean = 0;

        for (int i = 0; i < iterations; i++) {
            auto ompBegin = std::chrono::high_resolution_clock::now();
            omp_diffuse(matrixSize, Axis::ZERO, valueOmp, oldValueOmp, diff, dt);
            auto ompEnd = std::chrono::high_resolution_clock::now();
            auto ompTime = std::chrono::duration_cast<std::chrono::milliseconds>(ompEnd - ompBegin).count();

            ompTimeMean += ompTime;
        }
        ompTimeMean /= iterations;
        
        std::cout << BOLD RED "OMP Diffuse: " << ompTimeMean << RESET " millis" << std::endl;

        speedup = (double) serialTimeMean / (double) ompTimeMean;
        efficiency = speedup / num_threads;
        std::cout << BOLD BLUE "Speedup: " << RESET << speedup << " ";
        std::cout << BOLD GREEN "Efficiency: " << RESET << efficiency << std::endl << std::endl;

        speedup = (double) serialTimeMean / (double) ompTimeMean;
        efficiency = speedup / num_threads;
        file << std::fixed << std::setprecision(2) << ompTimeMean << "," << std::setprecision(2) << speedup << "," << std::setprecision(2) << efficiency << "\n";
        
        printf("Speedup: %f\n", speedup);
        // CUDA ----------------------------------------------------------------------------------------------
        for (int i = 0; i < iterations; i++) {
            cuda_diffuse(matrixSize, Axis::ZERO, valueOmp, oldValueOmp, diff, dt);
        }

        cudaTimeMean /= iterations;
        
        std::cout << BOLD GREEN "CUDA Diffuse: " << cudaTimeMean << RESET " millis" << std::endl;

        speedup = (double) serialTimeMean / (double) cudaTimeMean;
        std::cout << BOLD BLUE "Speedup: " << RESET << speedup << " "<<std::endl<<std::endl;




        std::cout << std::endl << std::endl;
    }
}

void testAdvect(int maxSize, int iterations) {    
    std::ofstream file;
    file.open("diffuse_results.csv");
    file << "Matrix size,Serial,OMP,Speedup,Efficiency\n";

    for (int matrixSize = 100; matrixSize <= maxSize; matrixSize+= 50)
    {
        // SETUP ----------------------------------------------------------------------------------------------
        std::cout<<"------------------------------------------------------------------"<<std::endl;
        double dt = 0.2;
        double diff = 0.0;
        double visc =  0.0000001f;

        double speedup, efficiency;

        std::srand(unsigned(std::time(nullptr)));


        // ------- Serial --------
        // Value
        std::vector<double> value(matrixSize * matrixSize);
        std::generate(value.begin(), value.end(), randdouble);

        // OldValue
        std::vector<double> oldValue(matrixSize * matrixSize);
        std::generate(oldValue.begin(), oldValue.end(), randdouble);
        std::fill(oldValue.begin(), oldValue.end(), 0);
        

        // vX
        std::vector<double> vX(matrixSize * matrixSize);
        std::generate(vX.begin(), vX.end(), randdouble);

        // vX0
        std::vector<double> vX0(matrixSize * matrixSize);
        std::generate(vX0.begin(), vX0.end(), randdouble);
        std::fill(vX0.begin(), vX0.end(), 0);
        
        // vY
        std::vector<double> vY(matrixSize * matrixSize);
        std::generate(vY.begin(), vY.end(), randdouble);

        // vY0
        std::vector<double> vY0(matrixSize * matrixSize);
        std::generate(vY0.begin(), vY0.end(), randdouble);
        std::fill(vY0.begin(), vY0.end(), 0);
        

        // ------- OMP -------- 
        // Value
        std::vector<double> valueOmp(matrixSize * matrixSize);
        std::copy(value.begin(), value.end(), valueOmp.begin());

        // OldValue
        std::vector<double> oldValueOmp(matrixSize * matrixSize);
        std::copy(oldValue.begin(), oldValue.end(), oldValueOmp.begin());

        
        // vX
        std::vector<double> vXOmp(matrixSize * matrixSize);
        std::generate(vXOmp.begin(), vXOmp.end(), randdouble);

        // vX0
        std::vector<double> vX0Omp(matrixSize * matrixSize);
        std::generate(vX0Omp.begin(), vX0Omp.end(), randdouble);
        std::fill(vX0Omp.begin(), vX0Omp.end(), 0);
        
        // vY
        std::vector<double> vYOmp(matrixSize * matrixSize);
        std::generate(vYOmp.begin(), vYOmp.end(), randdouble);

        // vY0
        std::vector<double> vY0Omp(matrixSize * matrixSize);
        std::generate(vY0Omp.begin(), vY0Omp.end(), randdouble);
        std::fill(vY0Omp.begin(), vY0Omp.end(), 0);

        // SERIAL ----------------------------------------------------------------------------------------------
        std::cout << BOLD BLUE "Matrix size: " << RESET << matrixSize << std::endl;
        for (int i = 0; i < iterations; i++) {
            auto serialBegin = std::chrono::high_resolution_clock::now();

            advect(matrixSize, Axis::ZERO, vX, vX0, vX0, vY0, dt);

            auto serialEnd = std::chrono::high_resolution_clock::now();
            auto serialTime = std::chrono::duration_cast<std::chrono::milliseconds>(serialEnd - serialBegin).count();

            serialTimeMean += serialTime;


        }

        serialTimeMean /= iterations;
        std::cout << BOLD YELLOW "Advect: " << serialTimeMean << RESET " millis "<<std::endl<<std::endl;

        file << matrixSize << "," << serialTimeMean << ",";


        // OMP ----------------------------------------------------------------------------------------------
        int num_threads = omp_get_max_threads();
        omp_set_num_threads(4);

        // Calculate how many cells as maximum per thread
        const int max_rows = (int)(ceil((matrixSize-2) / num_threads) + 2);
        max_cells = max_rows * (matrixSize-2);        

        int64_t ompTimeMean = 0;

        for (int i = 0; i < iterations; i++) {
            auto ompBegin = std::chrono::high_resolution_clock::now();

            omp_advect(matrixSize, Axis::ZERO, vXOmp, vX0Omp, vX0Omp, vY0Omp, dt);

            auto ompEnd = std::chrono::high_resolution_clock::now();
            auto ompTime = std::chrono::duration_cast<std::chrono::milliseconds>(ompEnd - ompBegin).count();

            ompTimeMean += ompTime;
        }
        ompTimeMean /= iterations;
        
        std::cout << BOLD RED "OMP Advect: " << ompTimeMean << RESET " millis" << std::endl;

        speedup = (double) serialTimeMean / (double) ompTimeMean;
        efficiency = speedup / num_threads;
        std::cout << BOLD BLUE "Speedup: " << RESET << speedup << " ";
        std::cout << BOLD GREEN "Efficiency: " << RESET << efficiency << std::endl << std::endl;

        speedup = (double) serialTimeMean / (double) ompTimeMean;
        efficiency = speedup / num_threads;
        file << std::fixed << std::setprecision(2) << ompTimeMean << "," << std::setprecision(2) << speedup << "," << std::setprecision(2) << efficiency << "\n";
        
        printf("Speedup: %f\n", speedup);
        // // CUDA ----------------------------------------------------------------------------------------------
        // for (int i = 0; i < iterations; i++) {
        //     cuda_diffuse(matrixSize, Axis::ZERO, valueOmp, oldValueOmp, diff, dt);
        // }

        // cudaTimeMean /= iterations;
        
        // std::cout << BOLD GREEN "CUDA Advect: " << cudaTimeMean << RESET " millis" << std::endl;

        // speedup = (double) serialTimeMean / (double) cudaTimeMean;
        // std::cout << BOLD BLUE "Speedup: " << RESET << speedup << " "<<std::endl<<std::endl;




        // std::cout << std::endl << std::endl;
    }
}



double randdouble() {
    return ((double) rand() / (RAND_MAX));
}

bool double_equals(double a, double b, double epsilon)
{
    return std::abs(a - b) < epsilon;
}

// Diffuse ----------------------------------------------------------------------------------------------

void diffuse(int N, Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusion, double dt) {
    double diffusionRate = dt * diffusion * N * N;

    lin_solve(N, mode, value, oldValue, diffusionRate);
}

void omp_diffuse(int N, Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusion, double dt) {
    double diffusionRate = dt * diffusion * N * N;

    omp_lin_solve(N, mode, value, oldValue, diffusionRate);
}

void cuda_diffuse(int N, Axis mode, std::vector<double> &value, std::vector<double> &oldValue, double diffusion, double dt) {
    double diffusionRate = dt * diffusion * N * N;
    
    dim3 BlockSize(16, 16, 1);
    dim3 GridSize((N+15)/16, (N+15)/16, 1);

    cudaEvent_t cudaStart, cudaStop;	
    float milliseconds = 0;

    cudaEventCreate(&cudaStart);
    cudaEventCreate(&cudaStop);

    double* d_value;
    double* d_oldValue;

    cudaMalloc(&d_value, N * N * sizeof(double));
    cudaMalloc(&d_oldValue, N * N * sizeof(double));

    cudaMemcpy(d_value, &value[0], N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_oldValue, &oldValue[0], N * N * sizeof(double), cudaMemcpyHostToDevice);

    std::cout<<"Hello from host"<<std::endl;

    cudaEventRecord(cudaStart);    
    
    for (int i = 0; i < ITERATIONS; i++) 
        kernel_lin_solve<<<GridSize, BlockSize>>>(N, mode, &value[0], &oldValue[0], diffusionRate);

    cudaEventRecord(cudaStop);

    cudaMemcpy(&value[0], d_value, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(cudaStop);
    cudaEventElapsedTime(&milliseconds, cudaStart, cudaStop);

    printf("Time for the kernel: %f ms\n", milliseconds);
    cudaTimeMean += milliseconds;


    cudaFree(d_value);
    cudaFree(d_oldValue);

    cudaEventDestroy(cudaStart);
    cudaEventDestroy(cudaStop);
}


// Advect ----------------------------------------------------------------------------------------------
void advect(int N, Axis mode, std::vector<double> &value, std::vector<double> &oldValue, std::vector<double> &vX, std::vector<double> &vY, double dt) {
    double i0, i1, j0, j1;

    double dt0 = dt * (N - 2);

	double s0, s1, t0, t1;
	double tmp1, tmp2, x, y;

	double Ndouble = N - 2;

	for(int j = 1; j < N - 1; j++) {
		for(int i = 1; i < N - 1; i++) {
            double v1 = vX[index(i, j, N)];
            double v2 = vY[index(i, j, N)];
            tmp1 = dt0 * v1;
            tmp2 = dt0 * v2;
            x = (double) i - tmp1;
            y = (double) j - tmp2;

            if(x < 0.5f) x = 0.5f;
            if(x > Ndouble + 0.5f) x = Ndouble + 0.5f;
            i0 = floor(x);
            i1 = i0 + 1.0f;
            if(y < 0.5f) y = 0.5f;
            if(y > Ndouble + 0.5f) y = Ndouble + 0.5f;
            j0 = floor(y);
            j1 = j0 + 1.0f;

            s1 = x - i0;
            s0 = 1.0f - s1;
            t1 = y - j0;
            t0 = 1.0f - t1;

            int i0i = i0;
            int i1i = i1;
            int j0i = j0;
            int j1i = j1;

			value[index(i, j, N)] =
				s0 * (t0 * oldValue[index(i0i, j0i, N)] + t1 * oldValue[index(i0i, j1i, N)]) +
				s1 * (t0 * oldValue[index(i1i, j0i, N)] + t1 * oldValue[index(i1i, j1i, N)]);
            }
    }
	set_bnd(N, mode, value);

}


void omp_advect(int N, Axis mode, std::vector<double> &value, std::vector<double> &oldValue, std::vector<double> &vX, std::vector<double> &vY, double dt) {
    double i0, i1, j0, j1;

    double dt0 = dt * (N - 2);

	double s0, s1, t0, t1;
	double tmp1, tmp2, x, y;

	double Ndouble = N - 2;

    #pragma omp parallel for default(shared) collapse(2)
	for(int j = 1; j < N - 1; j++) {
		for(int i = 1; i < N - 1; i++) {
            double v1 = vX[index(i, j, N)];
            double v2 = vY[index(i, j, N)];
            tmp1 = dt0 * v1;
            tmp2 = dt0 * v2;
            x = (double) i - tmp1;
            y = (double) j - tmp2;

            if(x < 0.5f) x = 0.5f;
            if(x > Ndouble + 0.5f) x = Ndouble + 0.5f;
            i0 = floor(x);
            i1 = i0 + 1.0f;
            if(y < 0.5f) y = 0.5f;
            if(y > Ndouble + 0.5f) y = Ndouble + 0.5f;
            j0 = floor(y);
            j1 = j0 + 1.0f;

            s1 = x - i0;
            s0 = 1.0f - s1;
            t1 = y - j0;
            t0 = 1.0f - t1;

            int i0i = i0;
            int i1i = i1;
            int j0i = j0;
            int j1i = j1;

			value[index(i, j, N)] =
				s0 * (t0 * oldValue[index(i0i, j0i, N)] + t1 * oldValue[index(i0i, j1i, N)]) +
				s1 * (t0 * oldValue[index(i1i, j0i, N)] + t1 * oldValue[index(i1i, j1i, N)]);
            }
    }
	set_bnd(N, mode, value);

}


// Lin Solve ----------------------------------------------------------------------------------------------

void lin_solve(int N, Axis mode, std::vector<double> &nextValue, std::vector<double> &value, double diffusionRate) {
    double c = 1 + 4 * diffusionRate;
    double cRecip = 1.0 / c;
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

void omp_lin_solve(int N, Axis mode, std::vector<double> &nextValue, std::vector<double> &value, double diffusionRate) {
    double c = 1 + 4 * diffusionRate;
    double cRecip = 1.0 / c;
    for (int k = 0; k < ITERATIONS; k++)
    {
        #pragma omp parallel for default(shared) schedule(guided) collapse(2)
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

__global__ void kernel_lin_solve(int N, Axis mode, double* nextValue, double* value, double diffusionRate) {


    double c = 1 + 4 * diffusionRate;
    double cRecip = 1.0 / c;
    
    int col = threadIdx.x+blockIdx.x*blockDim.x;
	int row = threadIdx.y+blockIdx.y*blockDim.y;

	if(col == 0 || col >= N - 1 || row == 0 || row >= N - 1) return;

    printf("Hello from col: %d row: %d\n", col, row);

    nextValue[IX(row, col)] = (value[IX(row,col)]
                        + diffusionRate * (
            nextValue[IX(row + 1, col)]
            + nextValue[IX(row - 1, col)]
            + nextValue[IX(row, col + 1)]
            + nextValue[IX(row, col - 1)]
    )) * cRecip;

        // __syncthreads();
        // if (col == 1 && row == 1)
        //     kernel_set_bnd(N, mode, nextValue);
        // __syncthreads();

}

// Set Bnd ----------------------------------------------------------------------------------------------

void set_bnd(int N, Axis mode, std::vector<double> &attr) {
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

__device__ void kernel_set_bnd(int N, Axis mode, double *attr) {
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