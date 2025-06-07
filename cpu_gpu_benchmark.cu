// cpu_gpu_benchmark.cu
// Benchmarks CPU vs GPU simulation time across grid sizes, time steps, gamma values, and different CUDA block sizes. Also prints system information and checks CPU-GPU correctness.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "cpu_solver.h"
#include "gpu_solver.h"
#include "implicit_gpu_solver.h"

double compute_rmse(double* a, double* b, int size) {
    double mse = 0.0;
    for (int i = 0; i < size; ++i) {
        double diff = a[i] - b[i];
        mse += (diff * diff);
    }
    return mse / size;
}

int main() {
    // Print system information
    int n_cores = sysconf(_SC_NPROCESSORS_ONLN);
    printf("CPU cores: %d\n", n_cores);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s, %d SMs, %ld MB global mem\n", prop.name, prop.multiProcessorCount, prop.totalGlobalMem / (1024 * 1024));

    int sizes[] = {100, 200, 300, 1000};
    int time_steps_list[] = {100, 200, 400};
    
    // making gamma and delta pairs that satisfy the gamma / (delta^2) < 0.5 condtion
    struct ParamPair {
        double gamma;
        double delta;
    };
    ParamPair param_pairs[] = {
        {0.0010, 0.1},
        {0.0050, 0.2},
        {0.0100, 0.2}, 
        {0.0005, 0.05},
    };
    int num_pairs = 4;

    dim3 block_sizes[] = { dim3(8,8), dim3(16,16), dim3(32,32) };

    FILE* f = fopen("timing_results.csv", "w");
    fprintf(f, "Grid,Steps,Gamma,Delta,BlockX,BlockY,CPU (s),GPU (ms),Speedup,RMSE\n");

    for (int s = 0; s < 4; ++s) {
        int width = sizes[s], height = sizes[s];
        size_t grid_size = (width + 2) * (height + 2);

        for (int p = 0; p < num_pairs; ++p) {
            double gamma = param_pairs[p].gamma;
            double delta = param_pairs[p].delta;

            if (gamma / (delta * delta) >= 0.5) {
                printf("Skipping unstable pair: gamma=%.4f, delta=%.4f (gamma/delta²=%.4f)\n",
                    gamma, delta, gamma / (delta * delta));
                continue;
            }

            for (int ts = 0; ts < 3; ++ts) {
                int time_steps = time_steps_list[ts];

                double *grid_cpu = (double*) calloc(grid_size, sizeof(double));
                double *grid_gpu = (double*) calloc(grid_size, sizeof(double));
                double *grid_gpu_implicit = (double*) calloc(grid_size, sizeof(double));
                double *grid_analytical = (double*) calloc(grid_size, sizeof(double));

                compute_analytical_solution_sin(grid_analytical, width, height, delta, time_steps * delta);

                initialize_grid_sin(grid_cpu, width, height, delta);
                initialize_grid_sin(grid_gpu, width, height, delta);
                initialize_grid_sin(grid_gpu_implicit, width, height, delta);

                clock_t start_cpu = clock();
                evolve_cpu(grid_cpu, width, height, time_steps, delta, gamma);
                clock_t end_cpu = clock();
                double cpu_time = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;

                for (int b = 0; b < 3; ++b) {
                    dim3 threadsPerBlock = block_sizes[b];
                    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

                    float gpu_time = 0.0f;
                    evolve_gpu(grid_gpu, width, height, time_steps, delta, gamma, &gpu_time, threadsPerBlock, numBlocks);

                    float gpu_time_implicit = 0.0f;
                    evolve_gpu_implicit(grid_gpu_implicit, width, height, time_steps, delta, gamma, &gpu_time_implicit, threadsPerBlock, numBlocks, 100);

                    double rmse_cpu = compute_rmse(grid_cpu, grid_analytical, grid_size);
                    double rmse_gpu = compute_rmse(grid_analytical, grid_gpu, grid_size);
                    double rmse_gpu_implicit = compute_rmse(grid_analytical, grid_gpu_implicit, grid_size);

                    double speedup = (cpu_time * 1000.0) / gpu_time;

                    printf("%3dx%-3d | %5d | gamma=%.4f | delta=%.4f | Block %2dx%2d | CPU: %.4fs | GPU: %.2fms | Speedup: %.2fx | (Implicit) GPU: %.2fms | CPU MSE: %.6f | GPU MSE: %.6f | Implicit GPU MSE: %.6f\n",
                        width, height, time_steps, gamma, delta, threadsPerBlock.x, threadsPerBlock.y,
                        cpu_time, gpu_time, speedup, gpu_time_implicit, rmse_cpu, rmse_gpu, rmse_gpu_implicit);

                    fprintf(f, "%dx%d,%d,%.4f,%.4f,%d,%d,%.4f,%.2f,%.2f,%.6f\n",
                        width, height, time_steps, gamma, delta, threadsPerBlock.x, threadsPerBlock.y,
                        cpu_time, gpu_time, speedup, rmse_gpu);
                }

                free(grid_cpu);
                free(grid_gpu);
                free(grid_gpu_implicit);
            }
        }
    }

    fclose(f);
    return 0;
}

