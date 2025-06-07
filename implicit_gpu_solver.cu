#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "implicit_gpu_solver.h"


__global__
void jacobi_parallel_update(double* V_new, double* V_old, double lambda, double* b, int width_steps, int height_steps ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > width_steps || j > height_steps) return;

    V_new[i * (height_steps + 2) + j] = lambda / (1 + 4 * lambda) * (V_old[i * (height_steps + 2) + j - 1] + V_old[i * (height_steps + 2) + j + 1] + 
                                        V_old[(i - 1) * (height_steps + 2) + j] + V_old[(i + 1) * (height_steps + 2) + j])
                                            + b[i * (height_steps + 2) + j] / (1 + 4 * lambda);
}

__global__
void parallel_L2_dist(double* V1, double* V2, int N, int chunk_size, double* results) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int begin = index * chunk_size;
    int end = (begin + chunk_size < N) ? begin + chunk_size : N;
    double local = 0;

    while (begin < end) {
        local += (V1[begin] - V2[begin]) * (V1[begin] - V2[begin]);
        begin++;
    }
    results[index] = local;
}

__global__
void initialize_random(double* V, int N, int chunk_size, int height_steps, int width_steps, int k) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int begin = index * chunk_size;
    int end = (begin + chunk_size < N) ? begin + chunk_size : N;

    curandStatePhilox4_32_10_t state;
    curand_init(k /*seed*/, index, 0, &state);
    while (begin < end) {
        if (begin / (height_steps + 2)  == 0 || begin / (height_steps + 2) == width_steps + 1 || begin % (height_steps + 2) == 0 || begin % (height_steps + 2)== height_steps + 1) {
            V[begin] = 0;
        }
        else {
            V[begin] =  curand_uniform_double(&state);
        }
        begin++;
    }
}


void evolve_gpu_implicit(double* U_n, int width_steps, int height_steps, int time_steps, double delta, double gamma, float* elapsed_time_ms, dim3 threadsPerBlock, dim3 numBlocks, int max_iter = 100 ) {    
    int N = (width_steps+2) * (height_steps+2);
    double lambda = gamma / (delta * delta);

    //for parallel jacobi update 
    double *V_new, *V_old, *b;
    cudaMalloc(&V_new, N * sizeof(double));
    cudaMalloc(&V_old, N * sizeof(double));
    cudaMalloc(&b, N * sizeof(double));
    cudaMemcpy(b, U_n,  N * sizeof(double), cudaMemcpyHostToDevice);

    //for parallel L2 distance computation and for random initialization 
    int threads_per_block = 128, n_blocks = 48, total_threads = 6144;
    int chunk_size = (N + total_threads - 1) / total_threads;
    double *dist_l2, *dist_l2_gpu;
    dist_l2 = (double*) malloc(total_threads * sizeof(double));
    cudaMalloc(&dist_l2_gpu, total_threads * sizeof(double));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int k = 0; k < time_steps; k++) {
        initialize_random<<<n_blocks, threads_per_block>>>(V_old, N, chunk_size, height_steps, width_steps, k);
        int i = 0;
        double epsilon = 1;

        while (i < max_iter && epsilon > 0.000001 ) {
            jacobi_parallel_update<<<numBlocks, threadsPerBlock>>>(V_new, V_old, lambda, b, width_steps, height_steps);
            cudaDeviceSynchronize();
            jacobi_parallel_update<<<numBlocks, threadsPerBlock>>>(V_old, V_new, lambda, b, width_steps, height_steps);
            cudaDeviceSynchronize();

            parallel_L2_dist<<<n_blocks, threads_per_block>>>(V_old, V_new, N, chunk_size, dist_l2_gpu);
            cudaMemcpy(dist_l2, dist_l2_gpu, total_threads * sizeof(double), cudaMemcpyDeviceToHost);
            epsilon = 0;
            for (int j = 0; j < total_threads; j++) {
                epsilon += dist_l2[j];
            }
            i++;
        }
        //our approximate solution is now stored in V_old

        // if ( i >= max_iter) {
        //     std::cout << "No convergence, epsilon = "  << epsilon << std::endl; 
        // }
        // else {
        //     std::cout << "Coverged in " << 2 * i << " iterations, epsilon = "  << epsilon << std::endl; 
        // }

        cudaMemcpy(b, V_old, N * sizeof(double), cudaMemcpyDeviceToDevice);
    }   

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(elapsed_time_ms, start, stop);
    
    cudaMemcpy(U_n, V_old, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(V_new);
    cudaFree(V_old);
    cudaFree(b);
    free(dist_l2);
    cudaFree(dist_l2_gpu);
}


// int main() {
//     int width_steps = 100, height_steps = 100, time_steps = 1000;
//     double delta = 0.1, gamma = 0.001;
//     double *grid = (double*) calloc((width_steps + 2) * (height_steps + 2), sizeof(double));
//     //double *grid_new = (double*) calloc((width_steps + 2) * (height_steps + 2), sizeof(double));


//     initialize_grid_sin(grid, width_steps, height_steps, delta);
//     evolve_parallel(grid, gamma/(delta * delta),width_steps, height_steps, time_steps);

//     write_grid_to_file("matrix_numerical.dat", grid, width_steps, height_steps);
//     write_analytical_to_file("matrix_analytical.dat", width_steps, height_steps, delta, gamma, time_steps * delta);

//     system("gnuplot heatmap_numerical.gp");
//     system("gnuplot heatmap_analytical.gp");
    
//     free(grid);
//     //free(grid_new);

// }