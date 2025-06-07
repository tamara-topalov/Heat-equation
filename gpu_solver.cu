#include <stdio.h>
#include <cuda_runtime.h>
#include "cpu_solver.h"
#include "gpu_solver.h"

__global__ void evolve_kernel(double* grid, double* new_grid, int width, int height, double delta, double gamma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < width + 1 && j < height + 1) {
        int idx = i * (height + 2) + j;
        new_grid[idx] = (1 - 4 * gamma / (delta * delta)) * grid[idx] +
                        (gamma / (delta * delta)) * (
                            grid[(i - 1) * (height + 2) + j] +
                            grid[(i + 1) * (height + 2) + j] +
                            grid[i * (height + 2) + j - 1] +
                            grid[i * (height + 2) + j + 1]);
    }
}

void evolve_gpu(double* h_grid, int width, int height, int time_steps, double delta, double gamma, float* elapsed_time_ms, dim3 threadsPerBlock, dim3 numBlocks) {
    int size = (width + 2) * (height + 2) * sizeof(double);
    double *d_grid, *d_new_grid;
    cudaMalloc(&d_grid, size);
    cudaMalloc(&d_new_grid, size);
    cudaMemcpy(d_grid, h_grid, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int t = 0; t < time_steps / 2; ++t) {
        evolve_kernel<<<numBlocks, threadsPerBlock>>>(d_grid, d_new_grid, width, height, delta, gamma);
        cudaDeviceSynchronize();
        evolve_kernel<<<numBlocks, threadsPerBlock>>>(d_new_grid, d_grid, width, height, delta, gamma);
        cudaDeviceSynchronize();
    }
    if (time_steps % 2 == 1) {
        evolve_kernel<<<numBlocks, threadsPerBlock>>>(d_grid, d_new_grid, width, height, delta, gamma);
        cudaDeviceSynchronize();
        cudaMemcpy(d_grid, d_new_grid, size, cudaMemcpyDeviceToDevice);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(elapsed_time_ms, start, stop);

    cudaMemcpy(h_grid, d_grid, size, cudaMemcpyDeviceToHost);
    cudaFree(d_grid);
    cudaFree(d_new_grid);
}


void evolve_gpu_with_frames(double* h_grid, int width, int height, int time_steps, double delta, double gamma, float* elapsed_time_ms, int frame_interval) {
    int size = (width + 2) * (height + 2) * sizeof(double);
    double *d_grid, *d_new_grid;
    cudaMalloc(&d_grid, size);
    cudaMalloc(&d_new_grid, size);
    cudaMemcpy(d_grid, h_grid, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int t = 0; t < time_steps/2; ++t) {
        evolve_kernel<<<numBlocks, threadsPerBlock>>>(d_grid, d_new_grid, width, height, delta, gamma);
        cudaDeviceSynchronize();
        evolve_kernel<<<numBlocks, threadsPerBlock>>>(d_new_grid, d_grid, width, height, delta, gamma);
        cudaDeviceSynchronize();


        if (t % frame_interval == 0 || t == time_steps / 2 - 1) {
            cudaMemcpy(h_grid, d_grid, size, cudaMemcpyDeviceToHost);
            char filename[64];
            sprintf(filename, "frame_%03d.dat", 2 * t);
            write_grid_to_file(filename, h_grid, width, height);
        }
    }

    if (time_steps % 2 == 1) {
        evolve_kernel<<<numBlocks, threadsPerBlock>>>(d_grid, d_new_grid, width, height, delta, gamma);
        cudaDeviceSynchronize();
        cudaMemcpy(d_grid, d_new_grid, size, cudaMemcpyDeviceToDevice);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(elapsed_time_ms, start, stop);

    cudaMemcpy(h_grid, d_grid, size, cudaMemcpyDeviceToHost);
    cudaFree(d_grid);
    cudaFree(d_new_grid);
}