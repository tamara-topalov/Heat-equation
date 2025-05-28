#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "gpu_solver.h"
#include "cpu_solver.h"

int main() {
    int width = 300, height = 300;
    int time_steps = 200;
    int frame_interval = 25;
    double delta = 0.1, gamma = 0.001;
    size_t grid_size = (width + 2) * (height + 2);

    double* grid = (double*) calloc(grid_size, sizeof(double));
    reset_grid(grid, width, height);

    float gpu_time = 0.0f;
    evolve_gpu_with_frames(grid, width, height, time_steps, delta, gamma, &gpu_time, frame_interval);

    printf("Completed heatmap generation in %.2f ms.\n", gpu_time);
    free(grid);
    return 0;
}