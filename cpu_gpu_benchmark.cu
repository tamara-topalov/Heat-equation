#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cpu_solver.h"
#include "gpu_solver.h"


int main() {
    int sizes[] = {100, 200, 300};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int time_steps_list[] = {100, 200, 400};
    int n_steps = sizeof(time_steps_list) / sizeof(time_steps_list[0]);
    double gamma_list[] = {0.001, 0.005, 0.01};
    int n_gamma = sizeof(gamma_list) / sizeof(gamma_list[0]);
    double delta = 0.1;

    FILE* f = fopen("timing_results.csv", "w");
    if (!f) { perror("Failed to open output file"); return 1; }
    fprintf(f, "Grid,Steps,Gamma,CPU (s),GPU (ms),Speedup\n");

    printf("Grid     | Steps | Gamma  | CPU Time (s) | GPU Time (ms) | Speedup \n");
    printf("-------------------------------------------------------------------\n");

    for (int s = 0; s < n_sizes; ++s) {
        int width = sizes[s], height = sizes[s];
        size_t grid_size = (width + 2) * (height + 2);

        for (int g = 0; g < n_gamma; ++g) {
            double gamma = gamma_list[g];

            for (int ts = 0; ts < n_steps; ++ts) {
                int time_steps = time_steps_list[ts];

                double *grid_cpu = (double*) calloc(grid_size, sizeof(double));
                double *grid_gpu = (double*) calloc(grid_size, sizeof(double));

                reset_grid(grid_cpu, width, height);
                reset_grid(grid_gpu, width, height);

                clock_t start_cpu = clock();
                evolve_cpu(grid_cpu, width, height, time_steps, delta, gamma);
                clock_t end_cpu = clock();
                double cpu_time = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;

                float gpu_time = 0.0f;
                evolve_gpu(grid_gpu, width, height, time_steps, delta, gamma, &gpu_time);

                double speedup = (cpu_time * 1000.0) / gpu_time;

                printf("%3dx%-3d  | %5d | %.4f | %11.4f | %13.2f | %7.2fx \n",
                    width, height, time_steps, gamma, cpu_time, gpu_time, speedup);
                fprintf(f, "%dx%d,%d,%.4f,%.4f,%.2f\n",
                    width, height, time_steps, gamma, cpu_time, gpu_time);

                free(grid_cpu);
                free(grid_gpu); 
            }
        }
    }

    fclose(f);
    return 0;
}
