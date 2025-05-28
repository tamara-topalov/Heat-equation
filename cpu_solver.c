#include <stdio.h>
#include <stdlib.h>
#include "cpu_solver.h"

void evolve_cpu(double* grid, int width, int height, int time_steps, double delta, double gamma) {
    double* new_grid = (double*) calloc((width + 2) * (height + 2), sizeof(double));

    for (int t = 0; t < time_steps; ++t) {
        for (int i = 1; i < width + 1; ++i) {
            for (int j = 1; j < height + 1; ++j) {
                new_grid[i * (height + 2) + j] = (1 - 4 * gamma / (delta * delta)) * grid[i * (height + 2) + j] +
                    (gamma / (delta * delta)) * (
                        grid[(i - 1) * (height + 2) + j] +
                        grid[(i + 1) * (height + 2) + j] +
                        grid[i * (height + 2) + j - 1] +
                        grid[i * (height + 2) + j + 1]);
            }
        }
        for (int i = 1; i < width + 1; ++i)
            for (int j = 1; j < height + 1; ++j)
                grid[i * (height + 2) + j] = new_grid[i * (height + 2) + j];
    }
    free(new_grid);
}

void write_grid_to_file(const char* filename, double* grid, int width, int height) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    for (int i = 1; i <= width; i++) {
        for (int j = 1; j <= height; j++) {
            fprintf(file, "%f ", grid[i * (height + 2) + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void reset_grid(double* grid, int width, int height) {
    for (int i = 0; i < (width + 2) * (height + 2); ++i)
        grid[i] = 0.0;
    int mid_i = width / 2, mid_j = height / 2;
    grid[mid_i * (height + 2) + mid_j] = 100.0;
}