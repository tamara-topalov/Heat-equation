#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cpu_solver.h"

void update_grid(double* new_grid, double* grid, int width, int height, double delta, double gamma) {
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
}

void evolve_cpu(double* grid, int width, int height, int time_steps, double delta, double gamma) {
    double* new_grid = (double*) calloc((width + 2) * (height + 2), sizeof(double));

    for (int t = 0; t < time_steps / 2; ++t) {
        update_grid(new_grid, grid, width, height, delta, gamma);
        update_grid(grid, new_grid, width, height, delta, gamma);
    }
    if (time_steps % 2 == 1) {
        update_grid(new_grid, grid, width, height, delta, gamma);
        for (int i = 0; i < (width + 2) * (height + 2); ++i)
            grid[i] = new_grid[i];
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

void initialize_grid_sin(double* grid, int width, int height, double delta) {
    for (int i = 1; i <= width; i++) {
        for (int j = 1; j <= height; j++) {
            double x = i * delta;
            double y = j * delta;
            grid[i * (height + 2) + j] = sin(M_PI * x / (width * delta)) * sin(M_PI * y / (height * delta));
        }
    }

}

// Compute the analytical solution for the heat equation with a sinusoidal initial condition
void compute_analytical_solution_sin(double* grid, int width, int height, double delta, double t) {
    double Lx = width * delta;
    double Ly = height * delta;

    for (int i = 1; i <= width; i++) {
        for (int j = 1; j <= height; j++) {
            double x = i * delta;
            double y = j * delta;
            double u = sin(M_PI * x / Lx) * sin(M_PI * y / Ly) *
                       exp(-M_PI * M_PI * t * (1.0 / (Lx * Lx) + 1.0 / (Ly * Ly)));

            grid[i * (height + 2) + j] = u;
        }
    }
}
