#include <stdio.h>
#include <stdlib.h>


void evolve(double* grid, int width, int height, int time_steps, double delta, double gamma) {
    double *new_grid = (double*) calloc((width + 2) * (height + 2 ), sizeof(double));
    int i, j, k;
    for (k = 0; k < time_steps; k++) {
        for (i = 1; i < width + 1; i++) {
            for (j = 1; j < height; j++) {
                new_grid[i * (height + 2) + j] = (1 - 4 * gamma / (delta * delta)) * grid[i * (height + 2) + j] +
                    (gamma / (delta * delta)) * (grid[(i - 1) * (height + 2) + j] +
                                                 grid[(i + 1) * (height + 2) + j] +
                                                 grid[i * (height + 2) + j - 1] +
                                                 grid[i * (height + 2) + j + 1]);
            }
        }
    
        for (i = 1; i < width + 1; i++) {
            for (j = 1; j < height + 1; j++) {
                grid[i * (height + 2) + j] = new_grid[i * (height + 2) + j];
            }
        }
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


int main() {
    int width = 10, height = 20, time_steps = 40;
    double delta = 0.1, gamma = 0.001;
    double *grid = (double*) calloc((width + 2) * (height + 2), sizeof(double));
    grid[5 * (height + 2) + 10] = 30; // Set initial condition
    evolve(grid, width, height, time_steps, delta, gamma);
    write_grid_to_file("matrix.dat", grid, width, height);
    system("gnuplot heatmap.gp");
    free(grid);

}