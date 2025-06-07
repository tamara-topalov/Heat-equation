#ifndef CPU_SOLVER_H
#define CPU_SOLVER_H

#ifdef __cplusplus
extern "C" {
#endif

void evolve_cpu(double* grid, int width, int height, int time_steps, double delta, double gamma);
void write_grid_to_file(const char* filename, double* grid, int width, int height);
void reset_grid(double* grid, int width, int height);
void initialize_grid_sin(double* grid, int width, int height, double delta);
void compute_analytical_solution_sin(double* grid, int width, int height, double delta, double t);


#ifdef __cplusplus
}
#endif

#endif
