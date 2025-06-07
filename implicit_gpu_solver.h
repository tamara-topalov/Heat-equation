#ifndef IMPLICIT_GPU_SOLVER_H
#define IMPLICIT_GPU_SOLVER_H

void evolve_gpu_implicit(double* U_n, int width_steps, int height_steps, int time_steps, double delta, double gamma, float* elapsed_time_ms, dim3 threadsPerBlock, dim3 numBlocks, int max_iter);   


#endif