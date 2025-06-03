#ifndef GPU_SOLVER_H
#define GPU_SOLVER_H

void evolve_gpu(double* h_grid, int width, int height, int time_steps, double delta, double gamma, float* elapsed_time_ms, dim3 threadsPerBlock, dim3 numBlocks);
void evolve_gpu_with_frames(double* h_grid, int width, int height, int time_steps, double delta, double gamma, float* elapsed_time_ms, int frame_interval);

#endif