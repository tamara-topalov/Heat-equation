NVCC := /usr/local/cuda-12.6.3/bin/nvcc
CC := gcc
CFLAGS := -O2
NVCCFLAGS := -O2

TARGETS := benchmark visualize

all: $(TARGETS)

benchmark: cpu_gpu_benchmark.cu cpu_solver.c gpu_solver.cu
	$(NVCC) $(NVCCFLAGS) cpu_gpu_benchmark.cu cpu_solver.c gpu_solver.cu -o benchmark

visualize: heatmap_visualizer.cu cpu_solver.c gpu_solver.cu
	$(NVCC) $(NVCCFLAGS) heatmap_visualizer.cu cpu_solver.c gpu_solver.cu -o visualize

frames:
	bash render_frames.sh

gif: frames
	convert -delay 40 -loop 0 frames/frame_*.png heat_animation.gif

clean:
	rm -f benchmark visualize *.o *.dat frame_*.png heat_animation.gif
