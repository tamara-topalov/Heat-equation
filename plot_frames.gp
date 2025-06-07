# plot_frames.gp
set terminal pngcairo size 800,800
set output output
set title sprintf("Heat Diffusion at %s", input)
unset key
set palette rgbformulae 22,13,-31
set pm3d map
set cbrange [0:1]
splot input matrix with image
