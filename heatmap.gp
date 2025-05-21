# heatmap.gp
set terminal pngcairo size 600,400
set output 'heatmap.png'
set view map
set size ratio -1
set palette defined (0 "white", 1 "blue", 2 "green", 3 "yellow", 4 "red")
unset key
plot 'matrix.dat' matrix with image
