#!/bin/bash

mkdir -p frames

for f in frame_*.dat; do
  out="frames/${f%.dat}.png"
  gnuplot -e "input='$f'; output='$out'" plot_frames.gp
done

echo "All frames rendered to ./frames"