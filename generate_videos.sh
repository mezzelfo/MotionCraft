#!/bin/bash

cd src

export CUDA_VISIBLE_DEVICES=1

# Pixel-Space Experiments
for example in "satellite" "dragons" "earth" "meltingman" "glass"
do
   python clip_maker_pixel.py --example $example --tag "PixelSpace_nearest" --interpolationmode "nearest"
done

# LatentSpace Experiments
for example in "satellite" "dragons" "earth" "meltingman" "glass"
do
   python clip_maker_latent.py --example $example --tag "MotionCraft_nearest" --interpolationmode "nearest"
   # python clip_maker_latent.py --example $example --tag "MotionCraft_bilinear" --interpolationmode "bilinear"
   # python clip_maker_latent.py --example $example --tag "MotionCraft_bicubic" --interpolationmode "bicubic"
done

# ffmpeg -framerate 5 -i frame_%03d_2.png -c:v libx264 -profile:v baseline -level 3.0 -pix_fmt yuv420p out.mp4