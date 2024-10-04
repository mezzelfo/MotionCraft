from flowsrepo import example_registry
import torch
import os
import time
import argparse
import numpy as np
import PIL.Image as Image

args = argparse.ArgumentParser()
args.add_argument("--tag", type=str, default="PixelSpace")
args.add_argument(
    "--example",
    type=str,
    required=True,
)
args.add_argument(
    "--interpolationmode",
    type=str,
    choices=["bilinear", "nearest", "bicubic"],
    default="nearest",
)
args.add_argument("--device", type=str, default="cuda:0")
args = args.parse_args()

##### Parameters
torch.manual_seed(11)
device = args.device
N = 512
image_warper = example_registry[args.example](N=N)
image = image_warper.get_default_image()
image = image.resize((N, N))
image = torch.from_numpy(np.array(image) / 255).float().permute(2, 0, 1)
image = image.unsqueeze(0)

folder_path = f"output/{args.tag}/{time.time()}_{args.example}"
os.makedirs(folder_path, exist_ok=True)

def save_output(output, name):
    frame = output.mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()[0]
    Image.fromarray(frame).save(f"{folder_path}/{name}.png")

image_orig = image.clone()
warped = image.clone()

framesteps = image_warper.get_default_framesteps()
for f, (framestep) in enumerate(framesteps):
    warped = image_warper.warp(
        t=framestep,
        previous_frame=warped,
        original_frame=image_orig,
        mode=args.interpolationmode,
    )

    save_output(warped, f"frame_{f:03}")

# ffmpeg -framerate 5 -i frame_%03d.png -c:v libx264 -profile:v baseline -level 3.0 -pix_fmt yuv420p out.mp4
