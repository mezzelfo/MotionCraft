from flowsrepo import example_registry
from tqdm import tqdm
import torch
import os
import time
import argparse
from DiffusersUtils import (
    StableDiffusionManager,
    get_attention_processor,
    get_attention_processor_from_pattern,
)

args = argparse.ArgumentParser()
args.add_argument("--tag", type=str, default="MotionCraft")
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
args.add_argument("--tau", type=int, default=400)
args.add_argument("--num_inference_steps", type=int, default=200)
args.add_argument("--guidance_scale", type=float, default=7.5)
args.add_argument(
    "--crossframeattention_pattern", type=str, default="[[0,0],[1,1],[0,1]]"
)
args.add_argument("--invert", action=argparse.BooleanOptionalAction, default=True)
args.add_argument("--spatialeta", action=argparse.BooleanOptionalAction, default=True)
args.add_argument("--SDXL", action=argparse.BooleanOptionalAction, default=False)

args = args.parse_args()

##### Parameters
torch.manual_seed(11)
device = args.device
N = 128 if args.SDXL else 64
image_warper = example_registry[args.example](N=N)
prompt, negative_prompt = image_warper.get_default_prompt()

# prompt = ""
# clean |-------------------|> noise
# 0     |----tau------------|> 1000
# 0     |---*-----*------*--|> * num inference steps
num_inference_steps = args.num_inference_steps
guidance_scale = args.guidance_scale
folder_path = f"output/{args.tag}/{time.time()}_{args.example}"
os.makedirs(folder_path, exist_ok=True)
SDM = StableDiffusionManager(device, args.tau, SDXL=args.SDXL)

single_attention_processor = get_attention_processor(
    video_length=1, crossframe_attn="disabled", should_record_history=False
)

cross_attention_processor = get_attention_processor_from_pattern(
    args.crossframeattention_pattern
)

SDM.pipeline.unet.set_attn_processor(single_attention_processor)
z_tau_orig = image_warper.get_ztau_orig(SDM, num_inference_steps)
z_tau = z_tau_orig.clone()

framesteps = image_warper.get_default_framesteps()
for f, (framestep) in enumerate(tqdm(framesteps)):
    warped_latent = image_warper.warp(
        t=framestep,
        previous_frame=z_tau,
        original_frame=z_tau_orig,
        mode=args.interpolationmode,
    )
    if args.spatialeta:
        spatial_eta = image_warper.get_spatial_eta(t=framestep)
        spatial_eta = spatial_eta.repeat(cross_attention_processor.video_length, 1, 1, 1)
        spatial_eta[:-1] = 0.0
        spatial_eta = spatial_eta.to(warped_latent.device, warped_latent.dtype)
        print(spatial_eta.abs().sum())
    else:
        spatial_eta = 0.0

    if cross_attention_processor.video_length == 2:
        z_pair = torch.cat([z_tau_orig, warped_latent], dim=0)
    else:
        z_pair = torch.cat([z_tau_orig, z_tau, warped_latent], dim=0)

    # Generate with cross attention
    SDM.pipeline.unet.set_attn_processor(cross_attention_processor)
    generated_latents = SDM.partial_generation(
        z=z_pair,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        eta=spatial_eta,
        guidance_scale=guidance_scale,
        negative_prompt=[prompt] * (len(z_pair) - 1) + [negative_prompt],
    )

    generated_frames = SDM.latent_to_image(generated_latents)
    for i, img in enumerate(generated_frames):
        img.save(f"{folder_path}/frame_{f:03}_{i}.png")

    # Update the z0 for the next iteration
    if not args.invert:
        print("Warning: not inverting")
        z_tau = warped_latent
    else:
        SDM.pipeline.unet.set_attn_processor(single_attention_processor)
        z_tau = SDM.partial_inversion(
            z=generated_latents[[-1]],
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
        )
