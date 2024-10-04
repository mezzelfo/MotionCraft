import torch
import torch.nn.functional as F
import os


class BaseFlow:
    def __init__(self, N: int):
        self.N = N
        X, Y = torch.meshgrid(torch.arange(self.N), torch.arange(self.N), indexing="xy")
        self.XY = torch.stack([X, Y], dim=-1)
        self.neg_prompt = "poorly drawn,cartoon, 2d, disfigured, bad art, deformed, poorly drawn, extra limbs, close up, b&w, weird colors, blurry"
        self.this_path = os.path.dirname(os.path.abspath(__file__))

    def get_default_prompt(self) -> str:
        return (self.pos_prompt, self.neg_prompt)

    def get_default_image(self) -> torch.Tensor:
        raise NotImplementedError

    def get_default_framesteps(self) -> torch.Tensor:
        raise NotImplementedError

    def get_flow(t) -> torch.Tensor:
        raise NotImplementedError

    def get_spatial_eta(self, t):
        return torch.zeros(1, 1, self.N, self.N, dtype=torch.float32)

    def warp(
        self,
        t,
        previous_frame,
        original_frame,
        mode,
        padding_mode="reflection",
    ):
        flow = self.get_flow(t).to(previous_frame.device, previous_frame.dtype)
        warped_image = torch.nn.functional.grid_sample(
            previous_frame,
            flow.unsqueeze(0),
            align_corners=True,
            mode=mode,
            padding_mode=padding_mode,
        )
        return warped_image

    def get_ztau_orig(self, SDM, num_inference_steps):
        image_output_size = SDM.image_output_size
        base_img = self.get_default_image().resize((image_output_size, image_output_size))
        z0 = SDM.image_to_latent(base_img)
        latent = SDM.partial_inversion(
            z=z0,
            prompt=self.get_default_prompt()[0],
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
        )
        return latent
