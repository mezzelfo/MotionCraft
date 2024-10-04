import torch
import PIL.Image as Image
import numpy as np
from .base_flow import BaseFlow


class SatelliteFlow(BaseFlow):
    def __init__(self, N) -> None:
        super().__init__(N=N)

        u = torch.linspace(-1, 1, N)
        v = torch.linspace(-1, 1, N)
        u, v = torch.meshgrid(u, v, indexing="xy")
        self.position_at_time_tau = torch.stack([u, v], dim=-1)

        self.pos_prompt = "a satellite image of a city"

    def get_spatial_eta(self, t):
        flow = self.get_flow(t)
        mask = (
            (flow.abs() > 1)
            .any(dim=-1)
            .reshape(1, 1, self.N, self.N)
            .float()
        )
        return mask

    def get_default_image(self) -> torch.Tensor:
        image = Image.open(f"{self.this_path}/data/satellite/satellite.png")
        return image

    def get_default_framesteps(self) -> torch.Tensor:
        img_fraction = 1/64
        return torch.full((64,), img_fraction*2)

    def get_flow(self, tau):
        position_at_time_zero = self.position_at_time_tau - torch.tensor([0, tau], device=self.position_at_time_tau.device)
        return position_at_time_zero
