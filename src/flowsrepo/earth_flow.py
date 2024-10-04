import torch
import PIL.Image as Image
import numpy as np
from .base_flow import BaseFlow


class EarthFlow(BaseFlow):
    def __init__(self, N, radius=0.83) -> None:
        super().__init__(N=N)
        self.radius = radius

        u = torch.linspace(-1, 1, N)
        v = torch.linspace(-1, 1, N)
        u, v = torch.meshgrid(u, v, indexing="xy")

        self.position_at_time_tau = torch.stack(
            [torch.sqrt(radius**2 - u**2 - v**2), u, v], dim=-1
        )
        self.space_mask = self.position_at_time_tau.isnan().any(dim=-1)

        self.pos_prompt = "a close up of a picture of the earth from space"

    def get_spatial_eta(self, t):
        flow = self.get_flow(t)
        mask = (
            (flow.abs() > 1)
            .any(dim=-1)
            .reshape(1, 1, self.N, self.N)
            .float()
        )
        mask2 = (
            (flow.abs() > 0.01)
            .any(dim=-1)
            .reshape(1, 1, self.N, self.N)
            .float()
        )
        maskfinal = mask*0.2 + mask2*0.8
        return maskfinal


    def get_default_image(self) -> torch.Tensor:
        image = Image.open(f"{self.this_path}/data/earth/earth.jpg")
        return image

    def get_default_framesteps(self) -> torch.Tensor:
        return torch.full((10,), 2*torch.pi / 20)

    def get_flow(self, tau):

        # Rotation matrix for the angle tau around the z-axis
        rotation_matrix = torch.tensor(
            [
                [torch.cos(tau), -torch.sin(tau), 0],
                [torch.sin(tau), torch.cos(tau), 0],
                [0, 0, 1],
            ],
            device=self.position_at_time_tau.device,
            dtype=self.position_at_time_tau.dtype,
        )

        # Reverse the rotation to get the position at time 0
        position_at_time_zero = torch.einsum(
            "ij,nmj->nmi", rotation_matrix.T, self.position_at_time_tau
        )

        # Patch the nans caused by the points not being on the sphere
        # Those points (at infinity) should not move
        position_at_time_zero = torch.where(
            self.space_mask[..., None], self.position_at_time_tau, position_at_time_zero
        )

        # Patch the points that are behind the sphere
        # by setting them to nan
        mask = position_at_time_zero[..., 0] < 0
        position_at_time_zero[mask] = 9999.9

        # Project the points to the plane
        projected_position_at_time_zero = position_at_time_zero[..., [1, 2]]
        return projected_position_at_time_zero
