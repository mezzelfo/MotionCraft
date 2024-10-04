import torch
import PIL.Image as Image
import numpy as np
from .base_flow import BaseFlow


class MeltingManFlow(BaseFlow):
    def __init__(self, N) -> None:
        super().__init__(N=N)

        flow_hist, position_hist = self.load_flow() #(F,H,W,2), (F,1,H,W)
        current_bin_pos = torch.where(position_hist[0] > 0, 1, 0)
        current_bin_pos = current_bin_pos.to(torch.float32).unsqueeze(0).to(self.XY.device)

        self.precomputed_bin_masks = [current_bin_pos]
        self.precomputed_flows = [(self.XY.float() / (self.N - 1))*2-1]

        for i in range(len(flow_hist)):
            end_point = self.XY - flow_hist[[i]].to(self.XY.device) * self.N
            end_point = end_point / (self.N - 1)
            end_point = 2 * end_point - 1
            end_point = end_point.to(torch.float32)
            current_bin_pos = torch.nn.functional.grid_sample(
                current_bin_pos.to(end_point.device),
                end_point,
                align_corners=True,
                mode="bilinear",
                padding_mode="reflection",
            )
            current_bin_pos = (current_bin_pos>0)*1.0
            self.precomputed_flows.append(end_point.squeeze())
            self.precomputed_bin_masks.append(current_bin_pos)

        self.pos_prompt = "transparent man made by water and smoke, in style of Yoji Shinkawa and Hyung-tae Kim, trending on ArtStation, dark fantasy, great composition, concept art, highly  human  made of water and foam, in the style of Pierre Koenig, red pigment, pastel paint, pink color scheme"


    def get_spatial_eta(self, t):
        eta = self.precomputed_bin_masks[t]
        eta[..., -5:, :] = 1
        return eta

    def get_default_image(self) -> torch.Tensor:
        image = Image.open(f"{self.this_path}/data/meltingman/meltingman.png")
        return image

    def get_default_framesteps(self) -> torch.Tensor:
        return torch.tensor(list(range(len(self.precomputed_flows))))

    def load_flow(self):
        flow = np.load(f"{self.this_path}/data/meltingman/phisimu_particleflow_meltingman_6.npy")
        flow = torch.from_numpy(flow).float().squeeze()
        flow = flow.permute(0, 3, 1, 2)
        flow = torch.nn.functional.interpolate(
            flow, size=[self.N, self.N], mode="bilinear"
        )

        flow = self.manage_flow(flow)

        position = np.load(f"{self.this_path}/data/meltingman/phisimu_particleposition_meltingman_6.npy")
        position = torch.from_numpy(position).float()
        position = position.unsqueeze(1)
        position = torch.nn.functional.interpolate(
            position, size=[self.N, self.N], mode="nearest"
        )

        flow = flow / 64.0

        return flow, position

    def manage_flow(self, x):
        x = x.permute(0,3,2,1)
        x = torch.rot90(x, 2, [1, 2])
        x = torch.flip(x, [2])
        x = x * torch.tensor([1, -1], dtype=torch.float32, device=x.device)
        x = x.nan_to_num(0.0)
        return x

    def get_flow(self, t) -> torch.Tensor:
        return self.precomputed_flows[t]
