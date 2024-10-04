import torch
import PIL.Image as Image
import numpy as np
from .base_flow import BaseFlow


class DragonsFlow(BaseFlow):
    def __init__(self, N) -> None:
        super().__init__(N=N)
        
        self.pos_prompt = "Two dragons fighting while breathing fires to each other. The flames are blazing and majestic light. Theatrical, character concept art by ruan jia, thomas kinkade, and  trending on Artstation."

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
            current_bin_pos = current_bin_pos + torch.nn.functional.grid_sample(
                current_bin_pos.to(end_point.device),
                end_point,
                align_corners=True,
                mode="bilinear",
            )
            current_bin_pos = (current_bin_pos>0)*1.0
            self.precomputed_flows.append(end_point.squeeze())
            self.precomputed_bin_masks.append(current_bin_pos)

    def get_spatial_eta(self, t):
        return self.precomputed_bin_masks[0]

    def get_default_image(self) -> torch.Tensor:
        image = Image.open(f"{self.this_path}/data/dragons/dragons.png")
        return image

    def get_default_framesteps(self) -> torch.Tensor:
        return torch.tensor(list(range(len(self.precomputed_flows))))

    def load_flow(self, subj=0):
        smoke_flow = np.load(f"{self.this_path}/data/dragons/phisimu_flow_dragon_ball1.npy")
        smoke_pos = np.load(f"{self.this_path}/data/dragons/phisimu_position_dragon_ball1.npy")

        smoke_flow2 = np.load(f"{self.this_path}/data/dragons/phisimu_flow_dragon_ball2.npy")
        smoke_pos2 = np.load(f"{self.this_path}/data/dragons/phisimu_position_dragon_ball2.npy")

        # Select the subject
        smoke_flow = smoke_flow[:, subj]
        smoke_pos = smoke_pos[:, subj]
        smoke_flow2 = smoke_flow2[:, subj]
        smoke_pos2 = smoke_pos2[:, subj]

        smoke_flow = torch.from_numpy(smoke_flow).float()
        smoke_pos = torch.from_numpy(smoke_pos).float()
        smoke_flow2 = torch.from_numpy(smoke_flow2).float()
        smoke_pos2 = torch.from_numpy(smoke_pos2).float()

        smoke_flow = smoke_flow.permute(0, 3, 1, 2)
        smoke_pos = smoke_pos.unsqueeze(1)
        smoke_flow2 = smoke_flow2.permute(0, 3, 1, 2)
        smoke_pos2 = smoke_pos2.unsqueeze(1)

        smoke_pos = torch.nn.functional.interpolate(
            smoke_pos, size=[self.N, self.N], mode="nearest"
        )
        smoke_pos2 = torch.nn.functional.interpolate(
            smoke_pos2, size=[self.N, self.N], mode="nearest"
        )

        smoke_flow = torch.nn.functional.interpolate(
            smoke_flow, size=[self.N, self.N], mode="bilinear"
        )
        smoke_flow2 = torch.nn.functional.interpolate(
            smoke_flow2, size=[self.N, self.N], mode="bilinear"
        )

        smoke_flow = self.manage_flow(smoke_flow)
        smoke_flow2 = self.manage_flow(smoke_flow2)
        flow_hist = (
            smoke_flow * smoke_pos.squeeze(1).unsqueeze(-1)
            + smoke_flow2 * smoke_pos2.squeeze(1).unsqueeze(-1)
        ).squeeze()

        flow_hist = flow_hist / 64.0

        position_hist = smoke_pos + smoke_pos2
        return flow_hist, position_hist  #(F,H,W,2), (F,1,H,W)

    def manage_flow(self, x):
        x = x.permute(0,3,2,1)
        x = torch.rot90(x, 2, [1, 2])
        x = torch.flip(x, [2])
        x = x * torch.tensor([1, -1], dtype=torch.float32, device=x.device)
        return x

    def get_flow(self, t) -> torch.Tensor:
        return self.precomputed_flows[t]
