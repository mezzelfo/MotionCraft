from typing import List, Optional, Union
import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler


class MyDDIMScheduler(DDIMScheduler):
    tau_slice_wrt_num_train_timesteps = slice(None)

    def set_timesteps(
        self, num_inference_steps: int, device: Union[str, torch.device] = None
    ):
        super().set_timesteps(num_inference_steps, device)
        num_train_timesteps = len(self.betas)
        normalize_timestep = lambda t: int(round(t * num_inference_steps / num_train_timesteps))
        normalized_slice = slice(
            normalize_timestep(self.tau_slice_wrt_num_train_timesteps.start) if self.tau_slice_wrt_num_train_timesteps.start is not None else None,
            normalize_timestep(self.tau_slice_wrt_num_train_timesteps.stop) if self.tau_slice_wrt_num_train_timesteps.stop is not None else None,
            normalize_timestep(self.tau_slice_wrt_num_train_timesteps.step) if self.tau_slice_wrt_num_train_timesteps.step is not None else None,
        )
        self.timesteps = self.timesteps[normalized_slice]

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ):
        if type(eta) != float:
            # We are calling the function using a spatial eta
            orig_get_variance = self._get_variance
            self._get_variance = lambda timestep, prev_timestep: orig_get_variance(timestep, prev_timestep) * (eta**2)
            out = super().step(
                model_output,
                timestep,
                sample,
                eta=1.0,
                use_clipped_model_output=use_clipped_model_output,
                generator=generator,
                variance_noise=variance_noise,
                return_dict=return_dict,
            )
            self._get_variance = orig_get_variance
            return out
        else:
            return super().step(
                model_output,
                timestep,
                sample,
                eta=eta,
                use_clipped_model_output=use_clipped_model_output,
                generator=generator,
                variance_noise=variance_noise,
                return_dict=return_dict,
            )


class MyDDIMInverseScheduler(DDIMInverseScheduler):
    tau_slice_wrt_num_train_timesteps = slice(None)

    def set_timesteps(
        self, num_inference_steps: int, device: Union[str, torch.device] = None
    ):
        super().set_timesteps(num_inference_steps, device)
        num_train_timesteps = len(self.betas)
        normalize_timestep = lambda t: int(round(t * num_inference_steps / num_train_timesteps))
        normalized_slice = slice(
            normalize_timestep(self.tau_slice_wrt_num_train_timesteps.start) if self.tau_slice_wrt_num_train_timesteps.start is not None else None,
            normalize_timestep(self.tau_slice_wrt_num_train_timesteps.stop) if self.tau_slice_wrt_num_train_timesteps.stop is not None else None,
            normalize_timestep(self.tau_slice_wrt_num_train_timesteps.step) if self.tau_slice_wrt_num_train_timesteps.step is not None else None,
        )
        self.timesteps = self.timesteps[normalized_slice]


def get_all_schedulers(original_config, tau, num_inference_steps=1000, device='cpu'):
    

    scheduler_full_generation = MyDDIMScheduler.from_config(original_config)
    scheduler_partial_generation = MyDDIMScheduler.from_config(original_config)
    scheduler_full_inversion = MyDDIMInverseScheduler.from_config(original_config)
    scheduler_partial_inversion = MyDDIMInverseScheduler.from_config(original_config)
    scheduler_partial_generation_remaining = MyDDIMScheduler.from_config(original_config)

    scheduler_full_generation.tau_slice_wrt_num_train_timesteps = slice(None)
    scheduler_partial_generation.tau_slice_wrt_num_train_timesteps = slice(-tau, None)
    scheduler_full_inversion.tau_slice_wrt_num_train_timesteps = slice(None)
    scheduler_partial_inversion.tau_slice_wrt_num_train_timesteps = slice(0, tau)
    scheduler_partial_generation_remaining.tau_slice_wrt_num_train_timesteps = slice(None, -tau)

    scheduler_full_generation.set_timesteps(num_inference_steps, device)
    scheduler_partial_generation.set_timesteps(num_inference_steps, device)
    scheduler_full_inversion.set_timesteps(num_inference_steps, device)
    scheduler_partial_inversion.set_timesteps(num_inference_steps, device)
    scheduler_partial_generation_remaining.set_timesteps(num_inference_steps, device)

    return {
        'scheduler_full_generation' : scheduler_full_generation,
        'scheduler_partial_generation' : scheduler_partial_generation,
        'scheduler_full_inversion' : scheduler_full_inversion,
        'scheduler_partial_inversion' : scheduler_partial_inversion,
        'scheduler_partial_generation_remaining' : scheduler_partial_generation_remaining
    }
