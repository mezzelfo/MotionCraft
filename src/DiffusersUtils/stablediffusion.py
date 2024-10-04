import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.image_processor import VaeImageProcessor
from DiffusersUtils import get_all_schedulers
from copy import deepcopy
from PIL import Image
import inspect


class StableDiffusionManager:
    def __init__(self, device, tau, SDXL=False) -> None:
        # Save parameters
        self.device = device
        self.tau = tau

        if not SDXL:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                use_safetensors=False,
                safety_checker=None,
                local_files_only = False,
            ).to(self.device)
            self.image_output_size = 512
        else:
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=False,
                safety_checker=None,
                local_files_only = True,
            ).to(self.device)
            self.image_output_size = 1024
            print('UPCASTING VAE')
            self.pipeline.upcast_vae()

        self.pipeline.set_progress_bar_config(disable=True)

        self.pipeline_call_params = inspect.signature(self.pipeline.__call__).parameters
        self.pipeline_call_params = set(list(self.pipeline_call_params))

        self.original_scheduler = deepcopy(self.pipeline.scheduler)

        self.schedulers = get_all_schedulers(
            self.original_scheduler.config,
            tau=tau,
            num_inference_steps=1000,
            device=device,
        )

        self.alphabar = self.schedulers['scheduler_full_generation'].alphas_cumprod

        # Initialize generator
        self.generator = torch.Generator(device=device).manual_seed(0)

    @torch.no_grad()
    def image_to_latent(self, image: Image):
        if type(self.pipeline) == StableDiffusionXLPipeline:
            print('UPCASTING VAE')
            self.pipeline.upcast_vae()
        vae = self.pipeline.vae
        image_proc: VaeImageProcessor = self.pipeline.image_processor
        latent = image_proc.preprocess(image).to(vae.device, vae.dtype)
        # print("Warning: try not to use me too often as the autoencoder is not perfect.")
        # image = torch.from_numpy(np.array(image) / 255).permute(2, 0, 1).float()
        # image = image.unsqueeze(0) * 2 - 1
        # latent = image.to(self.device)
        my_gen = torch.Generator(device=self.device).manual_seed(184750)
        latent = vae.encode(latent).latent_dist.sample(generator=my_gen)
        latent = latent * vae.config.scaling_factor
        return latent

    @torch.no_grad()
    def latent_to_image(self, z: torch.Tensor) -> Image:
        if type(self.pipeline) == StableDiffusionXLPipeline:
            print('UPCASTING VAE')
            self.pipeline.upcast_vae()
        vae = self.pipeline.vae
        z = z.to(vae.device, vae.dtype)
        z = z / vae.config.scaling_factor
        image = vae.decode(z).sample
        image = (image + 1).div(2).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")
        return [Image.fromarray(img) for img in image]

    @torch.no_grad()
    def _run_pipeline(self, z, prompt, **kwargs):
        assert len(z.shape) == 4
        assert set(list(kwargs.keys())).issubset(self.pipeline_call_params)
        output = self.pipeline(
            prompt=[prompt] * len(z),
            latents=z.to(self.pipeline.unet.device, self.pipeline.unet.dtype),
            **kwargs,  # i.e. num_inference_steps, eta, etc.
            output_type = "latent",
            return_dict = False,
        )
        return output[0]

    def full_generation(self, z, prompt, **kwargs):
        self.print_eta(kwargs)
        self.pipeline.scheduler = self.schedulers["scheduler_full_generation"]
        return self._run_pipeline(z=z, prompt=prompt, **kwargs)

    def partial_generation(self, z, prompt, **kwargs):
        self.print_eta(kwargs)
        self.pipeline.scheduler = self.schedulers["scheduler_partial_generation"]
        return self._run_pipeline(z=z, prompt=prompt, **kwargs)
    
    def partial_generation_remaining(self, z, prompt, **kwargs):
        self.print_eta(kwargs)
        self.pipeline.scheduler = self.schedulers["scheduler_partial_generation_remaining"]
        return self._run_pipeline(z=z, prompt=prompt, **kwargs)

    def print_guidance_scale_inversion(self, kwargs):
        guidance_scale = kwargs.get("guidance_scale", 7.5)
        if guidance_scale > 1.0:
            print(
                f"Warning: An high guidance scale ({guidance_scale}) may affect the inversion quality."
            )
            # cfr https://arxiv.org/pdf/2211.09794.pdf page 4 "We observe that such a guidance scale amplifies the accumulated error"

    def print_eta(self, kwargs):
        eta = kwargs.get("eta", 0.0)
        if type(eta) == torch.Tensor:
            # We are calling the function using a spatial eta
            eta = eta.mean().item()
        if eta >= 0.9999:
            print(
                f"Warning: A high eta ({eta}) may affect the quality of the generated image."
            )

    def full_inversion(self, z, prompt, **kwargs):
        self.print_guidance_scale_inversion(kwargs)
        self.print_eta(kwargs)
        self.pipeline.scheduler = self.schedulers["scheduler_full_inversion"]
        return self._run_pipeline(z=z, prompt=prompt, **kwargs)

    def partial_inversion(self, z, prompt, **kwargs):
        self.print_guidance_scale_inversion(kwargs)
        self.print_eta(kwargs)
        self.pipeline.scheduler = self.schedulers["scheduler_partial_inversion"]
        return self._run_pipeline(z=z, prompt=prompt, **kwargs)
