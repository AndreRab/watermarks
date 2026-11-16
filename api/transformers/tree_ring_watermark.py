import time
from typing import ClassVar

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from huggingface_hub import snapshot_download
from PIL import Image
from scipy.stats import ncx2
from tqdm.auto import tqdm

from .base_watermark_transformer import BaseWaterMarkTransformer


class TreeRingWatermarkTransformer(BaseWaterMarkTransformer):
    can_embed: ClassVar[bool] = False
    can_generate: ClassVar[bool] = True
    can_recover: ClassVar[bool] = False

    def __init__(
        self,
        seed: int,
        device: str,
        model_dtype: torch.dtype,
        watermark_seed: int,
        model_id: str,
        model_revision: str,
        image_size: int = 512,
        generation_steps: int = 50,
        detection_steps: int = 50,
        guidance_scale: float = 7.5,
        watermark_radius: int = 10,
        watermark_channel: int = 3,
        threshold: float = 0.01,
        model_variant: str = "fp16",
        detection_prompt: str = "",
        detection_guidance_scale: float = 1.0,
    ):
        self._seed = seed
        self._device = device
        self._model_dtype = model_dtype
        self._watermark_seed = watermark_seed
        self._model_id = model_id
        self._model_revision = model_revision
        self._image_size = image_size
        self._generation_steps = generation_steps
        self._detection_steps = detection_steps
        self._guidance_scale = guidance_scale
        self._watermark_radius = watermark_radius
        self._watermark_channel = watermark_channel
        self._threshold = threshold
        self._model_variant = model_variant
        self._detection_prompt = detection_prompt
        self._detection_guidance_scale = detection_guidance_scale

        self._pipe = self._load_pipe()
        latent_size = image_size // self._pipe.vae_scale_factor
        self._latent_shape = (1, self._pipe.unet.config.in_channels, latent_size, latent_size)
        self._key, self._mask = self._make_key(
            self._latent_shape, watermark_seed, watermark_radius, watermark_channel
        )

    def generate(
        self, request: str = "", with_watermark: bool = True
    ) -> tuple[Image.Image, torch.Tensor]:
        initial_noise = self._random_noise(self._seed)
        if with_watermark:
            initial_noise = self._inject_watermark(initial_noise, self._key, self._mask)

        latent = self._diffusion_trajectory(
            initial_noise, self._generation_steps,
            prompt=request, guidance=self._guidance_scale,
        )
        return self._decode_latent(latent), initial_noise

    def detect(self, media: Image.Image) -> tuple[dict[str, float | bool], torch.Tensor]:
        recovered_noise = self._recover_noise(media)
        metrics = self._watermark_metrics(recovered_noise)
        return metrics, recovered_noise

    def _load_pipe(self):
        model_dir = snapshot_download(
            repo_id=self._model_id,
            revision=self._model_revision,
            allow_patterns=[
                "model_index.json", "scheduler/scheduler_config.json", "tokenizer/*",
                "text_encoder/config.json", f"text_encoder/model.{self._model_variant}.safetensors",
                "unet/config.json", f"unet/diffusion_pytorch_model.{self._model_variant}.safetensors",
                "vae/config.json", f"vae/diffusion_pytorch_model.{self._model_variant}.safetensors",
            ],
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            model_dir, torch_dtype=self._model_dtype, variant=self._model_variant,
            use_safetensors=True, low_cpu_mem_usage=True, local_files_only=True,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        pipe.to(self._device)
        for module in (pipe.unet, pipe.vae, pipe.text_encoder):
            module.eval().requires_grad_(False)
        return pipe

    @torch.inference_mode()
    def _text_embedding(self, prompt):
        tokens = self._pipe.tokenizer(
            prompt, padding="max_length", max_length=self._pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        return self._pipe.text_encoder(tokens.input_ids.to(self._device))[0].to(self._model_dtype)

    @torch.inference_mode()
    def _ddim_transfer(self, x, prediction, alpha_from, alpha_to):
        x0 = (x.float() - (1 - alpha_from).sqrt() * prediction.float()) / alpha_from.sqrt()
        return alpha_to.sqrt() * x0 + (1 - alpha_to).sqrt() * prediction.float()

    @torch.inference_mode()
    def _diffusion_trajectory(self, initial, steps, inverse=False, prompt="", guidance=1.0):
        self._pipe.scheduler.set_timesteps(steps, device=self._device)
        timesteps = self._pipe.scheduler.timesteps.flip(0) if inverse else self._pipe.scheduler.timesteps
        alphas = self._pipe.scheduler.alphas_cumprod.to(device=self._device, dtype=torch.float32)
        alpha_final = self._pipe.scheduler.final_alpha_cumprod.to(device=self._device, dtype=torch.float32)
        step_ratio = self._pipe.scheduler.config.num_train_timesteps // steps
        empty_embedding = self._text_embedding("")
        embedded_prompt = self._text_embedding(prompt)
        x = initial.to(device=self._device, dtype=torch.float32).clone()
        label = "DDIM inversion" if inverse else "DDIM generation"

        for t in tqdm(timesteps, desc=label, leave=False):
            ti = int(t.item())
            previous = ti - step_ratio
            alpha_from = alphas[ti]
            alpha_to = alphas[previous] if previous >= 0 else alpha_final
            if inverse:
                alpha_from, alpha_to = alpha_to, alpha_from
            model_input = x.to(self._model_dtype)
            eps = self._pipe.unet(model_input, t, encoder_hidden_states=empty_embedding).sample.float()
            if guidance > 1:
                eps_cond = self._pipe.unet(model_input, t, encoder_hidden_states=embedded_prompt).sample.float()
                eps = eps + guidance * (eps_cond - eps)
            x = self._ddim_transfer(x, eps, alpha_from, alpha_to)

        result = x.detach().cpu()
        return result

    @torch.inference_mode()
    def _decode_latent(self, latent):
        decoded = self._pipe.vae.decode(
            latent.to(device=self._device, dtype=self._model_dtype) / self._pipe.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        rgb = ((decoded.float() / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        return Image.fromarray(rgb)

    @torch.inference_mode()
    def _encode_image(self, image):
        image = image.convert("RGB").resize((self._image_size, self._image_size), Image.Resampling.LANCZOS)
        array = np.asarray(image).astype(np.float32) / 127.5 - 1
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self._device, dtype=self._model_dtype)
        latent = self._pipe.vae.encode(tensor).latent_dist.mode().float() * self._pipe.vae.config.scaling_factor
        result = latent.detach().cpu()
        return result

    def _recover_noise(self, image):
        latent = self._encode_image(image)
        return self._diffusion_trajectory(
            latent, self._detection_steps, inverse=True,
            prompt=self._detection_prompt, guidance=self._detection_guidance_scale,
        )

    def _fft2c(self, x):
        return torch.fft.fftshift(torch.fft.fft2(x.detach().cpu().float()), dim=(-2, -1))

    def _circle_mask(self, size, radius):
        y, x = np.ogrid[:size, :size]
        y = y[::-1]
        return torch.from_numpy(((x - size // 2)**2 + (y - size // 2)**2) <= radius**2)

    def _make_key(self, shape, seed, radius, channel):
        g = torch.Generator(device="cpu").manual_seed(seed)
        source = self._fft2c(torch.randn(shape, generator=g))
        key = source.clone()
        for r in range(radius, 0, -1):
            disk = self._circle_mask(shape[-1], r)
            for c in range(shape[1]):
                key[:, c, disk] = source[0, c, 0, r].item()
        mask = torch.zeros(shape, dtype=torch.bool)
        mask[:, channel] = self._circle_mask(shape[-1], radius)
        return key, mask

    def _inject_watermark(self, noise, key, mask):
        spectrum = self._fft2c(noise)
        spectrum[mask] = key[mask]
        return torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1))).real.float()

    def _random_noise(self, seed):
        return torch.randn(self._latent_shape, generator=torch.Generator(device="cpu").manual_seed(seed))

    def _watermark_metrics(self, recovered_noise):
        y = self._fft2c(recovered_noise)[self._mask].numpy().astype(np.complex128)
        k = self._key[self._mask].numpy().astype(np.complex128)
        sigma2 = max(float(np.mean(np.abs(y)**2)), np.finfo(np.float64).tiny)
        eta = float(np.sum(np.abs(k - y)**2) / sigma2)
        noncentrality = float(np.sum(np.abs(k)**2) / sigma2)
        p = float(ncx2.cdf(eta, df=len(y), nc=noncentrality))
        return {
            "distance_l1": float(np.mean(np.abs(y - k))),
            "p_paper": p,
            "neg_log10_p": float(-np.log10(max(p, 1e-300))),
            "detected": bool(p < self._threshold),
        }
