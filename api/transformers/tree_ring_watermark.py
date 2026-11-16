from api.transformers.base_watermark_transformer import BaseWaterMarkTransformer

import torch
import numpy as np

class TreeRingWatermarkTransformer(BaseWaterMarkTransformer):
    can_embed: bool = False
    can_generate: bool = True
    can_recover: bool = False
    
    def __init__(self, 
                 seed: int,
                 device: str,
                 model_dtype: torch.dtype,
                 watermark: str,
                 generation_steps: int = 50,
                 guidance_scale: float = 1.0):
        super().__init__(watermark)
        self._pipe = None  # Initialize the pipeline to None
        self._device = device
        self._model_dtype = model_dtype
        
        self._seed = seed
        self._latent_shape = None
        self._generation_steps = generation_steps
        self._guidance_scale = guidance_scale

    def generate(self, request: str = '§') -> str:
        initial_noise = self._random_noise(seed=self._seed)  # Use a fixed seed for reproducibility
        latent = self._diffusion_trajectory(initial_noise, self._generation_steps, prompt=request, guidance=self._guidance_scale)
        return decode_latent(latent), initial_noise
    
    def detect(self, media: str) -> bool:
        # Implement the logic to detect a watermark in the media
        return "watermark" in media
    
    
    @torch.inference_mode()
    def _text_embedding(self, prompt):
        tokens = self._pipe.tokenizer(
            prompt, padding='max_length', max_length=self._pipe.tokenizer.model_max_length,
            truncation=True, return_tensors='pt',
        )
        kwargs = {}
        if getattr(self._pipe.text_encoder.config, 'use_attention_mask', False):
            kwargs['attention_mask'] = tokens.attention_mask.to(self._device)
        return self._pipe.text_encoder(tokens.input_ids.to(self._device), **kwargs)[0].to(self._model_dtype)

    @torch.inference_mode()
    def ddim_transfer(self, x, prediction, alpha_from, alpha_to):
        x0 = (x.float() - (1 - alpha_from).sqrt() * prediction.float()) / alpha_from.sqrt()
        return alpha_to.sqrt() * x0 + (1 - alpha_to).sqrt() * prediction.float()

    @torch.inference_mode()
    def diffusion_trajectory(self, initial, steps, *, inverse=False, prompt='', guidance=1.0):
        assert self._pipe.scheduler.config.prediction_type == 'epsilon'
        self._pipe.scheduler.set_timesteps(steps, device=DEVICE)
        timesteps = self._pipe.scheduler.timesteps.flip(0) if inverse else self._pipe.scheduler.timesteps
        alphas = self._pipe.scheduler.alphas_cumprod.to(device=DEVICE, dtype=torch.float32)
        alpha_final = self._pipe.scheduler.final_alpha_cumprod.to(device=DEVICE, dtype=torch.float32)
        step_ratio = self._pipe.scheduler.config.num_train_timesteps // steps
        unconditional = self._text_embedding('')
        conditional = self._text_embedding(prompt) if guidance > 1 else unconditional
        x = initial.to(device=DEVICE, dtype=torch.float32).clone()
        label = 'DDIM inversion' if inverse else 'DDIM generation'
        for t in tqdm(timesteps, desc=label, leave=False):
            ti = int(t.item())
            previous = ti - step_ratio
            alpha_from = alphas[ti]
            alpha_to = alphas[previous] if previous >= 0 else alpha_final
            if inverse:
                alpha_from, alpha_to = alpha_to, alpha_from
            model_input = x.to(MODEL_DTYPE)
            eps = self._pipe.unet(model_input, t, encoder_hidden_states=unconditional).sample.float()
            if guidance > 1:
                # Two sequential calls instead of batch=2.
                eps_cond = self._pipe.unet(model_input, t, encoder_hidden_states=conditional).sample.float()
                eps = eps + guidance * (eps_cond - eps)
            x = self.ddim_transfer(x, eps, alpha_from, alpha_to)
        if not torch.isfinite(x).all():
            raise FloatingPointError('DDIM produced NaN/Inf. Restart the runtime and check the model / dtype.')
        result = x.detach().cpu()
        del x, unconditional, conditional
        pipe.maybe_free_model_hooks()
        return result

    @torch.inference_mode()
    def decode_latent(latent):
        decoded = pipe.vae.decode(
            latent.to(device=DEVICE, dtype=MODEL_DTYPE) / pipe.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        if not torch.isfinite(decoded).all():
            raise FloatingPointError('VAE decode: NaN/Inf. Try an FP32 VAE; see troubleshooting.')
        rgb = ((decoded.float() / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        pipe.maybe_free_model_hooks()
        return Image.fromarray(rgb, mode='RGB')

    @torch.inference_mode()
    def encode_image(image):
        image = image.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        array = np.asarray(image).astype(np.float32) / 127.5 - 1
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(DEVICE, dtype=MODEL_DTYPE)
        # Use mode(), not a random sample(): each image gives a deterministic input state.
        latent = pipe.vae.encode(tensor).latent_dist.mode().float() * pipe.vae.config.scaling_factor
        if not torch.isfinite(latent).all():
            raise FloatingPointError('VAE encode: NaN/Inf.')
        result = latent.detach().cpu()
        del latent, tensor
        pipe.maybe_free_model_hooks()
        return result

    def generate_image(noise, prompt):
        

    def recover_noise(image):
        latent = encode_image(image)
        return diffusion_trajectory(latent, DETECTION_STEPS, inverse=True, prompt='', guidance=1.0)
    
    
    def fft2c(x):
        return torch.fft.fftshift(torch.fft.fft2(x.detach().cpu().float()), dim=(-2, -1))

    def circle_mask(size, radius):
        y, x = np.ogrid[:size, :size]
        y = y[::-1]
        return torch.from_numpy(((x - size // 2)**2 + (y - size // 2)**2) <= radius**2)

    def make_key(shape, seed, radius, channel):
        g = torch.Generator(device='cpu').manual_seed(seed)
        source = fft2c(torch.randn(shape, generator=g))
        key = source.clone()
        for r in range(radius, 0, -1):
            disk = circle_mask(shape[-1], r)
            for c in range(shape[1]):
                key[:, c, disk] = source[0, c, 0, r].item()
        mask = torch.zeros(shape, dtype=torch.bool)
        mask[:, channel] = circle_mask(shape[-1], radius)
        return key, mask

    def inject_watermark(noise, key, mask):
        spectrum = fft2c(noise)
        spectrum[mask] = key[mask]
        return torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1))).real.float()

    def _random_noise(self, seed):
        return torch.randn(self._latent_shape, generator=torch.Generator(device='cpu').manual_seed(seed))