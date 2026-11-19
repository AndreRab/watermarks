from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from scipy.stats import binomtest

from .base_watermark_transformer import BaseWaterMarkTransformer


class _BCHCodec:
    """Shortened binary BCH(n=127, t=5, polynomial=137) error-correcting code for 56-bit payloads."""

    def __init__(self, prim_poly: int = 137, t: int = 5):
        self.m = prim_poly.bit_length() - 1
        self.n = (1 << self.m) - 1
        self.t = t
        self.exp = [0] * (2 * self.n)
        self.log = [0] * (self.n + 1)
        x = 1
        for i in range(self.n):
            self.exp[i] = x
            self.exp[i + self.n] = x
            self.log[x] = i
            x <<= 1
            if x & (1 << self.m):
                x ^= prim_poly

        roots: set[int] = set()
        for i in range(1, 2 * t + 1):
            r = i % self.n
            while r not in roots:
                roots.add(r)
                r = (r * 2) % self.n

        g = [1]
        for r in sorted(roots):
            new_g = [0] * (len(g) + 1)
            for idx, coeff in enumerate(g):
                new_g[idx] ^= coeff
                if coeff:
                    new_g[idx + 1] ^= self.exp[self.log[coeff] + r]
            g = new_g

        self.g = g
        self.ecc_bits = len(g) - 1
        self.ecc_bytes = (self.ecc_bits + 7) // 8
        self.pad_bits = self.ecc_bytes * 8 - self.ecc_bits

    def encode_bits(self, data_bits: list[int]) -> list[int]:
        reg = [0] * self.ecc_bits
        for b in data_bits:
            feedback = b ^ reg[0]
            reg = reg[1:] + [0]
            if feedback:
                for i in range(self.ecc_bits):
                    reg[i] ^= self.g[i + 1]
        return reg + [0] * self.pad_bits

    def decode_bits(self, packet_bits: list[int]) -> tuple[list[int], int]:
        codeword = list(packet_bits[: len(packet_bits) - self.pad_bits])
        length = len(codeword)
        syndromes = []
        for i in range(1, 2 * self.t + 1):
            s = 0
            for idx, bit in enumerate(codeword):
                if bit:
                    s ^= self.exp[(i * (length - 1 - idx)) % self.n]
            syndromes.append(s)

        if max(syndromes) == 0:
            return codeword[: -self.ecc_bits], 0

        c_poly = [1] + [0] * (2 * self.t)
        b_poly = [1] + [0] * (2 * self.t)
        l_deg = 0
        m_shift = 1
        b_val = 1
        for step in range(2 * self.t):
            d = syndromes[step]
            for i in range(1, l_deg + 1):
                if c_poly[i] and syndromes[step - i]:
                    d ^= self.exp[self.log[c_poly[i]] + self.log[syndromes[step - i]]]
            if d == 0:
                m_shift += 1
            else:
                temp = list(c_poly)
                scale = self.exp[(self.log[d] - self.log[b_val]) % self.n]
                for i in range(2 * self.t + 1 - m_shift):
                    if b_poly[i]:
                        c_poly[i + m_shift] ^= self.exp[self.log[scale] + self.log[b_poly[i]]]
                if 2 * l_deg <= step:
                    l_deg = step + 1 - l_deg
                    b_poly = temp
                    b_val = d
                    m_shift = 1
                else:
                    m_shift += 1

        if l_deg > self.t:
            return codeword[: -self.ecc_bits], -1

        err_positions = []
        for pos in range(length):
            inv_x = (self.n - pos) % self.n
            val = 1
            for i in range(1, l_deg + 1):
                if c_poly[i]:
                    val ^= self.exp[(self.log[c_poly[i]] + i * inv_x) % self.n]
            if val == 0:
                err_positions.append(length - 1 - pos)

        if len(err_positions) != l_deg:
            return codeword[: -self.ecc_bits], -1

        for idx in err_positions:
            codeword[idx] ^= 1
        return codeword[: -self.ecc_bits], len(err_positions)


class StegaStampEncoder(nn.Module):
    """U-Net style encoder from Tancik et al. (CVPR 2020), Section 4.1."""

    def __init__(self, secret_size: int = 100):
        super().__init__()
        self.secret_dense = nn.Linear(secret_size, 7500)
        self.conv1 = nn.Conv2d(6, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.up6 = nn.Conv2d(256, 128, 2)
        self.conv6 = nn.Conv2d(256, 128, 3, padding=1)
        self.up7 = nn.Conv2d(128, 64, 2)
        self.conv7 = nn.Conv2d(128, 64, 3, padding=1)
        self.up8 = nn.Conv2d(64, 32, 2)
        self.conv8 = nn.Conv2d(64, 32, 3, padding=1)
        self.up9 = nn.Conv2d(32, 32, 2)
        self.conv9 = nn.Conv2d(70, 32, 3, padding=1)
        self.conv10 = nn.Conv2d(32, 32, 3, padding=1)
        self.residual = nn.Conv2d(32, 3, 1)

    def _upsample_conv(self, x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
        upsampled = F.interpolate(x, scale_factor=2, mode="nearest")
        padded = F.pad(upsampled, (0, 1, 0, 1))
        return F.relu(conv(padded))

    def forward(self, image: torch.Tensor, secret: torch.Tensor) -> torch.Tensor:
        secret = F.relu(self.secret_dense(secret - 0.5))
        secret = secret.view(-1, 3, 50, 50)
        secret_enlarged = F.interpolate(secret, scale_factor=8, mode="nearest")

        inputs = torch.cat([secret_enlarged, image - 0.5], dim=1)
        conv1 = F.relu(self.conv1(inputs))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv4 = F.relu(self.conv4(conv3))
        conv5 = F.relu(self.conv5(conv4))

        up6 = self._upsample_conv(conv5, self.up6)
        conv6 = F.relu(self.conv6(torch.cat([conv4, up6], dim=1)))

        up7 = self._upsample_conv(conv6, self.up7)
        conv7 = F.relu(self.conv7(torch.cat([conv3, up7], dim=1)))

        up8 = self._upsample_conv(conv7, self.up8)
        conv8 = F.relu(self.conv8(torch.cat([conv2, up8], dim=1)))

        up9 = self._upsample_conv(conv8, self.up9)
        conv9 = F.relu(self.conv9(torch.cat([conv1, up9, inputs], dim=1)))
        return self.residual(conv9)


class StegaStampDecoder(nn.Module):
    """Spatial Transformer Network + CNN decoder from Tancik et al. (CVPR 2020), Section 4.2."""

    def __init__(self, secret_size: int = 100):
        super().__init__()
        self.stn_params = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(320000, 128),
            nn.ReLU(inplace=True),
        )
        self.stn_fc_weight = nn.Parameter(torch.zeros(128, 6))
        self.stn_fc_bias = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(21632, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, secret_size),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image - 0.5
        stn_features = self.stn_params(x)
        theta = (stn_features @ self.stn_fc_weight + self.stn_fc_bias).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        transformed = F.grid_sample(x, grid, align_corners=False)
        return self.decoder(transformed)


class StegaStampPerturbationPipeline(nn.Module):
    """Differentiable physical display-imaging perturbation pipeline from Section 3 (Fig. 3)."""

    def __init__(self, image_size: int = 400):
        super().__init__()
        self._image_size = image_size
        self.register_buffer("_dct_matrix", self._make_dct_matrix())
        self.register_buffer("_y_table", self._make_y_table())
        self.register_buffer("_c_table", self._make_c_table())

    def _make_dct_matrix(self) -> torch.Tensor:
        mat = torch.zeros((8, 8), dtype=torch.float32)
        for u in range(8):
            alpha = 1.0 / np.sqrt(8.0) if u == 0 else np.sqrt(2.0 / 8.0)
            for x in range(8):
                mat[u, x] = alpha * np.cos((2 * x + 1) * u * np.pi / 16.0)
        return mat

    def _make_y_table(self) -> torch.Tensor:
        table = np.array(
            [
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99],
            ],
            dtype=np.float32,
        )
        return torch.from_numpy(table)

    def _make_c_table(self) -> torch.Tensor:
        table = np.full((8, 8), 99.0, dtype=np.float32)
        table[:4, :4] = np.array(
            [
                [17, 18, 24, 47],
                [18, 21, 26, 66],
                [24, 26, 56, 99],
                [47, 66, 99, 99],
            ],
            dtype=np.float32,
        )
        return torch.from_numpy(table)

    def perspective_warp(
        self, image: torch.Tensor, max_shift: float = 0.1, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Section 3.1: Random homography perturbing the 4 corners within +/- max_shift (up to +/- 40 px)."""
        if max_shift <= 0:
            return image
        b, c, h, w = image.shape
        device, dtype = image.device, image.dtype
        src = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=torch.float32)
        shifts = (torch.rand((b, 4, 2), generator=generator) * 2.0 - 1.0) * (2.0 * max_shift)
        dst = src.unsqueeze(0) + shifts

        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        ones = torch.ones_like(x_grid)
        coords = torch.stack([x_grid, y_grid, ones], dim=-1).view(-1, 3)

        grids = []
        for i in range(b):
            h_mat = self._solve_homography(src, dst[i]).to(device=device)
            warped = coords @ h_mat.T
            warped_xy = warped[:, :2] / warped[:, 2:3].clamp(min=1e-6)
            grids.append(warped_xy.view(h, w, 2))
        grid = torch.stack(grids, dim=0).to(dtype=dtype)
        return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def _solve_homography(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        a_rows = []
        b_vals = []
        for i in range(4):
            x, y = src[i, 0].item(), src[i, 1].item()
            u, v = dst[i, 0].item(), dst[i, 1].item()
            a_rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
            a_rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
            b_vals.extend([u, v])
        a_mat = torch.tensor(a_rows, dtype=torch.float32)
        b_vec = torch.tensor(b_vals, dtype=torch.float32).unsqueeze(1)
        h_vec = torch.linalg.solve(a_mat, b_vec).squeeze(1)
        return torch.cat([h_vec, torch.ones(1, dtype=torch.float32)]).view(3, 3)

    def motion_and_defocus_blur(
        self,
        image: torch.Tensor,
        sigma_gauss: float = 1.5,
        angle_deg: float = 35.0,
        kernel_size: int = 7,
        mode: str = "gaussian",
    ) -> torch.Tensor:
        """Section 3.2: Straight-line motion blur (3-7 px) and Gaussian defocus blur (sigma in [1, 3])."""
        if kernel_size <= 1 or sigma_gauss <= 0:
            return image
        coords = torch.arange(kernel_size, device=image.device, dtype=torch.float32) - 0.5 * (kernel_size - 1)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        if mode == "motion":
            theta = np.deg2rad(angle_deg)
            dist = xx * float(np.cos(theta)) + yy * float(np.sin(theta))
            kernel = torch.exp(-(dist**2) / (2.0 * (0.6**2))) * ((xx.abs() + yy.abs()) < (0.5 * kernel_size))
        else:
            kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * (sigma_gauss**2)))
        kernel = (kernel / kernel.sum()).to(dtype=image.dtype)
        weight = kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
        pad = kernel_size // 2
        return F.conv2d(F.pad(image, (pad, pad, pad, pad), mode="reflect"), weight, groups=3)

    def color_manipulation(
        self,
        image: torch.Tensor,
        hue_shift: tuple[float, float, float] | float = 0.05,
        desaturation: float = 0.25,
        contrast: float = 1.15,
        brightness: float = 0.05,
    ) -> torch.Tensor:
        """Section 3.3: RGB channel offset [-0.1, 0.1], linear desaturation, and affine mx + b rescaling."""
        if isinstance(hue_shift, (int, float)):
            offset = torch.tensor([hue_shift, -0.5 * hue_shift, 0.5 * hue_shift], device=image.device, dtype=image.dtype)
        else:
            offset = torch.tensor(hue_shift, device=image.device, dtype=image.dtype)
        x = (image * contrast + brightness + offset.view(1, 3, 1, 1)).clamp(0.0, 1.0)
        lum_weights = torch.tensor([0.3, 0.6, 0.1], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        lum = (x * lum_weights).sum(dim=1, keepdim=True)
        x = (1.0 - desaturation) * x + desaturation * lum
        return x.clamp(0.0, 1.0)

    def gaussian_noise(
        self, image: torch.Tensor, sigma: float = 0.04, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Section 3.4: Photon/dark/shot imaging noise modeled as Gaussian sigma ~ U[0, 0.2]."""
        if sigma <= 0:
            return image
        noise = torch.randn(image.shape, generator=generator, dtype=torch.float32).to(image.device, image.dtype) * sigma
        return (image + noise).clamp(0.0, 1.0)

    def jpeg_compression(self, image: torch.Tensor, quality: float = 65.0) -> torch.Tensor:
        """Section 3.5 (Eq. 1): Differentiable JPEG compression with Shin & Song piecewise quantization."""
        if quality >= 100.0:
            return image
        factor = (5000.0 / quality if quality < 50.0 else 200.0 - 2.0 * quality) / 100.0 + 1e-4
        orig_dtype = image.dtype
        x = image.float() * 255.0
        b, _, h, w = x.shape

        r, g, bl = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * bl
        cb = -0.168736 * r - 0.331264 * g + 0.5 * bl + 128.0
        cr = 0.5 * r - 0.418688 * g - 0.081312 * bl + 128.0

        cb_down = F.avg_pool2d(cb, kernel_size=2, stride=2)
        cr_down = F.avg_pool2d(cr, kernel_size=2, stride=2)

        y_rec = self._jpeg_channel(y, self._y_table.to(x.device), factor, h, w)
        cb_rec = self._jpeg_channel(cb_down, self._c_table.to(x.device), factor, h // 2, w // 2)
        cr_rec = self._jpeg_channel(cr_down, self._c_table.to(x.device), factor, h // 2, w // 2)

        cb_up = F.interpolate(cb_rec, size=(h, w), mode="nearest") - 128.0
        cr_up = F.interpolate(cr_rec, size=(h, w), mode="nearest") - 128.0

        r_out = y_rec + 1.402 * cr_up
        g_out = y_rec - 0.344136 * cb_up - 0.714136 * cr_up
        b_out = y_rec + 1.772 * cb_up
        out = (torch.cat([r_out, g_out, b_out], dim=1).clamp(0.0, 255.0) / 255.0).to(orig_dtype)
        return out

    def _jpeg_channel(
        self, channel: torch.Tensor, q_table: torch.Tensor, factor: float, h: int, w: int
    ) -> torch.Tensor:
        b = channel.shape[0]
        dct_m = self._dct_matrix.to(channel.device)
        patches = channel.view(b, h // 8, 8, w // 8, 8).permute(0, 1, 3, 2, 4).reshape(b, -1, 8, 8) - 128.0
        dct_coeff = dct_m @ patches @ dct_m.T
        q_scaled = (q_table * factor).view(1, 1, 8, 8)
        norm_coeff = dct_coeff / q_scaled
        quantized = self._piecewise_quantize(norm_coeff)
        dequantized = quantized * q_scaled
        idct_patches = (dct_m.T @ dequantized @ dct_m) + 128.0
        return idct_patches.view(b, h // 8, w // 8, 8, 8).permute(0, 1, 3, 2, 4).reshape(b, 1, h, w)

    def _piecewise_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Equation (1) from paper: q(x) = x^3 for |x| < 0.5, else x (applied around nearest integer)."""
        rounded = torch.round(x)
        delta = x - rounded
        approx = torch.where(delta.abs() < 0.5, delta**3, delta)
        return rounded + approx

    def forward(self, image: torch.Tensor, strength: float = 0.5, seed: int = 0) -> torch.Tensor:
        if strength <= 0:
            return image
        gen = torch.Generator(device="cpu").manual_seed(seed)
        x = self.perspective_warp(image, max_shift=0.05 * strength, generator=gen)
        x = self.motion_and_defocus_blur(x, sigma_gauss=0.8 * strength, kernel_size=5)
        x = self.color_manipulation(
            x,
            hue_shift=0.04 * strength,
            desaturation=0.15 * strength,
            contrast=1.0 + 0.1 * strength,
            brightness=0.04 * strength,
        )
        x = self.gaussian_noise(x, sigma=0.03 * strength, generator=gen)
        jpeg_q = max(50.0, 100.0 - 35.0 * strength)
        return self.jpeg_compression(x, quality=jpeg_q)


class StegaStampWatermarkTransformer(BaseWaterMarkTransformer):
    can_embed: ClassVar[bool] = True
    can_generate: ClassVar[bool] = False
    can_recover: ClassVar[bool] = False

    def __init__(
        self,
        seed: int = 2026,
        device: str = "cpu",
        model_dtype: torch.dtype = torch.float32,
        model_id: str = "vadishev/stegastamp-original",
        model_revision: str = "b89ef7cdd9dd16cbdadb5ff5da971858789208b4",
        model_filename: str = "best.pt",
        image_size: int = 400,
        secret_size: int = 100,
        bch_polynomial: int = 137,
        bch_bits: int = 5,
        threshold: float = 0.01,
        default_payload: str = "Stega!!",
    ):
        self._seed = seed
        self._device = device
        self._model_dtype = model_dtype
        self._model_id = model_id
        self._model_revision = model_revision
        self._model_filename = model_filename
        self._image_size = image_size
        self._secret_size = secret_size
        self._bch_polynomial = bch_polynomial
        self._bch_bits = bch_bits
        self._threshold = threshold
        self._default_payload = default_payload

        self._bch = _BCHCodec(bch_polynomial, bch_bits)
        self._encoder, self._decoder = self._load_models()
        self._perturbations = StegaStampPerturbationPipeline(image_size).to(
            device=self._device, dtype=self._model_dtype
        )
        self._expected_bits = self.encode_payload(default_payload)

    @property
    def watermark(self) -> torch.Tensor:
        """Return the 100-bit target codeword tensor (56 payload bits + 40 BCH ECC bits + 4 pad bits)."""
        return self._expected_bits.clone()

    @property
    def perturbations(self) -> StegaStampPerturbationPipeline:
        """Return the differentiable physical display-imaging perturbation module (Section 3)."""
        return self._perturbations

    def encode_payload(self, payload: str | list[int] | np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert a <=7-char UTF-8 string (56 bits) or raw bit array into a 100-bit StegaStamp tensor."""
        if isinstance(payload, str):
            raw_bytes = payload.encode("utf-8")[:7].ljust(7, b" ")
            data_bits = [int(b) for byte in raw_bytes for b in format(byte, "08b")]
            ecc_bits = self._bch.encode_bits(data_bits)
            bits = data_bits + ecc_bits
            bits.extend([0] * (self._secret_size - len(bits)))
            return torch.tensor(bits[: self._secret_size], dtype=torch.float32)

        if isinstance(payload, torch.Tensor):
            arr = payload.detach().cpu().float().flatten().tolist()
        else:
            arr = np.asarray(payload, dtype=np.float32).flatten().tolist()

        bits = [1 if v >= 0.5 else 0 for v in arr]
        if len(bits) == 56:
            bits = bits + self._bch.encode_bits(bits)
        if len(bits) < self._secret_size:
            bits = bits + [0] * (self._secret_size - len(bits))
        return torch.tensor(bits[: self._secret_size], dtype=torch.float32)

    def decode_payload(self, bits: torch.Tensor) -> tuple[str | None, int]:
        """Run BCH error correction on the first 96 bits to recover the 56-bit (7-char) hyperlink code."""
        bit_list = [int(v >= 0.5) for v in bits.detach().cpu().float().flatten().tolist()[:96]]
        data_bits, bitflips = self._bch.decode_bits(bit_list)
        if bitflips == -1:
            return None, -1
        byte_vals = bytes(int("".join(str(b) for b in data_bits[i : i + 8]), 2) for i in range(0, 56, 8))
        try:
            return byte_vals.decode("utf-8").rstrip(" "), bitflips
        except UnicodeDecodeError:
            return None, -1

    @torch.inference_mode()
    def embed(
        self, media: Image.Image, payload: str | list[int] | np.ndarray | torch.Tensor | None = None
    ) -> tuple[Image.Image, torch.Tensor]:
        secret_bits = self._expected_bits if payload is None else self.encode_payload(payload)
        self._expected_bits = secret_bits.clone()

        image_tensor = self._preprocess_image(media)
        secret_tensor = secret_bits.unsqueeze(0).to(device=self._device, dtype=self._model_dtype)

        residual = self._encoder(image_tensor, secret_tensor)
        encoded = (image_tensor + residual).clamp(0.0, 1.0)
        return self._tensor_to_image(encoded), residual.detach().cpu().float()

    @torch.inference_mode()
    def detect(
        self,
        media: Image.Image,
        expected_payload: str | list[int] | np.ndarray | torch.Tensor | None = None,
    ) -> tuple[dict[str, float | int | str | bool | None], torch.Tensor]:
        target_bits = self._expected_bits if expected_payload is None else self.encode_payload(expected_payload)
        image_tensor = self._preprocess_image(media)
        logits = self._decoder(image_tensor)
        probs = torch.sigmoid(logits).detach().cpu().float()[0]
        pred_bits = (probs >= 0.5).float()

        metrics = self._watermark_metrics(pred_bits, probs, target_bits)
        return metrics, pred_bits

    @torch.inference_mode()
    def apply_perturbation(self, media: Image.Image, kind: str = "all", strength: float = 0.5) -> Image.Image:
        """Apply one or all physical display-imaging perturbations from Section 3 to a PIL Image."""
        tensor = self._preprocess_image(media)
        gen = torch.Generator(device="cpu").manual_seed(self._seed)
        if kind == "warp":
            out = self._perturbations.perspective_warp(tensor, max_shift=0.08 * strength, generator=gen)
        elif kind == "blur":
            out = self._perturbations.motion_and_defocus_blur(tensor, sigma_gauss=1.5 * strength, kernel_size=7)
        elif kind == "color":
            out = self._perturbations.color_manipulation(
                tensor,
                hue_shift=0.08 * strength,
                desaturation=0.3 * strength,
                contrast=1.0 + 0.25 * strength,
                brightness=0.08 * strength,
            )
        elif kind == "noise":
            out = self._perturbations.gaussian_noise(tensor, sigma=0.06 * strength, generator=gen)
        elif kind == "jpeg":
            out = self._perturbations.jpeg_compression(tensor, quality=max(50.0, 100.0 - 50.0 * strength))
        else:
            out = self._perturbations(tensor, strength=strength, seed=self._seed)
        return self._tensor_to_image(out)

    def _load_models(self) -> tuple[StegaStampEncoder, StegaStampDecoder]:
        ckpt_path = hf_hub_download(
            repo_id=self._model_id,
            filename=self._model_filename,
            revision=self._model_revision,
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        encoder_state = checkpoint.get("encoder_state") or checkpoint["encoder"]
        decoder_state = checkpoint.get("decoder_state") or checkpoint["decoder"]

        encoder = StegaStampEncoder(self._secret_size)
        decoder = StegaStampDecoder(self._secret_size)
        encoder.load_state_dict(encoder_state)
        decoder.load_state_dict(decoder_state)

        for module in (encoder, decoder):
            module.to(device=self._device, dtype=self._model_dtype)
            module.eval().requires_grad_(False)
        return encoder, decoder

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        fitted = ImageOps.fit(
            image.convert("RGB"),
            (self._image_size, self._image_size),
            method=Image.Resampling.LANCZOS,
        )
        array = np.asarray(fitted, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self._device, dtype=self._model_dtype)

    def _tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        rgb = (tensor.detach().cpu().float().clamp(0.0, 1.0)[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(rgb)

    def _watermark_metrics(
        self, pred_bits: torch.Tensor, probs: torch.Tensor, target_bits: torch.Tensor
    ) -> dict[str, float | int | str | bool | None]:
        target = target_bits.detach().cpu().float()
        matches = (pred_bits == target).float()
        num_bits = int(target.numel())
        num_correct = int(matches.sum().item())
        bit_accuracy = float(num_correct / num_bits)

        p_value = float(binomtest(num_correct, n=num_bits, p=0.5, alternative="greater").pvalue)
        decoded_text, bch_bitflips = self.decode_payload(pred_bits)
        bch_valid = bch_bitflips >= 0

        return {
            "bit_accuracy": bit_accuracy,
            "ber": float(1.0 - bit_accuracy),
            "bit_errors": int(num_bits - num_correct),
            "mean_confidence": float((2.0 * (probs - 0.5).abs()).mean().item()),
            "bch_bitflips": int(bch_bitflips),
            "decoded_payload": decoded_text,
            "p_value": p_value,
            "neg_log10_p": float(-np.log10(max(p_value, 1e-300))),
            "detected": bool(p_value < self._threshold or bch_valid),
        }
