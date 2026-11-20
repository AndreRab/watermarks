"""Qu et al. (2021). Uses approximate JND and difference expansion for RDH."""

import hashlib
import hmac
import math
import secrets
import struct
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch import nn
from torch.nn import functional as F

from .base_watermark_transformer import BaseWaterMarkTransformer, DetectionResult


class VGG16GradCAM:
    WEIGHTS_URL = "https://download.pytorch.org/models/vgg16-397923af.pth"

    def __init__(self, device: str = "cpu", checkpoint: str | Path | None = None,
                 model: nn.Module | None = None, target_layer: nn.Module | None = None):
        self.device = device
        self.checkpoint = checkpoint
        self.model = model
        self.target_layer = target_layer

    def _load(self):
        if self.model is None:
            # Module names and shapes match torchvision's pretrained VGG-16.
            layers = []
            channels = 3
            for item in (64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
                         512, 512, 512, "M", 512, 512, 512, "M"):
                if item == "M":
                    layers.append(nn.MaxPool2d(2, 2))
                else:
                    layers.extend((nn.Conv2d(channels, item, 3, padding=1), nn.ReLU()))
                    channels = item
            model = nn.Module()
            model.features = nn.Sequential(*layers)
            model.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(),
                nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(), nn.Linear(4096, 1000),
            )
            state = (torch.load(self.checkpoint, map_location="cpu", weights_only=True)
                     if self.checkpoint else torch.hub.load_state_dict_from_url(
                         self.WEIGHTS_URL, map_location="cpu", check_hash=True, weights_only=True,
                     ))
            model.load_state_dict(state)
            self.model = model
            self.target_layer = model.features[29]  # last convolution's ReLU
        self.model.to(self.device).eval().requires_grad_(False)

    def __call__(self, image: Image.Image, class_index: int | None = None) -> np.ndarray:
        self._load()
        rgb = np.asarray(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)).copy()
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        activations = []
        handle = self.target_layer.register_forward_hook(lambda module, inputs, output: activations.append(output))
        try:
            with torch.enable_grad():
                x = ((x - mean) / std).requires_grad_(True)
                if hasattr(self.model, "features") and hasattr(self.model, "classifier"):
                    features = self.model.features(x)
                    logits = self.model.classifier(self.model.avgpool(features).flatten(1))
                else:
                    logits = self.model(x)
                index = int(logits.argmax(1).item()) if class_index is None else class_index
                if not isinstance(index, int) or not 0 <= index < logits.shape[1]:
                    raise ValueError("class_index is outside the classifier's output range")
                gradient, = torch.autograd.grad(logits[0, index], activations[0])
                weights = gradient.mean(dim=(-2, -1), keepdim=True)
                cam = (weights * activations[0]).sum(dim=1, keepdim=True).relu()
                cam = F.interpolate(cam, size=(image.height, image.width), mode="bilinear", align_corners=False)
                result = cam[0, 0].detach().cpu().numpy()
        finally:
            handle.remove()
        maximum = float(result.max())
        return result / maximum if maximum > 0 else np.zeros_like(result)


def approximate_jnd(channel: np.ndarray) -> np.ndarray:
    value = channel.astype(np.float64)
    background = np.clip(ndimage.uniform_filter(value, size=5, mode="reflect"), 0, 255)
    adaptation = np.where(background <= 127, 17 * (1 - np.sqrt(background / 127)) + 3,
                          3 + 3 * (background - 127) / 128)
    variance = np.maximum(0, ndimage.uniform_filter(value * value, 5) - background * background)
    gx, gy = ndimage.sobel(value, axis=1) / 8, ndimage.sobel(value, axis=0) / 8
    bins = np.floor((np.arctan2(gy, gx) % np.pi) * 8 / np.pi).astype(int) % 8
    entropy = np.zeros_like(value)
    for index in range(8):
        frequency = ndimage.uniform_filter((bins == index).astype(float), size=5)
        entropy -= frequency * np.log2(np.maximum(frequency, 1e-12))
    masking = 0.3 * np.sqrt(variance) * (1 + entropy / 3)
    return np.maximum(1.0, adaptation + masking - 0.3 * np.minimum(adaptation, masking))


class GradCAMJNDWatermarkTransformer(BaseWaterMarkTransformer):
    can_embed: ClassVar[bool] = True
    can_generate: ClassVar[bool] = False
    can_recover: ClassVar[bool] = True
    RECOVERY = struct.Struct(">IIII")  # row, column, square side, block side

    def __init__(self, key: str, device: str = "cpu", ratio: int = 3,
                 alpha: float = 0.6, beta: float = 0.5, block_size: int = 3,
                 class_index: int | None = None, x0: float = 0.4, logistic_lambda: float = 3.991,
                 vgg_checkpoint: str | Path | None = None,
                 gradcam: Callable | None = None, jnd_model: Callable | None = None):
        self._key = hashlib.sha256(key.encode("utf-8")).digest()
        self.ratio, self.alpha, self.beta = ratio, alpha, beta
        self.block_size, self.class_index = block_size, class_index
        self.x0, self.logistic_lambda = x0, logistic_lambda
        self.gradcam = gradcam if gradcam is not None else VGG16GradCAM(device, vgg_checkpoint)
        self.jnd_model = jnd_model if jnd_model is not None else approximate_jnd
        self.rdh = DifferenceExpansionRDH()

    @staticmethod
    def _array(image: Image.Image) -> np.ndarray:
        if image.mode not in ("L", "RGB"):
            raise ValueError("Use an 8-bit L or RGB image; convert explicitly before embedding")
        return np.asarray(image).copy()

    def _selection_bits(self) -> np.ndarray:
        return self._logistic_bits(self.block_size, self.x0, self.logistic_lambda)

    @staticmethod
    def _logistic_bits(block_size: int, x0: float, logistic_lambda: float) -> np.ndarray:
        sequence = []
        value = x0
        for _ in range(block_size ** 2):
            value = logistic_lambda * value * (1 - value)
            sequence.append(value)
        return (np.argsort(sequence, kind="stable") % 2).reshape(block_size, block_size)

    def _positions(self, theta: np.ndarray, side: int, channels: int) -> np.ndarray:
        return self._expand_positions(theta, side, channels, self.block_size, self.x0, self.logistic_lambda)

    @staticmethod
    def _expand_positions(theta, side, channels, block, x0, logistic_lambda) -> np.ndarray:
        grid = theta.reshape(side // block, side // block, channels)
        expanded = grid.repeat(block, 0).repeat(block, 1).astype(np.int32)
        selection = np.tile(GradCAMJNDWatermarkTransformer._logistic_bits(block, x0, logistic_lambda),
                            (side // block, side // block))
        return expanded - 1 + selection[..., None]

    def select_roi(self, media: Image.Image, heatmap: np.ndarray | None = None) -> tuple[int, int, int]:
        if heatmap is None:
            heatmap = self.gradcam(media, self.class_index)
        heatmap = np.asarray(heatmap, dtype=float)
        from matplotlib import colormaps
        red = colormaps["jet"](np.clip(heatmap, 0, 1))[..., 0]
        labels, count = ndimage.label(red > 0.5, structure=np.ones((3, 3)))
        if count:
            sizes = np.bincount(labels.reshape(-1)); sizes[0] = 0
            row, column = ndimage.center_of_mass(labels == int(sizes.argmax()))
        else:
            row, column = (media.height - 1) / 2, (media.width - 1) / 2
        unit = 2 * self.block_size
        side = min(math.ceil(media.height / (self.ratio * unit)),
                   math.ceil(media.width / (self.ratio * unit))) * unit
        top = int(np.clip(round(row - side / 2), 0, media.height - side))
        left = int(np.clip(round(column - side / 2), 0, media.width - side))
        return top, left, side

    def _visible_embed(self, array: np.ndarray, logo: np.ndarray, roi: tuple[int, int, int]):
        top, left, side = roi
        source = array[..., None] if array.ndim == 2 else array
        channels = source.shape[2]
        result = source.copy()
        thresholds = np.stack([self.jnd_model(source[..., c]) for c in range(channels)], axis=-1)
        block = self.block_size
        count = side // block
        region = np.s_[top:top + side, left:left + side]

        def blocks(values):
            return values.reshape(count, block, count, block, channels).transpose(0, 2, 4, 1, 3)

        pixels = blocks(source[region].astype(np.int32))
        jnd = self.alpha * blocks(thresholds[region])
        bits = logo.reshape(count, block, count, block).transpose(0, 2, 1, 3)
        bits = bits[:, :, None].astype(np.int32)
        white = bits.all(axis=(-2, -1))
        selection = self._selection_bits()
        candidates, errors, scores = [], [], []
        for theta in range(1, 8):
            position = theta - 1 + selection
            delta = (bits - ((pixels >> position) & 1)) * (1 << position)
            candidates.append(pixels + delta)
            errors.append((delta ** 2).sum(axis=(-2, -1)))
            scores.append(np.log10(np.clip(np.abs(delta) / jnd, 1, 10)).mean(axis=(-2, -1)))

        scores = np.stack(scores)
        allowed = ~white | (scores >= self.beta)
        infeasible = ~allowed.any(axis=0)
        allowed |= infeasible & np.isclose(scores, scores.max(axis=0))
        best = np.argmin(np.where(allowed, errors, np.inf), axis=0)
        selected = np.take_along_axis(np.stack(candidates), best[None, ..., None, None], axis=0)[0]
        result[region] = selected.transpose(0, 3, 1, 4, 2).reshape(side, side, channels)
        theta = (best + 1).astype(np.uint8).ravel()
        unsatisfied = int(infeasible.sum())
        positions = self._positions(theta, side, channels)
        original_bits = ((source[top:top + side, left:left + side].astype(np.int32) >> positions) & 1).astype(np.uint8)
        visible = result[..., 0] if array.ndim == 2 else result
        return visible, theta, original_bits, unsatisfied

    def _seal(self, data: bytes) -> bytes:
        nonce = secrets.token_bytes(16)
        stream = hashlib.shake_256(b"GCJND-stream" + self._key + nonce).digest(len(data))
        encrypted = bytes(a ^ b for a, b in zip(data, stream))
        tag = hmac.digest(self._key, b"GCJND-auth" + nonce + encrypted, "sha256")
        return nonce + tag + encrypted

    def _open(self, payload: bytes) -> bytes:
        if len(payload) < 48:
            raise InvalidWatermarkError("Incomplete authenticated recovery packet")
        nonce, tag, encrypted = payload[:16], payload[16:48], payload[48:]
        expected = hmac.digest(self._key, b"GCJND-auth" + nonce + encrypted, "sha256")
        if not hmac.compare_digest(tag, expected):
            raise InvalidWatermarkError("Wrong key or damaged recovery packet")
        stream = hashlib.shake_256(b"GCJND-stream" + self._key + nonce).digest(len(encrypted))
        return bytes(a ^ b for a, b in zip(encrypted, stream))

    def embed(self, media: Image.Image, payload: Image.Image,
              *, heatmap: np.ndarray | None = None) -> tuple[Image.Image, dict]:
        array = self._array(media)
        roi = self.select_roi(media, heatmap)
        top, left, side = roi
        logo = np.asarray(payload.convert("L").resize((side, side), Image.Resampling.NEAREST)) >= 128
        visible, theta, original_bits, unsatisfied = self._visible_embed(array, logo, roi)
        digest = hashlib.sha256(array.tobytes()).digest()
        recovery = (self.RECOVERY.pack(top, left, side, self.block_size) + digest
                    + struct.pack(">dd", self.x0, self.logistic_lambda)
                    + theta.tobytes() + np.packbits(original_bits.reshape(-1)).tobytes())
        marked, report = self.rdh.embed(visible, self._seal(zlib.compress(recovery, 9)))
        mse = float(np.mean((array.astype(float) - marked.astype(float)) ** 2))
        report.update({"roi": roi, "unsatisfied_white_blocks": unsatisfied,
                       "psnr": float(10 * np.log10(255 ** 2 / mse)) if mse else float("inf"),
                       "jnd_model": getattr(self.jnd_model, "__name__", type(self.jnd_model).__name__),
                       "rdh_method": "difference_expansion"})
        return Image.fromarray(marked), report

    def extract(self, media: Image.Image) -> tuple[Image.Image, Image.Image]:
        array = self._array(media)
        visible, payload = self.rdh.extract(array)
        raw = bounded_decompress(self._open(payload), array.size * 2 + 128)
        top, left, side, block = self.RECOVERY.unpack(raw[:16])
        digest = raw[16:48]
        x0, logistic_lambda = struct.unpack(">dd", raw[48:64])
        source = visible[..., None] if visible.ndim == 2 else visible
        channels = source.shape[2]
        count = (side // block) ** 2 * channels
        bit_count = side * side * channels
        theta = np.frombuffer(raw[64:64 + count], dtype=np.uint8)
        positions = self._expand_positions(theta, side, channels, block, x0, logistic_lambda)
        region = source[top:top + side, left:left + side].astype(np.int32)
        logo_bits = (region >> positions) & 1
        original_bits = unpack_bytes(raw[64 + count:], bit_count).reshape(side, side, channels)
        restored = region + (original_bits.astype(np.int32) - logo_bits) * (1 << positions)
        source[top:top + side, left:left + side] = restored.astype(np.uint8)
        original = source[..., 0] if array.ndim == 2 else source
        if hashlib.sha256(original.tobytes()).digest() != digest:
            raise InvalidWatermarkError("Recovered pixels do not match the original")
        return Image.fromarray(original), Image.fromarray((logo_bits[..., 0] * 255).astype(np.uint8))

    def recover_original(self, media: Image.Image) -> Image.Image:
        return self.extract(media)[0]

    def detect(self, media: Image.Image) -> DetectionResult[Image.Image]:
        try:
            _, logo = self.extract(media)
        except InvalidWatermarkError:
            return DetectionResult(detected=False)
        return DetectionResult(detected=True, payload=logo, metrics={"recovery_verified": 1.0})


class CapacityError(ValueError):
    """The carrier cannot hold the payload and its recovery overhead losslessly."""


class InvalidWatermarkError(ValueError):
    """Missing, damaged, or incompatible reversible watermark."""


def unpack_bytes(data: bytes, count: int | None = None) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), count=count)


def bounded_decompress(data: bytes, limit: int) -> bytes:
    decoder = zlib.decompressobj()
    try:
        result = decoder.decompress(data, limit + 1)
    except zlib.error as exc:
        raise InvalidWatermarkError("Invalid compressed recovery data") from exc
    if len(result) > limit or not decoder.eof or decoder.unused_data:
        raise InvalidWatermarkError("Invalid recovery data size")
    return result


class DifferenceExpansionRDH:
    MAGIC = b"GCJND001"
    # magic, reserved samples, compressed map bytes, payload bytes, CRC32
    HEADER = struct.Struct(">8sIIII")

    @staticmethod
    def _flatten(image: np.ndarray) -> np.ndarray:
        # Pair spatial neighbours within each channel, rather than RGB components.
        return np.moveaxis(image, -1, 0).reshape(-1).copy() if image.ndim == 3 else image.reshape(-1).copy()

    @staticmethod
    def _reshape(flat: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        if len(shape) == 3:
            return np.moveaxis(flat.reshape(shape[2], shape[0], shape[1]), 0, -1).copy()
        return flat.reshape(shape).copy()

    @staticmethod
    def _pairs(flat: np.ndarray, reserve: int):
        count = (flat.size - reserve) // 2
        pairs = flat[reserve:reserve + 2 * count].reshape(-1, 2).astype(np.int32)
        average = (pairs[:, 0] + pairs[:, 1]) // 2
        difference = pairs[:, 0] - pairs[:, 1]
        return pairs, average, difference

    def embed(self, image: np.ndarray, payload: bytes) -> tuple[np.ndarray, dict[str, int]]:
        if image.dtype != np.uint8 or image.ndim not in (2, 3):
            raise ValueError("RDH requires a uint8 grayscale or RGB array")
        original = self._flatten(image)
        # Try small differences first to reduce carrier distortion.
        best_capacity = 0
        for threshold in (2, 4, 8, 16, 32, 64, 127):
            reserve = self.HEADER.size * 8
            for _ in range(32):
                if reserve >= original.size:
                    break
                pairs, average, difference = self._pairs(original, reserve)
                expanded = 2 * difference
                # Both possible embedded bits must stay in the uint8 range.
                eligible = ((np.abs(difference) <= threshold)
                            & (average + (expanded + 2) // 2 <= 255)
                            & (average + (expanded + 1) // 2 >= 0)
                            & (average - expanded // 2 <= 255)
                            & (average - (expanded + 1) // 2 >= 0))
                location = zlib.compress(np.packbits(eligible).tobytes(), level=9)
                needed_reserve = (self.HEADER.size + len(location)) * 8
                if needed_reserve > reserve:
                    reserve = needed_reserve
                    continue
                capacity = int(eligible.sum())
                best_capacity = max(best_capacity, capacity - reserve)
                required = reserve + len(payload) * 8
                if capacity < required:
                    break
                bits = np.concatenate((original[:reserve] & 1, unpack_bytes(payload)))
                selected = np.flatnonzero(eligible)[:required]
                expanded = 2 * difference[selected] + bits.astype(np.int32)
                pairs[selected, 0] = average[selected] + (expanded + 1) // 2
                pairs[selected, 1] = average[selected] - expanded // 2
                flat = original.copy()
                flat[reserve:reserve + pairs.size] = pairs.reshape(-1).astype(np.uint8)
                checksum = zlib.crc32(payload)
                header = self.HEADER.pack(self.MAGIC, reserve, len(location), len(payload), checksum)
                bootstrap = unpack_bytes(header + location)
                flat[:bootstrap.size] = (flat[:bootstrap.size] & 254) | bootstrap
                return self._reshape(flat, image.shape), {
                    "rdh_payload_bytes": len(payload), "rdh_capacity_bits": capacity,
                    "rdh_bootstrap_bits": reserve, "rdh_threshold": threshold,
                }
        raise CapacityError(
            f"Insufficient reversible capacity: need {len(payload) * 8} payload bits, "
            f"at most {max(0, best_capacity)} after bootstrap overhead. "
            "Use a larger/smoother image or a smaller logo (increase ratio)."
        )

    def extract(self, image: np.ndarray) -> tuple[np.ndarray, bytes]:
        flat = self._flatten(image)
        header_bits = self.HEADER.size * 8
        if flat.size < header_bits:
            raise InvalidWatermarkError("Image is too small for the RDH header")
        header = np.packbits(flat[:header_bits] & 1).tobytes()
        magic, reserve, map_length, payload_length, checksum = self.HEADER.unpack(header)
        if magic != self.MAGIC:
            raise InvalidWatermarkError("No Grad-CAM/JND recovery header found")
        map_end = (self.HEADER.size + map_length) * 8
        if not header_bits <= map_end <= reserve < flat.size or reserve % 8:
            raise InvalidWatermarkError("Invalid RDH bootstrap length")
        pairs, average, difference = self._pairs(flat, reserve)
        required = reserve + payload_length * 8
        if required > len(pairs):
            raise InvalidWatermarkError("Invalid RDH payload length")
        location_bytes = np.packbits(flat[header_bits:map_end] & 1).tobytes()
        location = bounded_decompress(location_bytes, (len(pairs) + 7) // 8)
        if len(location) != (len(pairs) + 7) // 8:
            raise InvalidWatermarkError("Invalid location map")
        selected = np.flatnonzero(unpack_bytes(location, len(pairs)))[:required]
        if len(selected) != required:
            raise InvalidWatermarkError("Incomplete reversible carrier")
        bits = (difference[selected] & 1).astype(np.uint8)
        original_difference = difference[selected] // 2
        pairs[selected, 0] = average[selected] + (original_difference + 1) // 2
        pairs[selected, 1] = average[selected] - original_difference // 2
        if (pairs < 0).any() or (pairs > 255).any():
            raise InvalidWatermarkError("Corrupted reversible carrier")
        flat[reserve:reserve + pairs.size] = pairs.reshape(-1).astype(np.uint8)
        flat[:reserve] = (flat[:reserve] & 254) | bits[:reserve]
        payload = np.packbits(bits[reserve:]).tobytes()
        if zlib.crc32(payload) != checksum:
            raise InvalidWatermarkError("Recovery payload checksum mismatch")
        return self._reshape(flat, image.shape), payload
