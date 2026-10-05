"""High-pass image views and small 2-D spectrum helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .transforms import native_view


def gaussian_blur_rgb(images: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """Apply a channelwise Gaussian blur to CHW or BCHW RGB tensors."""
    if images.ndim == 3:
        images = images.unsqueeze(0)
        squeeze = True
    elif images.ndim == 4:
        squeeze = False
    else:
        raise ValueError("Gaussian blur expects CHW or BCHW RGB tensors.")
    if images.shape[1] != 3 or kernel_size < 1 or kernel_size % 2 == 0 or sigma <= 0:
        raise ValueError("Expected three channels, a positive odd kernel, and positive sigma.")
    axis = torch.arange(kernel_size, device=images.device, dtype=images.dtype) - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-(axis * axis) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.expand(3, 1, kernel_size, kernel_size).contiguous()
    pad = kernel_size // 2
    blurred = F.conv2d(F.pad(images, (pad, pad, pad, pad), mode="reflect"), kernel, groups=3)
    return blurred.squeeze(0) if squeeze else blurred


def highpass_view(
    images: torch.Tensor,
    crop: int = 358,
    kernel_size: int = 5,
    sigma: float = 1.0,
    gain: float = 2.0,
    offset: float = 0.5,
) -> torch.Tensor:
    """Blur RGB values before normalization and map the residual into [0, 1]."""
    cropped = native_view(images, crop=crop)
    low_frequency = gaussian_blur_rgb(cropped, kernel_size=kernel_size, sigma=sigma)
    return (gain * (cropped - low_frequency) + offset).clamp(0.0, 1.0)


def spectrum(image: torch.Tensor | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Hann-windowed, orthonormal 2-D power and log-power arrays."""
    if isinstance(image, torch.Tensor):
        image = image.detach().float().cpu().numpy()
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 3:
        if array.shape[0] in (1, 3):
            array = array.mean(axis=0)
        elif array.shape[-1] in (1, 3):
            array = array.mean(axis=-1)
        else:
            raise ValueError("A 3-D image must have a one- or three-channel axis.")
    if array.ndim != 2 or min(array.shape) < 2:
        raise ValueError("spectrum expects an HxW, CHW, or HWC image with both dimensions >= 2.")
    height, width = array.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    centered = (array - array.mean()) * window
    transform = np.fft.fftshift(np.fft.fft2(centered, norm="ortho"))
    power = np.abs(transform) ** 2
    return power, 10.0 * np.log10(power + 1e-16)
