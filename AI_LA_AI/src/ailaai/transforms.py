"""Standard crop and resampling views."""

from __future__ import annotations

from typing import Any

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def native_view(images: torch.Tensor, crop: int = 358) -> torch.Tensor:
    """Take the centered crop without rescaling its pixels."""
    if images.ndim not in (3, 4):
        raise ValueError("native_view expects CHW or BCHW RGB tensors.")
    return TF.center_crop(images, [crop, crop])


def resampled_view(
    images: torch.Tensor,
    crop: int = 358,
    size: int = 224,
    interpolation: Any = InterpolationMode.BILINEAR,
    antialias: bool = True,
) -> torch.Tensor:
    """Keep the Native crop's field of view while sampling through a smaller grid."""
    cropped = native_view(images, crop=crop)
    small = TF.resize(cropped, [size, size], interpolation=interpolation, antialias=antialias)
    return TF.resize(small, [crop, crop], interpolation=interpolation, antialias=antialias)
