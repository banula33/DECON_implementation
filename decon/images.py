"""Shared RGB and mask conversion without changing pixel coordinates."""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch import Tensor


def load_rgb(image: Any) -> Image.Image:
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must be a path, PIL image, or uint8 HxWx3 RGB array")
    return Image.fromarray(array)


def mask_image(mask: Tensor) -> Image.Image:
    return Image.fromarray((mask.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8))


def isolate(image: Image.Image, mask: Tensor) -> Image.Image:
    if tuple(mask.shape) != (image.height, image.width):
        raise ValueError("mask must match the original image resolution")
    return Image.composite(image, Image.new("RGB", image.size, "white"), mask_image(mask))
