"""Prepare person-centered diffusion inputs without changing visible pixels."""

import numpy as np
from PIL import Image


def prepare_completion(image, human_mask, completion_mask, *, context=0.12, dilation=12):
    """Square crop with margin; dilation is measured at 640px diffusion size.

    The larger mask is used only for generation. The caller must composite
    exclusively into the original completion mask, preserving visible pixels.
    """
    union = np.asarray(human_mask, dtype=bool) | np.asarray(completion_mask, dtype=bool)
    y, x = np.where(union)
    if not len(x):
        raise ValueError("Cannot crop an empty person")
    span = max(int(x.max()-x.min()+1), int(y.max()-y.min()+1))
    side = max(8, int(np.ceil(span * (1 + 2*context))))
    left = int((int(x.min()) + int(x.max()) + 1 - side)//2)
    top = int((int(y.min()) + int(y.max()) + 1 - side)//2)
    box = (left, top, left+side, top+side)
    # Explicit white padding, including people touching image edges.
    crop = Image.new('RGB', (side, side), 'white')
    crop.paste(image, (-left, -top))
    mask = Image.fromarray(np.asarray(completion_mask, dtype=np.uint8)*255)
    mask = mask.crop(box)
    radius = max(1, int(round(dilation * side / 640))) if dilation else 0
    if radius:
        try:
            from scipy.ndimage import maximum_filter
        except ImportError:
            # Lightweight installations can still use the adapter with test
            # doubles; official model environments include SciPy for SMPL.
            from PIL import ImageFilter
            mask = mask.filter(ImageFilter.MaxFilter(2*radius+1))
        else:
            mask = Image.fromarray(maximum_filter(np.asarray(mask), size=2*radius+1, mode='constant'))
    return crop, mask.convert('RGB'), box


def restore_completion(generated, original, completion_mask, box):
    left, top, right, bottom = box
    generated = generated.resize((right-left, bottom-top), Image.Resampling.LANCZOS)
    canvas = original.copy()
    canvas.paste(generated, (left, top))
    mask = Image.fromarray(np.asarray(completion_mask, dtype=np.uint8)*255)
    return Image.composite(canvas, original, mask)
