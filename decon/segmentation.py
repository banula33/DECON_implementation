"""Conservative cleanup of binary masks at the original image resolution."""
import numpy as np
import torch
from scipy import ndimage


def refine_mask(mask, *, sigma=0.65, min_area=12, points=None, labels=None):
    """Smooth subpixel-scale jaggies and remove only tiny islands/enclosed holes.

    All sizeable disconnected components are retained (e.g. an occluded hand).
    Components containing a positive prompt are protected from area removal;
    negative prompts protect holes. No largest-component or convex-hull filling.
    sigma is in original-image pixels; zero disables smoothing.
    """
    if not np.isfinite(sigma) or sigma < 0 or min_area < 0:
        raise ValueError('Mask smoothing and minimum area must be nonnegative')
    original = mask.detach().cpu().numpy().astype(bool)
    if original.ndim != 2:
        raise ValueError('Mask must be two-dimensional')
    if not original.any() or original.all() or (sigma == 0 and min_area == 0):
        return mask.clone().bool()
    result = original.copy()
    seeds = []
    for point, value in zip(points or [], labels or []):
        x, y = np.rint(point).astype(int)
        if 0 <= y < result.shape[0] and 0 <= x < result.shape[1]:
            seeds.append((y, x, int(value)))

    def clean(array, foreground):
        components, _ = ndimage.label(array, structure=np.ones((3, 3)))
        sizes = np.bincount(components.ravel())
        small = sizes < min_area
        small[0] = False
        if not foreground:
            # Background connected to any border is never an enclosed hole.
            border = np.concatenate((components[0], components[-1], components[:, 0], components[:, -1]))
            small[np.unique(border)] = False
        for y, x, value in seeds:
            if value == int(foreground):
                small[components[y, x]] = False
        return array & ~small[components]

    if min_area:
        result = clean(result, True)
        result = ~clean(~result, False)
    if sigma:
        # Padding prevents the crop boundary from biasing smoothing at the edge.
        padded = np.pad(result, 2, mode='edge')
        distance = ndimage.distance_transform_edt(padded) - ndimage.distance_transform_edt(~padded)
        result = ndimage.gaussian_filter(distance, sigma=sigma, mode='nearest')[2:-2, 2:-2] > 0
    if min_area:
        result = clean(result, True)
    for y, x, _ in seeds:
        result[y, x] = original[y, x]
    # An uncertain tiny mask must not silently become an empty person.
    if not result.any():
        result = original
    return torch.as_tensor(result.copy(), device=mask.device, dtype=torch.bool)
