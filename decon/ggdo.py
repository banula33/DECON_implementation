"""Geometry-Guided Decoupling Optimization (GGDO)."""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


Renderer = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class GGDOConfig:
    translation_steps: int = 100
    pose_shape_steps: int = 300
    translation_lr: float = 1e-2
    pose_shape_lr: float = 1e-3


@dataclass
class GGDOResult:
    vertices: Tensor
    geometry_mask: Tensor
    losses: list[float]


def _mask_loss(predicted: Tensor, target: Tensor) -> Tensor:
    predicted = predicted.float()
    target = target.float().to(device=predicted.device)
    if predicted.shape != target.shape:
        raise ValueError(
            f"renderer mask shape {tuple(predicted.shape)} does not match "
            f"target shape {tuple(target.shape)}"
        )
    return torch.mean((predicted - target) ** 2)


def optimize_geometry(
    vertices: Tensor,
    segmented_mask: Tensor,
    render_mask: Renderer,
    *,
    pose_shape_delta: Tensor | None = None,
    config: GGDOConfig | None = None,
) -> GGDOResult:
    """Align a projected human mask using the paper's two optimization stages.

    The renderer is injected so production can use PyTorch3D while tests can
    use a small differentiable renderer. The second-stage vertex delta is the
    adapter-neutral equivalent of optimizing SMPL pose and shape.
    """
    config = config or GGDOConfig()
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape [vertex_count, 3]")
    if segmented_mask.ndim != 2:
        raise ValueError("segmented_mask must have shape [height, width]")

    base_vertices = vertices.detach().clone()
    translation = torch.zeros(3, device=vertices.device, dtype=vertices.dtype, requires_grad=True)
    losses: list[float] = []

    translation_optimizer = torch.optim.Adam([translation], lr=config.translation_lr)
    for _ in range(config.translation_steps):
        translation_optimizer.zero_grad()
        loss = _mask_loss(render_mask(base_vertices + translation), segmented_mask)
        loss.backward()
        translation_optimizer.step()
        losses.append(float(loss.detach().cpu()))

    if pose_shape_delta is None:
        pose_shape_delta = torch.zeros_like(base_vertices)
    elif pose_shape_delta.shape != base_vertices.shape:
        raise ValueError("pose_shape_delta must have the same shape as vertices")
    else:
        pose_shape_delta = pose_shape_delta.detach().clone()
    pose_shape_delta.requires_grad_(True)

    pose_shape_optimizer = torch.optim.Adam(
        [pose_shape_delta, translation], lr=config.pose_shape_lr
    )
    for _ in range(config.pose_shape_steps):
        pose_shape_optimizer.zero_grad()
        optimized_vertices = base_vertices + translation + pose_shape_delta
        loss = _mask_loss(render_mask(optimized_vertices), segmented_mask)
        loss.backward()
        pose_shape_optimizer.step()
        losses.append(float(loss.detach().cpu()))

    optimized_vertices = (base_vertices + translation + pose_shape_delta).detach()
    geometry_mask = render_mask(optimized_vertices).detach()
    return GGDOResult(optimized_vertices, geometry_mask, losses)
