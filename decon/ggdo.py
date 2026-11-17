"""Two-stage SMPL silhouette fitting (paper equations 3 and 4)."""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

Renderer = Callable[[Tensor], Tensor]
BodyModel = Callable[[Tensor, Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class GGDOConfig:
    translation_steps: int = 100
    pose_shape_steps: int = 300
    translation_lr: float = 1e-2
    pose_shape_lr: float = 1e-3

    def __post_init__(self):
        if min(self.translation_steps, self.pose_shape_steps) < 0:
            raise ValueError("optimization steps must be nonnegative")
        if min(self.translation_lr, self.pose_shape_lr) <= 0:
            raise ValueError("learning rates must be positive")


@dataclass
class GGDOResult:
    vertices: Tensor
    geometry_mask: Tensor
    losses: list[float]
    pose: Tensor | None = None
    shape: Tensor | None = None
    translation: Tensor | None = None


def _mask_loss(predicted: Tensor, target: Tensor) -> Tensor:
    target = target.to(predicted).float()
    if predicted.shape != target.shape:
        raise ValueError(f"renderer mask shape {tuple(predicted.shape)} does not match target shape {tuple(target.shape)}")
    loss = torch.mean((predicted.float() - target) ** 2)
    if not torch.isfinite(loss):
        raise ValueError("GGDO encountered a non-finite silhouette loss")
    return loss


def optimize_geometry(
    vertices: Tensor,
    segmented_mask: Tensor,
    render_mask: Renderer,
    *,
    body_model: BodyModel | None = None,
    pose: Tensor | None = None,
    shape: Tensor | None = None,
    translation: Tensor | None = None,
    config: GGDOConfig | None = None,
) -> GGDOResult:
    """Fit translation, then SMPL pose/shape/translation; retain the best fit.

    Vertices and translations use the original OpenCV camera coordinate frame.
    Without a body model only translation-only fitting is supported. Free vertex
    deformation is deliberately excluded: it cannot preserve the SMPL prior.
    Loss history includes the initial loss and every post-update loss.
    """
    config = config or GGDOConfig()
    if vertices.ndim != 2 or vertices.shape[-1] != 3 or not torch.isfinite(vertices).all():
        raise ValueError("vertices must be finite with shape [vertex_count, 3]")
    if segmented_mask.ndim != 2 or not torch.isfinite(segmented_mask).all():
        raise ValueError("segmented_mask must be finite with shape [height, width]")
    if not segmented_mask.any() or (segmented_mask < 0).any() or (segmented_mask > 1).any():
        raise ValueError("segmented_mask must be nonempty with values in [0, 1]")
    base = vertices.detach().clone()
    if body_model is None:
        if config.pose_shape_steps:
            raise ValueError("SMPL body_model is required for pose/shape optimization")
        trans = torch.zeros(3, device=vertices.device, dtype=vertices.dtype, requires_grad=True)
        theta = beta = None
        parameters = [trans]
        def mesh():
            return base + trans
    else:
        if pose is None or shape is None or translation is None:
            raise ValueError("SMPL pose, shape and translation are required")
        if pose.numel() != 72 or shape.numel() != 10 or translation.numel() != 3:
            raise ValueError("Expected SMPL pose (72), shape (10), translation (3)")
        theta, beta, trans = [x.detach().to(vertices).clone().requires_grad_(True) for x in (pose, shape, translation)]
        parameters = [theta, beta, trans]
        def mesh():
            return body_model(theta, beta, trans)
        with torch.no_grad():
            if not torch.allclose(mesh(), base, atol=1e-4, rtol=1e-4):
                raise ValueError("SMPL parameters do not reproduce the estimated vertices; check body model and coordinates")

    def evaluate():
        return _mask_loss(render_mask(mesh()), segmented_mask)

    with torch.no_grad():
        best_loss = float(evaluate())
    losses = [best_loss]
    best = [p.detach().clone() for p in parameters]
    for steps, lr, active in (
        (config.translation_steps, config.translation_lr, [trans]),
        (config.pose_shape_steps, config.pose_shape_lr, parameters),
    ):
        # Freeze pose/shape during the first stage, then unfreeze them.
        for parameter in parameters:
            parameter.requires_grad_(any(parameter is item for item in active))
        optimizer = torch.optim.Adam(active, lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = evaluate()
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in active):
                raise ValueError("Renderer/body model must provide finite gradients to every optimized parameter")
            optimizer.step()
            with torch.no_grad():
                value = float(evaluate())
                losses.append(value)
                if value < best_loss:
                    best_loss = value
                    best = [p.detach().clone() for p in parameters]
        with torch.no_grad():
            for parameter, saved in zip(parameters, best):
                parameter.copy_(saved)

    with torch.no_grad():
        fitted = mesh().detach()
        mask = render_mask(fitted).detach()
    return GGDOResult(fitted, mask, losses,
                      theta.detach() if theta is not None else None,
                      beta.detach() if beta is not None else None, trans.detach())
