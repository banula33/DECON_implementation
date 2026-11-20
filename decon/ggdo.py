"""Two-stage SMPL fitting with separate paper and released-code objectives."""

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
    mask_metric: str = 'mse'
    mask_weight: float = 1.0
    pose_prior_weight: float = 0.0
    lock_depth: bool = False
    stage2_translation: bool = True
    amsgrad: bool = False
    plateau_scheduler: bool = False
    fit_at_renderer_resolution: bool = False

    @classmethod
    def released_code(cls):
        """Recoverable optimizer settings from upstream ggdo.py, not its missing renderer."""
        return cls(translation_steps=100, pose_shape_steps=100,
                   translation_lr=0.03, pose_shape_lr=0.002,
                   mask_metric='l1', mask_weight=100, pose_prior_weight=50,
                   lock_depth=True, stage2_translation=False,
                   amsgrad=True, plateau_scheduler=True, fit_at_renderer_resolution=True)

    def __post_init__(self):
        if min(self.translation_steps, self.pose_shape_steps) < 0:
            raise ValueError("optimization steps must be nonnegative")
        if min(self.translation_lr, self.pose_shape_lr) <= 0:
            raise ValueError("learning rates must be positive")
        if self.mask_metric not in ('mse', 'l1') or self.mask_weight <= 0 or self.pose_prior_weight < 0:
            raise ValueError("Invalid mask metric or objective weights")


@dataclass
class GGDOResult:
    vertices: Tensor
    geometry_mask: Tensor
    losses: list[float]
    pose: Tensor | None = None
    shape: Tensor | None = None
    translation: Tensor | None = None


def _mask_loss(predicted: Tensor, target: Tensor, metric='mse') -> Tensor:
    target = target.to(predicted).float()
    if predicted.shape != target.shape:
        raise ValueError(f"renderer mask shape {tuple(predicted.shape)} does not match target shape {tuple(target.shape)}")
    residual = predicted.float() - target
    loss = residual.square().mean() if metric == 'mse' else residual.abs().mean()
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
    """Fit translation, then SMPL parameters; retain a coherent best state.

    Vertices and translations use the original OpenCV camera coordinate frame.
    Without a body model only translation-only fitting is supported. Free vertex
    deformation is deliberately excluded: it cannot preserve the SMPL prior.
    Loss history includes the initial and every post-update objective value.
    The released-code profile uses scaled L1 plus a pose prior, not MSE.
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

    original_pose = theta.detach().clone() if theta is not None else None
    original_depth = trans.detach()[..., 2].clone()
    stage_two = False
    def evaluate():
        loss = config.mask_weight * _mask_loss(render_mask(mesh()), segmented_mask, config.mask_metric)
        if stage_two and theta is not None:
            loss = loss + config.pose_prior_weight * (theta-original_pose).square().mean()
        return loss

    with torch.no_grad():
        best_loss = float(evaluate())
    losses = [best_loss]
    best = [p.detach().clone() for p in parameters]
    second_parameters = parameters if config.stage2_translation or theta is None else [theta, beta]
    for stage, (steps, lr, active) in enumerate((
        (config.translation_steps, config.translation_lr, [trans]),
        (config.pose_shape_steps, config.pose_shape_lr, second_parameters),
    )):
        stage_two = stage == 1
        with torch.no_grad():
            best_loss = float(evaluate())
        best = [p.detach().clone() for p in parameters]
        # Freeze pose/shape during the first stage, then unfreeze them.
        for parameter in parameters:
            parameter.requires_grad_(any(parameter is item for item in active))
        optimizer = torch.optim.Adam(active, lr=lr, amsgrad=config.amsgrad)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, min_lr=1e-5, patience=5,
        ) if config.plateau_scheduler else None
        for _ in range(steps):
            optimizer.zero_grad()
            loss = evaluate()
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in active):
                raise ValueError("Renderer/body model must provide finite gradients to every optimized parameter")
            if config.lock_depth and trans.grad is not None:
                trans.grad[..., 2] = 0
            optimizer.step()
            with torch.no_grad():
                if config.lock_depth:
                    trans[..., 2].copy_(original_depth)
                value = float(evaluate())
                losses.append(value)
                if value < best_loss:
                    best_loss = value
                    best = [p.detach().clone() for p in parameters]
            if scheduler is not None:
                scheduler.step(float(loss.detach()))
        with torch.no_grad():
            for parameter, saved in zip(parameters, best):
                parameter.copy_(saved)

    with torch.no_grad():
        fitted = mesh().detach()
        mask = render_mask(fitted).detach()
    return GGDOResult(fitted, mask, losses,
                      theta.detach() if theta is not None else None,
                      beta.detach() if beta is not None else None, trans.detach())
