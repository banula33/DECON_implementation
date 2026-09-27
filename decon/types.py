"""Data contracts for the model-independent decoupling pipeline."""

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor


@dataclass
class HumanEstimate:
    """One SAT-HMR estimate and its corresponding SAM 2 mask."""

    vertices: Tensor
    mask: Tensor
    camera: Any
    pose: Optional[Tensor] = None
    shape: Optional[Tensor] = None
    translation: Optional[Tensor] = None
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[-1] != 3:
            raise ValueError("vertices must have shape [vertex_count, 3]")
        if self.mask.ndim != 2:
            raise ValueError("mask must have shape [height, width]")
        if self.translation is not None and self.translation.shape != (3,):
            raise ValueError("translation must have shape [3]")


@dataclass
class DecoupledPerson:
    """Outputs for one human after GGDO and optional inpainting."""

    estimate: HumanEstimate
    optimized_vertices: Tensor
    geometry_mask: Tensor
    completion_mask: Tensor
    image: Optional[Any] = None
    losses: Optional[list[float]] = None
