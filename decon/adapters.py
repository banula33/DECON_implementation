"""Replaceable interfaces for SAT-HMR, SAM 2, PyTorch3D, and PowerPaint."""

from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from torch import Tensor

from .types import HumanEstimate


class GeometryEstimator(Protocol):
    def estimate(self, image: Any) -> Sequence[HumanEstimate]: ...


class Segmenter(Protocol):
    def segment(self, image: Any, bounding_box: Tensor) -> Tensor: ...


class Inpainter(Protocol):
    def inpaint(self, image: Any, human_mask: Tensor, completion_mask: Tensor) -> Any: ...


class MeshRenderer(Protocol):
    def render_mask(self, vertices: Tensor, camera: Any) -> Tensor: ...


class ImageWriter(Protocol):
    def save(self, image: Any, path: Path) -> None: ...


BoundingBoxer = Callable[[Tensor, Any], Tensor]
