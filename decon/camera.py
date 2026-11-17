"""Original-resolution pinhole camera; coordinates follow OpenCV (x right, y down)."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PinholeCamera:
    intrinsics: Tensor
    image_size: tuple[int, int]  # height, width

    def __post_init__(self) -> None:
        if self.intrinsics.shape != (3, 3) or not torch.isfinite(self.intrinsics).all():
            raise ValueError("intrinsics must be a finite 3x3 matrix")
        if min(self.image_size) <= 0 or self.intrinsics[0, 0] <= 0 or self.intrinsics[1, 1] <= 0:
            raise ValueError("image dimensions and focal lengths must be positive")

    def project(self, vertices: Tensor) -> Tensor:
        if not torch.isfinite(vertices).all() or (vertices[:, 2] <= 0).any():
            raise ValueError("Camera-space vertices must be finite and in front of the camera")
        pixels = vertices @ self.intrinsics.to(vertices).T
        return pixels[:, :2] / pixels[:, 2:3]

    def bounding_box(self, vertices: Tensor, padding: float = 0.025) -> Tensor:
        xy = self.project(vertices)
        lower, upper = xy.amin(0), xy.amax(0)
        margin = (upper - lower) * padding
        h, w = self.image_size
        limit = vertices.new_tensor([w - 1, h - 1])
        lower = torch.minimum((lower - margin).clamp(min=0), limit)
        upper = torch.minimum((upper + margin).clamp(min=0), limit)
        if (upper <= lower).any():
            raise ValueError("Projected person has no usable bounding box inside the image")
        return torch.cat((lower, upper))

    def to_pytorch3d(self, device: torch.device):
        from pytorch3d.utils import cameras_from_opencv_projection

        return cameras_from_opencv_projection(
            R=torch.eye(3, device=device)[None],
            tvec=torch.zeros(1, 3, device=device),
            camera_matrix=self.intrinsics.to(device=device, dtype=torch.float32)[None],
            image_size=torch.tensor([self.image_size], device=device, dtype=torch.float32),
        )

    @classmethod
    def from_resized(cls, intrinsics: Tensor, resized_size: tuple[int, int], image_size: tuple[int, int]):
        """Undo SAT-HMR's resize, including integer rounding on the shorter edge."""
        h, w = image_size
        rh, rw = resized_size
        matrix = intrinsics.detach().clone().float()
        matrix[0] *= w / rw
        matrix[1] *= h / rh
        return cls(matrix, image_size)
