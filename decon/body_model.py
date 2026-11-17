"""Differentiable SMPL wrapper matching SAT-HMR's neutral, uncentered body."""

from pathlib import Path

import torch
from torch import Tensor


class SMPLBodyModel:
    def __init__(self, model_path: str | Path, device: str = "cuda") -> None:
        import smplx

        self.device = torch.device(device)
        self.model = smplx.create(
            str(model_path), model_type="smpl", gender="neutral",
            create_global_orient=False, create_body_pose=False,
            create_betas=False, create_transl=False,
        ).to(self.device).eval()
        self.model.requires_grad_(False)
        self.faces = torch.as_tensor(self.model.faces.astype("int64"), device=self.device)
        # Dominant skinning joint gives every SMPL vertex a stable anatomical
        # region, including wrists/hands, without an extra parsing checkpoint.
        weights = getattr(self.model, "lbs_weights", None)
        self.vertex_parts = weights.argmax(dim=-1).detach() if weights is not None else None

    def __call__(self, pose: Tensor, shape: Tensor, translation: Tensor) -> Tensor:
        output = self.model(
            global_orient=pose.reshape(1, 72)[:, :3],
            body_pose=pose.reshape(1, 72)[:, 3:],
            betas=shape.reshape(1, -1),
        )
        return output.vertices[0] + translation.reshape(1, 3)
