"""Adapters for the official DECON dependency checkouts.

These wrappers load heavyweight upstream code lazily. The base package can
therefore be imported without SAM 2, PowerPaint, PyTorch3D, or SAT-HMR.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .types import HumanEstimate


class OfficialDependencyError(RuntimeError):
    """Raised when an official model checkout or checkpoint is unavailable."""


def _require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise OfficialDependencyError(f"{description} not found: {path}")
    return path


class OfficialSAM2Segmenter:
    """SAM 2 box-prompt adapter matching the official DECON `sam.py`."""

    def __init__(
        self,
        segmentation_root: str | Path,
        *,
        model_id: str = "facebook/sam2.1-hiera-large",
    ) -> None:
        self.root = _require_path(Path(segmentation_root), "SAM 2 checkout")
        self.model_id = model_id
        self._predictor: Any = None

    def _load(self) -> Any:
        if self._predictor is None:
            sys.path.insert(0, str(self.root))
            try:
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError as exc:
                raise OfficialDependencyError(
                    "Install the official SAM 2 dependencies from "
                    f"{self.root / 'INSTALL.md'}"
                ) from exc
            self._predictor = SAM2ImagePredictor.from_pretrained(self.model_id)
        return self._predictor

    def segment(self, image: Any, bounding_box: Tensor) -> Tensor:
        predictor = self._load()
        rgb = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
        box = bounding_box.detach().cpu().numpy().astype(np.float32)
        predictor.set_image(rgb)
        with torch.inference_mode():
            masks, _, _ = predictor.predict(box=box, multimask_output=False)
        return torch.from_numpy(np.asarray(masks[0], dtype=np.bool_))


class OfficialPowerPaintInpainter:
    """PowerPaint v2 adapter using the official `PowerPaintController`."""

    def __init__(
        self,
        inpainting_root: str | Path,
        checkpoint_dir: str | Path,
        *,
        prompt: str = "a complete person",
        negative_prompt: str = "low quality, blurry, distorted",
        fitting_degree: float = 1.0,
        ddim_steps: int = 50,
        scale: float = 7.5,
        seed: int = 42,
        version: str = "ppt-v2",
        local_files_only: bool = True,
    ) -> None:
        self.root = _require_path(Path(inpainting_root), "PowerPaint checkout")
        self.checkpoint_dir = _require_path(Path(checkpoint_dir), "PowerPaint checkpoint")
        self.settings = (prompt, negative_prompt, fitting_degree, ddim_steps, scale, seed, version, local_files_only)
        self._controller: Any = None

    def _load(self) -> Any:
        if self._controller is None:
            if not torch.cuda.is_available():
                raise OfficialDependencyError(
                    "The official PowerPaint controller requires CUDA; no CUDA device is available"
                )
            sys.path.insert(0, str(self.root))
            try:
                from app import PowerPaintController
            except ImportError as exc:
                raise OfficialDependencyError(
                    "Install PowerPaint dependencies from "
                    f"{self.root / 'README.md'}"
                ) from exc
            *_, version, _ = self.settings
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self._controller = PowerPaintController(
                dtype, str(self.checkpoint_dir), self.settings[-1], version
            )
        return self._controller

    def inpaint(self, image: Any, human_mask: Tensor, completion_mask: Tensor) -> Image.Image:
        controller = self._load()
        prompt, negative, fitting, steps, scale, seed, _, _ = self.settings
        pil_image = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image)).convert("RGB")
        mask = (completion_mask.detach().cpu().numpy().astype(np.uint8) * 255)
        input_image = {"image": pil_image, "mask": Image.fromarray(mask, mode="L")}
        outputs, _ = controller.predict(
            input_image, prompt, fitting, steps, scale, seed, negative,
            "text-guided", None, None,
        )
        return outputs[0]


class PyTorch3DSilhouetteRenderer:
    """Differentiable PyTorch3D silhouette renderer for GGDO."""

    def __init__(self, faces: Tensor, image_size: tuple[int, int], device: str = "cuda") -> None:
        try:
            from pytorch3d.renderer import (
                BlendParams, MeshRasterizer, RasterizationSettings, SoftSilhouetteShader,
            )
        except ImportError as exc:
            raise OfficialDependencyError("Install PyTorch3D for differentiable rendering") from exc
        self.faces = faces.to(device=device, dtype=torch.int64)
        self.device = torch.device(device)
        self.rasterizer = MeshRasterizer(
            raster_settings=RasterizationSettings(
                image_size=image_size, blur_radius=np.log(1.0 / 1e-4 - 1.0) * 1.5, faces_per_pixel=50
            )
        )
        self.shader = SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4, gamma=1e-4))

    def render_mask(self, vertices: Tensor, camera: Any) -> Tensor:
        from pytorch3d.structures import Meshes
        meshes = Meshes(verts=[vertices.to(self.device)], faces=[self.faces])
        fragments = self.rasterizer(meshes, cameras=camera)
        image = self.shader(fragments, meshes)[..., 3]
        return image[0]


class OfficialSATHMRGeometryEstimator:
    """Run official SAT-HMR inference and read its JSON/OBJ artifacts.

    The official SAT-HMR CLI expects an input directory and writes one JSON
    parameter file plus one OBJ per detected person. This adapter preserves
    that contract while returning the project's `HumanEstimate` objects.
    """

    def __init__(self, sat_hmr_root: str | Path, *, python_executable: str | None = None) -> None:
        self.root = _require_path(Path(sat_hmr_root), "SAT-HMR checkout")
        self.python = python_executable or sys.executable
        self.main = _require_path(self.root / "main.py", "SAT-HMR main.py")

    def estimate(self, image: Any) -> list[HumanEstimate]:
        image_path = Path(image) if isinstance(image, (str, Path)) else None
        if image_path is None:
            raise TypeError("OfficialSATHMRGeometryEstimator.estimate requires an image path")
        input_dir = self.root / "demo"
        output_dir = self.root / "demo_results"
        input_dir.mkdir(exist_ok=True)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir()
        copied = input_dir / image_path.name
        shutil.copy2(image_path, copied)
        env = os.environ.copy()
        subprocess.run(
            [self.python, str(self.main), "--cfg", "demo", "--mode", "infer"],
            cwd=self.root, env=env, check=True,
        )
        estimates: list[HumanEstimate] = []
        for params_path in sorted(output_dir.glob("*_smpl_para.json")):
            params = json.loads(params_path.read_text())
            mesh_path = params_path.with_name(params_path.name.replace("_smpl_para.json", "_smpl_mesh.obj"))
            try:
                import trimesh
                vertices = torch.from_numpy(np.asarray(trimesh.load(mesh_path, process=False).vertices)).float()
            except (ImportError, FileNotFoundError) as exc:
                raise OfficialDependencyError(
                    f"SAT-HMR mesh artifact missing or trimesh unavailable: {mesh_path}"
                ) from exc
            with Image.open(image_path) as pil_image:
                mask = torch.zeros((pil_image.height, pil_image.width), dtype=torch.bool)
            estimates.append(HumanEstimate(
                vertices=vertices, mask=mask, camera=params,
                pose=torch.tensor(params["poses"]), shape=torch.tensor(params["betas"]),
                translation=torch.tensor(params["transl"]), metadata={"source": str(params_path)},
            ))
        return estimates
