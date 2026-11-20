"""Adapters for the official DECON dependency checkouts.

These wrappers load heavyweight upstream code lazily. The base package can
therefore be imported without SAM 2, PowerPaint, PyTorch3D, or SAT-HMR.
"""

from __future__ import annotations

import os
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .types import HumanEstimate
from .camera import PinholeCamera
from .images import isolate, load_rgb, mask_image


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
        checkpoint: str | Path | None = None,
        model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        device: str = "cuda",
    ) -> None:
        self.root = _require_path(Path(segmentation_root).resolve(), "SAM 2 checkout")
        self.model_id = model_id
        self.checkpoint = Path(checkpoint).resolve() if checkpoint else None
        self.model_config = model_config
        self.device = device
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
            if self.checkpoint is not None:
                from sam2.build_sam import build_sam2
                _require_path(self.checkpoint, "SAM 2 checkpoint")
                self._predictor = SAM2ImagePredictor(build_sam2(self.model_config, str(self.checkpoint), device=self.device))
            else:
                self._predictor = SAM2ImagePredictor.from_pretrained(self.model_id, device=self.device)
        return self._predictor

    def segment(self, image: Any, bounding_box: Tensor, *, points=None, labels=None, regions=None) -> Tensor:
        rgb = np.array(load_rgb(image), copy=True)
        box = bounding_box.detach().cpu().numpy().astype(np.float32)
        if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= box[:2]).any():
            raise ValueError("Bounding box must be finite [left, top, right, bottom]")
        prompts = {}
        if labels is not None and points is None:
            raise ValueError("Labels require points")
        if points is not None:
            points = np.asarray(points, dtype=np.float32)
            labels = np.asarray(labels)
            if points.ndim != 2 or points.shape[1] != 2 or labels.shape != (len(points),):
                raise ValueError("Points must be Nx2 with one label per point")
            if not np.isfinite(points).all() or not np.isin(labels, [0, 1]).all():
                raise ValueError("Points must be finite and labels must be 0 or 1")
            if (points < 0).any() or (points[:, 0] >= rgb.shape[1]).any() or (points[:, 1] >= rgb.shape[0]).any():
                raise ValueError("Points must lie inside the input image")
            prompts = dict(point_coords=points, point_labels=labels.astype(np.int32))
        predictor = self._load()
        with torch.inference_mode():
            predictor.set_image(rgb)
        # The released sam.py encodes the image outside autocast and enables
        # bfloat16 for the prompted mask prediction only.
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16,
                                                   enabled=torch.device(self.device).type == 'cuda'):
            masks, _, _ = predictor.predict(box=box, multimask_output=False, **prompts)
        result = np.array(masks[0], dtype=np.bool_, copy=True)
        # A small hand can disappear when the entire high-resolution image is
        # encoded at SAM's fixed resolution. Re-encode explicitly selected local
        # regions, then add their visible foreground to the whole-person mask.
        for region in regions or []:
            if not isinstance(region, dict) or 'box' not in region or 'points' not in region or 'labels' not in region:
                raise ValueError("Each region must contain box, points, and labels")
            crop_box = np.asarray(region['box'])
            if crop_box.shape != (4,) or not np.isfinite(crop_box).all():
                raise ValueError("Region box must contain four finite coordinates")
            x0, y0, x1, y1 = crop_box.astype(int)
            if not (0 <= x0 < x1 <= rgb.shape[1] and 0 <= y0 < y1 <= rgb.shape[0]):
                raise ValueError("Region box must lie inside the image")
            region_points = np.asarray(region['points'], dtype=np.float32) - [x0, y0]
            region_labels = np.asarray(region['labels'])
            if region_points.ndim != 2 or region_points.shape[1] != 2 or region_labels.shape != (len(region_points),):
                raise ValueError("Region points must be Nx2 with one label per point")
            local_box = torch.as_tensor(region.get('prompt_box', [x0, y0, x1-1, y1-1]), dtype=torch.float32)
            if local_box.shape != (4,) or not torch.isfinite(local_box).all():
                raise ValueError("Region prompt_box must contain four finite coordinates")
            local_box -= torch.tensor([x0, y0, x0, y0])
            local = self.segment(rgb[y0:y1, x0:x1], local_box,
                                 points=region_points, labels=region_labels)
            result[y0:y1, x0:x1] |= local.numpy()
        return torch.from_numpy(result)

    def release(self) -> None:
        """Release model and cached image embeddings before completion."""
        self._predictor = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
        crop_to_person: bool = True,
        mask_dilation: int = 12,
        task: str = 'text-guided',
        debug_directory: str | Path | None = None,
    ) -> None:
        self.root = _require_path(Path(inpainting_root).resolve(), "PowerPaint checkout")
        self.checkpoint_dir = _require_path(Path(checkpoint_dir).resolve(), "PowerPaint checkpoint")
        self.settings = (prompt, negative_prompt, fitting_degree, ddim_steps, scale, seed, version, local_files_only)
        self._controller: Any = None
        if mask_dilation < 0:
            raise ValueError("mask_dilation must be nonnegative")
        self.crop_to_person = crop_to_person
        self.mask_dilation = mask_dilation
        if task not in ('text-guided', 'shape-guided'):
            raise ValueError("task must be text-guided or shape-guided")
        self.task = task
        self.debug_directory = Path(debug_directory) if debug_directory else None

    def _load(self) -> Any:
        if self._controller is None:
            if not torch.cuda.is_available():
                raise OfficialDependencyError(
                    "The official PowerPaint controller requires CUDA; no CUDA device is available"
                )
            sys.path.insert(0, str(self.root))
            try:
                app_path = _require_path(self.root / "app.py", "PowerPaint app.py")
                spec = importlib.util.spec_from_file_location("decon_powerpaint_app", app_path)
                module = importlib.util.module_from_spec(spec)
                # Upstream app.py disables gradients at import time. Restore
                # the caller's state so GGDO can fit subsequent people.
                with torch.set_grad_enabled(torch.is_grad_enabled()):
                    spec.loader.exec_module(module)
                PowerPaintController = module.PowerPaintController
            except ImportError as exc:
                raise OfficialDependencyError(
                    "Install PowerPaint dependencies from "
                    f"{self.root / 'README.md'}"
                ) from exc
            *_, version, _ = self.settings
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            if version == "ppt-v2":
                from .powerpaint_loader import build_v2_controller
                self._controller = build_v2_controller(
                    module, self.checkpoint_dir, dtype, self.settings[-1],
                )
            else:
                self._controller = PowerPaintController(
                    dtype, str(self.checkpoint_dir), self.settings[-1], version
                )
        return self._controller

    def inpaint(self, image: Any, human_mask: Tensor, completion_mask: Tensor) -> Image.Image:
        pil_image = load_rgb(image)
        human_mask = human_mask.detach().cpu().bool()
        completion_mask = completion_mask.detach().cpu().bool()
        if completion_mask.shape != human_mask.shape:
            raise ValueError("Human and completion masks must have the same shape")
        completion_mask = completion_mask & ~human_mask
        isolated = isolate(pil_image, human_mask)
        if not completion_mask.any():
            return isolated
        controller = self._load()
        prompt, negative, fitting, steps, scale, seed, _, _ = self.settings
        # PowerPaint v2 multiplies a HxWx3 image by its mask; it needs RGB.
        input_image = {"image": isolated, "mask": mask_image(completion_mask).convert("RGB")}
        box = None
        if self.crop_to_person:
            from .completion import prepare_completion
            crop, generation_mask, box = prepare_completion(
                isolated, human_mask, completion_mask, dilation=self.mask_dilation,
            )
            input_image = {"image": crop, "mask": generation_mask}
        if self.debug_directory is not None:
            self.debug_directory.mkdir(parents=True, exist_ok=True)
            input_image['image'].save(self.debug_directory/'generation_input.png')
            input_image['mask'].save(self.debug_directory/'generation_mask.png')
        outputs, _ = controller.predict(
            input_image, prompt, fitting, steps, scale, seed, negative,
            self.task, None, None,
        )
        if self.debug_directory is not None:
            import json
            outputs[0].save(self.debug_directory/'raw_generation.png')
            (self.debug_directory/'transform.json').write_text(json.dumps({
                'original_size':pil_image.size,'crop_box':box,
                'raw_generation_size':outputs[0].size,'task':self.task,
            },indent=2)+'\n')
        if box is not None:
            from .completion import restore_completion
            return restore_completion(outputs[0].convert("RGB"), isolated, completion_mask, box)
        generated = outputs[0].convert("RGB").resize(pil_image.size, Image.Resampling.LANCZOS)
        # Preserve visible pixels exactly and keep all other subjects/background out.
        return Image.composite(generated, isolated, mask_image(completion_mask))


class PyTorch3DSilhouetteRenderer:
    """Differentiable PyTorch3D silhouette renderer for GGDO."""

    def __init__(self, faces: Tensor, image_size: tuple[int, int], device: str = "cuda",
                 max_render_size: int | None = 1024) -> None:
        try:
            from pytorch3d.renderer import (
                BlendParams, MeshRasterizer, RasterizationSettings, SoftSilhouetteShader,
            )
        except ImportError as exc:
            raise OfficialDependencyError("Install PyTorch3D for differentiable rendering") from exc
        self.image_size = image_size
        if max_render_size is not None and max_render_size <= 0:
            raise ValueError("max_render_size must be positive or None")
        scale = min(1.0, max_render_size / max(image_size)) if max_render_size else 1.0
        self.raster_size = tuple(max(1, round(size * scale)) for size in image_size)
        self.faces = faces.to(device=device, dtype=torch.int64)
        self.device = torch.device(device)
        self.rasterizer = MeshRasterizer(
            raster_settings=RasterizationSettings(
                image_size=self.raster_size, blur_radius=np.log(1.0 / 1e-4 - 1.0) * 1e-4, faces_per_pixel=50,
                perspective_correct=True
            )
        )
        self.shader = SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4, gamma=1e-4))

    def render_mask_native(self, vertices: Tensor, camera: Any) -> Tensor:
        return self.render_mask(vertices, camera, restore_resolution=False)

    def render_mask(self, vertices: Tensor, camera: Any, *, restore_resolution=True) -> Tensor:
        from pytorch3d.structures import Meshes
        if isinstance(camera, PinholeCamera):
            if camera.image_size != self.image_size:
                raise ValueError("Camera and renderer image sizes differ")
            if self.raster_size != self.image_size:
                camera = PinholeCamera.from_resized(camera.intrinsics, camera.image_size, self.raster_size)
            camera = camera.to_pytorch3d(self.device)
        elif self.raster_size != self.image_size:
            raise ValueError("Bounded rendering requires a PinholeCamera; use max_render_size=None for other cameras")
        meshes = Meshes(verts=[vertices.to(self.device)], faces=[self.faces])
        fragments = self.rasterizer(meshes, cameras=camera)
        image = self.shader(fragments, meshes)[..., 3]
        mask = image[0]
        if restore_resolution and self.raster_size != self.image_size:
            mask = torch.nn.functional.interpolate(
                mask[None, None], size=self.image_size, mode="bilinear", align_corners=False,
            )[0, 0]
        return mask


class OfficialSATHMRGeometryEstimator:
    """Run SAT-HMR in an isolated subprocess and retain native camera coordinates.

    No demo inputs/results in the official checkout are overwritten. A separate
    Python executable can isolate SAT-HMR's dependencies from the other models.
    """

    def __init__(self, sat_hmr_root: str | Path, *, python_executable: str | None = None,
                 device: str = "cuda", threshold: float = 0.3, config: str = "demo") -> None:
        self.root = _require_path(Path(sat_hmr_root).resolve(), "SAT-HMR checkout")
        _require_path(self.root / "main.py", "SAT-HMR main.py")
        self.python = python_executable or sys.executable
        self.device = torch.device(device)
        if not 0 < threshold < 1:
            raise ValueError("SAT-HMR threshold must be between 0 and 1")
        self.threshold = threshold
        self.config = config

    def estimate(self, image: Any) -> list[HumanEstimate]:
        rgb = load_rgb(image)
        with tempfile.TemporaryDirectory(prefix="decon-sat-") as work:
            work = Path(work)
            inputs, outputs = work / "inputs", work / "outputs"
            inputs.mkdir()
            outputs.mkdir()
            rgb.save(inputs / "input.png")
            runner = Path(__file__).with_name("sat_hmr_runner.py")
            subprocess.run(
                [self.python, "-c",
                 "import runpy, sys; runpy.run_path(sys.argv.pop(1), run_name='__main__')",
                 str(runner), "--input-dir", str(inputs),
                 "--output-dir", str(outputs), "--threshold", str(self.threshold),
                 "--config", self.config],
                cwd=self.root, env=os.environ.copy(), check=True,
            )
            return self.read_predictions(outputs, (rgb.height, rgb.width))

    def read_predictions(self, output_dir: Path, image_size: tuple[int, int]) -> list[HumanEstimate]:
        estimates = []
        for path in sorted(output_dir.glob("person_*.npz")):
            with np.load(path, allow_pickle=False) as data:
                def tensor(key):
                    return torch.as_tensor(data[key].copy(), device=self.device, dtype=torch.float32)
                camera = PinholeCamera.from_resized(
                    tensor("intrinsics"), tuple(int(x) for x in data["resized_size"]), image_size,
                )
                metadata = {"person_id": path.stem, "coordinate_system": "opencv_camera"}
                if 'detection_box' in data:
                    box = tensor('detection_box')
                    rh, rw = (int(x) for x in data['resized_size'])
                    h, w = image_size
                    box *= box.new_tensor([w/rw, h/rh, w/rw, h/rh])
                    margin = (box[2:]-box[:2]) * 0.025
                    box = torch.cat((box[:2]-margin, box[2:]+margin))
                    box = torch.minimum(box.clamp(min=0), box.new_tensor([w-1,h-1,w-1,h-1]))
                    metadata['detection_box'] = box.cpu().tolist()
                estimates.append(HumanEstimate(
                    vertices=tensor("vertices"), mask=torch.zeros(image_size, dtype=torch.bool, device=self.device),
                    camera=camera, pose=tensor("pose"), shape=tensor("shape"),
                    translation=tensor("translation"),
                    metadata=metadata,
                ))
        return estimates
