"""Preflight checks for the official DECON human-decoupling dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def inspect_official_setup(official_root: str | Path) -> dict[str, Any]:
    root = Path(official_root)
    human = root / "Human_SMPL_Estimation"
    segmentation = root / "Img_Segmentation"
    inpainting = root / "Img_Inpainting"
    sat_weights = human / "weights" / "sat_hmr"
    smpl_weights = human / "weights" / "smpl_data" / "smpl"
    sam_checkpoints = segmentation / "checkpoints"
    powerpaint_checkpoints = inpainting / "checkpoints"

    def files(directory: Path) -> list[str]:
        return sorted(str(path.relative_to(root)) for path in directory.rglob("*") if path.is_file()) if directory.exists() else []

    return {
        "official_root": str(root),
        "sat_hmr_checkout": human.exists(),
        "sat_hmr_checkpoints": files(sat_weights),
        "smpl_checkpoints": files(smpl_weights),
        "sam2_checkout": segmentation.exists(),
        "sam2_checkpoints": files(sam_checkpoints),
        "powerpaint_checkout": inpainting.exists(),
        "powerpaint_checkpoints": files(powerpaint_checkpoints),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def format_preflight(report: dict[str, Any]) -> str:
    lines = [f"Official root: {report['official_root']}"]
    lines.append(f"PyTorch: {report['torch_version']} (CUDA: {report['cuda_available']})")
    lines.append(f"SAT-HMR checkout: {report['sat_hmr_checkout']}")
    lines.append(f"SAT-HMR files: {len(report['sat_hmr_checkpoints'])}")
    lines.append(f"SMPL files: {len(report['smpl_checkpoints'])}")
    lines.append(f"SAM 2 checkout: {report['sam2_checkout']}")
    lines.append(f"SAM 2 checkpoint files: {len(report['sam2_checkpoints'])}")
    lines.append(f"PowerPaint checkout: {report['powerpaint_checkout']}")
    lines.append(f"PowerPaint checkpoint files: {len(report['powerpaint_checkpoints'])}")
    if not report["cuda_available"]:
        lines.append("WARNING: official PyTorch3D and PowerPaint execution requires CUDA.")
    return "\n".join(lines)
