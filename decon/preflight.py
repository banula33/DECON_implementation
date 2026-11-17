"""Report dependency and checkpoint availability without loading any models."""

from importlib.util import find_spec
from pathlib import Path
from typing import Any

import torch


def inspect_official_setup(official_root: str | Path) -> dict[str, Any]:
    root = Path(official_root)
    human = root / 'Human_SMPL_Estimation'
    segmentation = root / 'Img_Segmentation'
    inpainting = root / 'Img_Inpainting'

    def files(directory):
        suffixes = {'.pth', '.pt', '.pkl', '.npz', '.npy', '.bin', '.safetensors'}
        return sorted(str(p.relative_to(root)) for p in directory.rglob('*')
                      if p.is_file() and p.suffix in suffixes) if directory.exists() else []

    required_smpl = ['SMPL_NEUTRAL.pkl', 'SMPL_MALE.pkl', 'SMPL_FEMALE.pkl',
                     'body_verts_smpl.npy', 'J_regressor_h36m_correct.npy', 'smpl_mean_params.npz']
    smpl = human / 'weights/smpl_data/smpl'
    return {
        'official_root': str(root),
        'sat_hmr_checkout': (human / 'main.py').is_file(),
        'sat_hmr_checkpoints': files(human / 'weights/sat_hmr'),
        'smpl_checkpoints': files(smpl),
        'missing_smpl_files': [name for name in required_smpl if not (smpl / name).is_file()],
        'sam2_checkout': (segmentation / 'sam2').is_dir(),
        'sam2_checkpoints': files(segmentation / 'checkpoints'),
        'powerpaint_checkout': (inpainting / 'app.py').is_file(),
        'powerpaint_checkpoints': files(inpainting / 'checkpoints'),
        'cuda_available': torch.cuda.is_available(),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'runtime_packages': {name: find_spec(name) is not None for name in ['smplx', 'pytorch3d', 'numpy', 'PIL']},
    }


def format_preflight(report: dict[str, Any]) -> str:
    lines = [f"Official root: {report['official_root']}",
             f"PyTorch: {report['torch_version']} (CUDA available: {report['cuda_available']})"]
    for name, label in [('sat_hmr', 'SAT-HMR'), ('sam2', 'SAM 2'), ('powerpaint', 'PowerPaint')]:
        lines.append(f"{label} checkout: {report[name + '_checkout']}; asset files: {len(report[name + '_checkpoints'])}")
    lines.append('Missing SMPL/support files: ' + (', '.join(report['missing_smpl_files']) or 'none'))
    lines.append('Runtime packages: ' + ', '.join(f'{name}={present}' for name, present in report['runtime_packages'].items()))
    if not report['cuda_available']:
        lines.append('BLOCKED: the official PowerPaint controller requires a CUDA GPU.')
    lines.append('Presence checks only: upstream environments and base-model caches still require a real inference run.')
    return '\n'.join(lines)
