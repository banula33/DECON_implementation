"""Reject incomplete weights before materializing a large meta-initialized model."""
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('accelerate')
safetensors = pytest.importorskip('safetensors.torch')

from decon.powerpaint_loader import build_v2_controller


@pytest.mark.parametrize('problem', ['missing', 'shape'])
def test_direct_loader_rejects_incomplete_or_mismatched_checkpoint(tmp_path, problem):
    folder = tmp_path / 'PowerPaint_Brushnet'
    folder.mkdir()
    state = {'weight': torch.zeros(2, 2), 'bias': torch.zeros(2)}
    if problem == 'missing':
        del state['bias']
    else:
        state['weight'] = torch.zeros(3, 2)
    safetensors.save_file(state, str(folder / 'diffusion_pytorch_model.safetensors'))
    module = SimpleNamespace(
        UNet2DConditionModel=SimpleNamespace(
            load_config=lambda *args, **kwargs: {},
            from_config=lambda config: torch.nn.Linear(2, 2),
        ),
        BrushNetModel=SimpleNamespace(
            from_unet=lambda *args, **kwargs: torch.nn.Linear(2, 2),
        ),
    )
    with pytest.raises(ValueError, match='every model tensor|shape mismatch'):
        build_v2_controller(module, tmp_path, torch.float16)
