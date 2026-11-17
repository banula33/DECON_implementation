"""Regression scenarios for person identity across occluded body parts.

The fake predictor models a box-only missed hand. This verifies prompt/data
flow, not pretrained SAM quality; real-image acceptance is still required.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from decon.artifacts import save_segmentations
from decon.official_adapters import OfficialSAM2Segmenter
from decon.pipeline import HumanDecouplingPipeline
from decon.types import HumanEstimate


def test_disconnected_palm_recovered_without_marking_hidden_forearm(tmp_path):
    h, w = 16, 24
    torso = np.zeros((h, w), dtype=bool)
    torso[3:13, 2:8] = True
    palm = np.zeros_like(torso)
    palm[5:7, 18:20] = True
    other = np.zeros_like(torso)
    other[2:14, 8:18] = True
    hidden_arm = np.zeros_like(torso)
    hidden_arm[5:7, 8:18] = True
    depths = [np.where(torso | palm | hidden_arm, 4.0, np.inf),
              np.where(other, 2.0, np.inf)]
    estimates = [SimpleNamespace(vertices=torch.full((1, 3), float(i)), camera=None, metadata=None)
                 for i in range(2)]
    boxes = [torch.tensor([1., 1., 21., 14.]), torch.tensor([8., 1., 21., 14.])]
    calls = []
    images = []

    def predict(**kw):
        calls.append(kw)
        i = 0 if kw['box'][0] < 8 else 1
        mask = torso.copy() if i == 0 else other.copy()
        assert kw['return_logits'] and not kw['multimask_output']
        if 'point_coords' in kw:
            assert kw['mask_input'].shape == (1, 8, 8)
            coords = np.rint(kw['point_coords']).astype(int)
            pos = coords[kw['point_labels'] == 1]
            neg = coords[kw['point_labels'] == 0]
            if i == 0:
                assert not hidden_arm[pos[:, 1], pos[:, 0]].any()
                assert other[neg[:, 1], neg[:, 0]].any()
                if palm[pos[:, 1], pos[:, 0]].any():
                    mask |= palm
        return np.where(mask[None], 5., -5.), np.array([0.9]), np.ones((1, 8, 8))

    adapter = OfficialSAM2Segmenter(tmp_path, device='cpu')
    adapter._predictor = SimpleNamespace(set_image=lambda rgb: images.append(rgb), predict=predict)
    renderer = SimpleNamespace(render_depth=lambda v, _: torch.from_numpy(depths[int(v[0, 0])]))
    masks = adapter.segment_people(Image.new('RGB', (w, h)), estimates, boxes, renderer=renderer)
    assert len(images) == 1 and len(calls) == 4
    assert torch.equal(masks[0], torch.from_numpy(torso | palm))
    assert not masks[0][hidden_arm].any()
    assert torch.equal(masks[1], torch.from_numpy(other))
    assert not (masks[0] & masks[1]).any()
    assert estimates[0].metadata['segmentation']['added_pixels_before_ownership'] == palm.sum()


def test_correction_outside_box_is_sent_and_is_negative_for_other_person(tmp_path):
    calls = []
    adapter = OfficialSAM2Segmenter(tmp_path, guidance='box',
                                   prompt_corrections={'0': {'positive': [[18, 5]], 'negative': [[2, 2]]}})
    def predict(**kw):
        calls.append(kw)
        return np.ones((1, 12, 24)), np.array([0.8]), np.zeros((1, 8, 8))
    adapter._predictor = SimpleNamespace(set_image=lambda _: None, predict=predict)
    estimates = [SimpleNamespace(metadata=None) for _ in range(2)]
    boxes = [torch.tensor([1., 1., 10., 10.]), torch.tensor([1., 1., 23., 10.])]
    adapter.segment_people(Image.new('RGB', (24, 12)), estimates, boxes, renderer=None)
    assert calls[0]['box'][2] >= 18
    assert [18, 5] in calls[1]['point_coords'][calls[1]['point_labels'] == 1].tolist()
    assert [18, 5] in calls[3]['point_coords'][calls[3]['point_labels'] == 0].tolist()
    # A predictor that ignores an explicit negative must be flagged for review.
    info = estimates[0].metadata['segmentation']
    assert info['needs_review']
    assert any(p['xy'] == [2, 2] for p in info['unmet_points'])


def test_invalid_corrections_fail_before_loading_models(tmp_path):
    adapter = OfficialSAM2Segmenter(tmp_path, prompt_corrections={'4': {'positive': [[1, 1]]}})
    adapter._load = lambda: pytest.fail('must reject invalid input before loading SAM')
    with pytest.raises(ValueError, match='Unknown person'):
        adapter.segment_people(Image.new('RGB', (8, 8)), [SimpleNamespace()], [torch.zeros(4)], renderer=None)


def test_geometry_mode_requires_depth_renderer(tmp_path):
    adapter = OfficialSAM2Segmenter(tmp_path)
    with pytest.raises(ValueError, match='render_depth'):
        adapter.segment_people(Image.new('RGB', (8, 8)), [SimpleNamespace()], [torch.zeros(4)], renderer=None)


def test_segmentation_only_uses_scene_interface_and_exports_without_ggdo(tmp_path):
    h, w = 8, 12
    camera = SimpleNamespace(bounding_box=lambda _: torch.tensor([0., 0., 11., 7.]))
    estimates = [HumanEstimate(torch.ones(3, 3), torch.zeros(h, w), camera) for _ in range(2)]
    masks = [torch.zeros(h, w, dtype=torch.bool) for _ in estimates]
    masks[0][1:4, 1:4] = True
    masks[0][1, 8] = True  # tiny, disconnected visible hand
    masks[1][4:7, 6:10] = True
    def segment_people(rgb, all_estimates, boxes, **kwargs):
        assert all_estimates == estimates and len(boxes) == 2
        assert kwargs['vertex_parts'] is not None
        return masks
    fail = lambda *args: pytest.fail('segmentation-only must not fit, render silhouettes or inpaint')
    pipeline = HumanDecouplingPipeline(
        SimpleNamespace(estimate=lambda _: estimates),
        SimpleNamespace(segment=fail, segment_people=segment_people),
        SimpleNamespace(render_mask=fail), SimpleNamespace(inpaint=fail),
        body_model=SimpleNamespace(vertex_parts=torch.zeros(3)),
    )
    rgb = Image.new('RGB', (w, h), color=(40, 50, 60))
    result = pipeline.segment(rgb)
    out = save_segmentations(rgb, result, tmp_path / 'results')
    assert json.loads((out / 'manifest.json').read_text())['stage'] == 'human_segmentation'
    for i, mask in enumerate(masks):
        folder = out / f'person_{i:03d}'
        assert np.array_equal(np.asarray(Image.open(folder / 'segmentation.png')) > 0, mask.numpy())
        assert (folder / 'segmentation_overlay.png').exists()
        assert (folder / 'prompts.png').exists()
        isolated = np.asarray(Image.open(folder / 'segmented.png'))
        assert (isolated[mask] == [40, 50, 60]).all()
        assert (isolated[~mask] == 255).all()
        assert not (folder / 'completed.png').exists()
    with pytest.raises(FileExistsError):
        save_segmentations(rgb, result, out)
