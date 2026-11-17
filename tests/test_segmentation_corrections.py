"""Regression coverage for correcting visible disconnected body regions."""

import numpy as np
import pytest

from decon.segmentation import exclusive_masks, validate_corrections


def palm_scene():
    shape = (12, 20)
    scores = [np.full(shape, -4.0, dtype=np.float32) for _ in range(2)]
    # The rear person's visible body and palm are separated by hidden anatomy.
    scores[1][2:10, 1:5] = 6.0
    scores[1][4:7, 11:14] = 6.0
    scores[0][2:10, 10:18] = 4.0
    depths = [np.full(shape, np.inf, dtype=np.float32) for _ in range(2)]
    depths[0][scores[0] > 0] = 1.0
    depths[1][scores[1] > 0] = 2.0
    return scores, depths


def test_palm_correction_overrides_incorrect_geometry_without_filling_hidden_arm():
    scores, depths = palm_scene()
    corrections = validate_corrections({'1': {'positive': [[12, 5]]}}, 2, scores[0].shape)
    masks = exclusive_masks(scores, depths, corrections=corrections)
    assert masks[1][4:7, 11:14].all()
    assert masks[1][2:10, 1:5].all()
    assert not masks[1][:, 5:10].any()
    assert not (masks[0] & masks[1]).any()
    for mask, score in zip(masks, scores):
        assert not (mask & (score <= 0)).any()


def test_correction_does_not_change_another_disconnected_ambiguous_region():
    scores, depths = palm_scene()
    scores[1][8:10, 11:14] = 6.0
    depths[1][8:10, 11:14] = 2.0
    corrections = validate_corrections({'1': {'positive': [[12, 5]]}}, 2, scores[0].shape)
    masks = exclusive_masks(scores, depths, corrections=corrections)
    assert masks[1][4:7, 11:14].all()
    assert masks[0][8:10, 11:14].all()


def test_negative_region_hint_prevents_whole_region_takeover():
    scores, depths = palm_scene()
    corrections = validate_corrections(
        {'1': {'positive': [[11, 5]], 'negative': [[13, 5]]}}, 2, scores[0].shape,
    )
    corrected = exclusive_masks(scores, depths, corrections=corrections)
    # Both instructions concern one shared region, so one positive click must
    # not reassign the entire region in conflict with the negative instruction.
    assert not corrected[1][4:7, 11:14].all()


def test_correction_never_invents_foreground_rejected_by_sam():
    scores, depths = palm_scene()
    corrections = validate_corrections({'1': {'positive': [[7, 5]]}}, 2, scores[0].shape)
    masks = exclusive_masks(scores, depths, corrections=corrections)
    assert not any(mask[5, 7] for mask in masks)
    for mask, score in zip(masks, scores):
        assert not (mask & (score <= 0)).any()


def test_missing_mesh_surface_does_not_disqualify_a_sam_claimant():
    # An unmodeled palm/clothing region may sit outside its person's SMPL
    # projection. Even two competing mesh surfaces cannot establish its depth.
    scores = [np.full((2, 2), value, dtype=np.float32) for value in (4, 3, 8)]
    depths = [np.full((2, 2), value, dtype=np.float32) for value in (1, 2, np.inf)]
    masks = exclusive_masks(scores, depths)
    assert masks[2].all()
    assert not masks[0].any() and not masks[1].any()


@pytest.mark.parametrize('data', [
    [1],
    {'0': {'positive': [[], []]}},
    {'0': {'positive': [[1, 2, 3]]}},
    {'0': {'positive': [[float('nan'), 2]]}},
    {'0': {'positive': [[20, 2]]}},
    {'0': {'positive': [[2, -1]]}},
    {'2': {'positive': [[1, 2]]}},
    {'0': {'positive': [[1, 2]], 'negative': [[1, 2]]}},
    {'0': {'positive': [[1, 2]]}, '1': {'positive': [[1, 2]]}},
])
def test_invalid_or_contradictory_corrections_are_rejected(data):
    with pytest.raises(ValueError):
        validate_corrections(data, 2, (12, 20))
