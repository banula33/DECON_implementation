"""Counterexamples for assigning disconnected visible parts to people."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from decon.camera import PinholeCamera
from decon.segmentation import (
    exclusive_masks,
    geometry_points,
    person_prompts,
    validate_corrections,
    visible_geometry,
)


def projected_estimate(pixels, depths, image_size):
    """Create camera-space vertices whose projections are the specified pixels."""
    xyz = torch.tensor([[x * z, y * z, z] for (x, y), z in zip(pixels, depths)])
    return SimpleNamespace(
        vertices=xyz,
        camera=PinholeCamera(torch.eye(3), image_size),
    )


def test_tiny_disconnected_palm_gets_prompt_without_connecting_hidden_arm():
    visible = np.zeros((24, 30), dtype=bool)
    visible[3:21, 2:14] = True  # A large visible torso.
    visible[7, 25] = True  # Only a single palm pixel survives an occluder.
    depth = np.where(visible, 2.0, np.inf)

    points = geometry_points(None, depth, visible)
    pixels = points.astype(int)

    assert [25, 7] in points.tolist()
    assert any(x < 14 for x, _ in points)
    assert visible[pixels[:, 1], pixels[:, 0]].all()
    assert not any(14 <= x < 25 for x, _ in points)


def test_connected_person_gets_prompts_on_both_hands_not_only_torso():
    visible = np.zeros((12, 18), dtype=bool)
    visible[2:10, 2:16] = True
    depth = np.where(visible, 2.0, np.inf)
    estimate = projected_estimate(
        [(7, 5), (8, 6), (2, 5), (3, 5), (14, 5), (15, 5), (12, 6)],
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0],
        visible.shape,
    )
    # Last vertex is another part behind the front surface of this same body.
    parts = torch.tensor([0, 0, 20, 20, 21, 21, 22])

    points = geometry_points(estimate, depth, visible, vertex_parts=parts)

    assert any(x <= 3 for x, _ in points)
    assert any(x >= 14 for x, _ in points)
    assert [12, 6] not in points.tolist()
    assert visible[points[:, 1].astype(int), points[:, 0].astype(int)].all()


def test_empty_visible_geometry_has_no_automatic_positive_points():
    visible = np.zeros((4, 5), dtype=bool)
    points = geometry_points(None, np.full(visible.shape, np.inf), visible)
    assert points.shape == (0, 2)


def test_visibility_follows_local_crossing_depth_and_omits_near_ties():
    first = np.array([[1.0, 4.0, 2.0, np.inf, 1.5]])
    second = np.array([[2.0, 3.0, 2.005, np.inf, np.inf]])

    visible = visible_geometry([first, second])

    assert visible[0].tolist() == [[True, False, False, False, True]]
    assert visible[1].tolist() == [[False, True, False, False, False]]
    reversed_visibility = visible_geometry([second, first])
    assert np.array_equal(reversed_visibility[1], visible[0])
    assert np.array_equal(reversed_visibility[0], visible[1])


def test_exclusive_masks_keep_disconnected_palm_and_never_fill_hidden_geometry():
    first = np.full((8, 12), -3.0)
    second = np.full(first.shape, -2.0)
    first[1:6, 1:4] = 3.0
    first[2, 10] = 2.0  # Disconnected palm of the first person.
    second[2:7, 5:8] = 4.0
    # A complete projected body extends through regions where SAM sees no person.
    depths = [np.full(first.shape, 1.0), np.full(first.shape, 2.0)]

    masks = exclusive_masks([first, second], depths)

    assert masks[0][2, 10]
    assert not (masks[0] & masks[1]).any()
    assert np.array_equal(masks[0] | masks[1], (first > 0) | (second > 0))
    assert not masks[0][2, 4]  # Never manufacture the missing connecting arm.


def test_overlap_uses_crossing_surface_depth_then_logits_for_ties_and_background():
    first = np.array([[2.0, 8.0, 1.0, 5.0, -1.0]])
    second = np.array([[8.0, 2.0, 6.0, 2.0, -2.0]])
    first_depth = np.array([[1.0, 4.0, 2.0, np.inf, 1.0]])
    second_depth = np.array([[2.0, 3.0, 2.005, np.inf, 2.0]])

    masks = exclusive_masks([first, second], [first_depth, second_depth])

    assert masks[0].tolist() == [[True, False, False, True, False]]
    assert masks[1].tolist() == [[False, True, True, False, False]]
    # Person ordering must not become a depth shortcut.
    reversed_masks = exclusive_masks([second, first], [second_depth, first_depth])
    assert np.array_equal(masks[0], reversed_masks[1])
    assert np.array_equal(masks[1], reversed_masks[0])


def test_geometry_cannot_steal_a_pixel_that_only_another_person_claims():
    logits = [np.array([[-1.0, 2.0]]), np.array([[1.0, -1.0]])]
    depths = [np.array([[1.0, 1.0]]), np.array([[3.0, 3.0]])]
    first, second = exclusive_masks(logits, depths)
    assert first.tolist() == [[False, True]]
    assert second.tolist() == [[True, False]]


def test_without_depth_overlapping_masks_use_logits_and_respect_threshold():
    logits = [np.array([[0.2, 2.0, 4.0]]), np.array([[0.1, 3.0, 1.0]])]
    first, second = exclusive_masks(logits, threshold=0.5)
    assert first.tolist() == [[False, False, True]]
    assert second.tolist() == [[False, True, False]]


@pytest.mark.parametrize("data", [
    {"2": {"positive": [[1, 1]]}},
    {"-1": {"positive": [[1, 1]]}},
    {"person_000": {"positive": [[1, 1]]}},
])
def test_corrections_reject_unknown_person_ids(data):
    with pytest.raises(ValueError, match="Unknown person index"):
        validate_corrections(data, 2, (20, 30))


@pytest.mark.parametrize("point", [
    [-1, 3], [30, 3], [3, 20], [3, float("nan")], [float("inf"), 3],
])
def test_corrections_reject_points_outside_original_image(point):
    with pytest.raises(ValueError, match="inside the original image"):
        validate_corrections({"0": {"positive": [point]}}, 2, (20, 30))


@pytest.mark.parametrize("data", [
    {"0": {"positive": [[8, 5]], "negative": [[8, 5]]}},
    {"0": {"positive": [[8, 5]]}, "1": {"positive": [[8, 5]]}},
    {"0": {"positive": [[8.1, 5.1]]}, "1": {"positive": [[8.2, 5.2]]}},
])
def test_corrections_reject_conflicting_ownership_of_same_pixel(data):
    with pytest.raises(ValueError, match="Contradictory"):
        validate_corrections(data, 2, (20, 30))


def test_positive_palm_correction_overrides_geometry_and_excludes_other_people():
    corrections = validate_corrections(
        {"0": {"positive": [[15, 5]], "negative": [[5, 5]]}},
        2, (20, 30),
    )
    automatic = [
        np.array([[5, 5], [14, 5], [3, 15]], dtype=np.float32),
        np.array([[15, 5], [25, 15]], dtype=np.float32),
    ]

    points, labels, sources = person_prompts(0, automatic, corrections)
    assigned = {tuple(point): label for point, label in zip(points, labels)}
    assert assigned[(15, 5)] == 1
    assert assigned[(5, 5)] == 0
    assert (14, 5) not in assigned  # Nearby contradictory geometry was suppressed.
    assert assigned[(3, 15)] == 1
    assert assigned[(25, 15)] == 0
    assert sources[list(map(tuple, points)).index((15, 5))] == "correction"

    other_points, other_labels, _ = person_prompts(1, automatic, corrections)
    other_assigned = {tuple(point): label for point, label in zip(other_points, other_labels)}
    assert other_assigned[(15, 5)] == 0
    assert other_assigned[(25, 15)] == 1
    assert not any(label == 1 and np.linalg.norm(point - [15, 5]) <= 6
                   for point, label in zip(other_points, other_labels))
