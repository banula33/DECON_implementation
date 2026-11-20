import pytest
import torch

from decon.ggdo import GGDOConfig, optimize_geometry


def gaussian_renderer(vertices):
    y, x = torch.meshgrid(torch.arange(24), torch.arange(24), indexing='ij')
    center = vertices[:, :2].mean(0)
    spread = (vertices[1, 0] - vertices[0, 0]).square() + 2
    return torch.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / spread)


def body(pose, shape, translation):
    width = 1 + shape[0]
    tilt = pose.reshape(-1)[0]
    zero = width * 0
    return torch.stack([torch.stack([-width, -tilt, zero]), torch.stack([width, tilt, zero])]) + translation


def test_translation_stage_reduces_mask_error():
    vertices = torch.tensor([[5., 8., 0.], [7., 8., 0.]])
    target = gaussian_renderer(vertices + torch.tensor([2., 1., 0.]))
    result = optimize_geometry(vertices, target, gaussian_renderer,
                               config=GGDOConfig(translation_steps=100, pose_shape_steps=0, translation_lr=0.05))
    assert result.losses[-1] < result.losses[0] * 0.05
    assert torch.allclose(result.vertices[1] - result.vertices[0], vertices[1] - vertices[0])


def test_pose_shape_stage_uses_body_model_and_preserves_inputs():
    pose, shape, trans = torch.zeros(24, 3), torch.zeros(10), torch.tensor([12., 12., 0.])
    original_shape = shape.clone()
    target_shape = shape.clone()
    target_shape[0] = 0.8
    vertices = body(pose, shape, trans)
    target = gaussian_renderer(body(pose, target_shape, trans))
    result = optimize_geometry(vertices, target, gaussian_renderer, body_model=body,
                               pose=pose, shape=shape, translation=trans,
                               config=GGDOConfig(translation_steps=0, pose_shape_steps=100, pose_shape_lr=0.03))
    assert result.losses[-1] < result.losses[0] * 0.01
    assert result.shape[0] > 0.7
    assert torch.equal(shape, original_shape)
    assert torch.allclose(result.vertices, body(result.pose, result.shape, result.translation))
    assert not result.vertices.requires_grad


def test_stage_one_freezes_pose_and_shape():
    pose, shape, trans = torch.zeros(24, 3), torch.zeros(10), torch.tensor([12., 12., 0.])
    vertices = body(pose, shape, trans)
    result = optimize_geometry(vertices, gaussian_renderer(vertices + 0.2), gaussian_renderer,
                               body_model=body, pose=pose, shape=shape, translation=trans,
                               config=GGDOConfig(translation_steps=10, pose_shape_steps=0))
    assert torch.equal(result.pose, pose)
    assert torch.equal(result.shape, shape)
    assert not torch.equal(result.translation, trans)


def test_requires_parametric_model_for_second_stage():
    with pytest.raises(ValueError, match='body_model'):
        optimize_geometry(torch.ones(2, 3), torch.ones(24, 24), gaussian_renderer)


@pytest.mark.parametrize('mask', [torch.zeros(24, 24), torch.full((24, 24), float('nan'))])
def test_invalid_masks_fail_early(mask):
    with pytest.raises(ValueError, match='segmented_mask'):
        optimize_geometry(torch.ones(2, 3), mask, gaussian_renderer)


def test_rejects_mismatched_coordinate_frames():
    pose, shape, trans = torch.zeros(24, 3), torch.zeros(10), torch.tensor([12., 12., 0.])
    with pytest.raises(ValueError, match='reproduce'):
        optimize_geometry(body(pose, shape, trans) + 1, torch.ones(24, 24), gaussian_renderer,
                          body_model=body, pose=pose, shape=shape, translation=trans)


def test_released_profile_keeps_depth_and_stage_two_translation_fixed():
    from dataclasses import replace
    pose, shape, trans = torch.zeros(24, 3), torch.zeros(10), torch.tensor([10., 10., 2.])
    vertices = body(pose, shape, trans)
    def depth_renderer(v):
        return gaussian_renderer(v + torch.stack([v[:, 2], v[:, 2], v[:, 2]], dim=1))
    target = depth_renderer(vertices + torch.tensor([1., 1., 1.]))
    first = optimize_geometry(vertices, target, depth_renderer, body_model=body,
        pose=pose, shape=shape, translation=trans,
        config=replace(GGDOConfig.released_code(),translation_steps=20,pose_shape_steps=0))
    both = optimize_geometry(vertices, target, depth_renderer, body_model=body,
        pose=pose, shape=shape, translation=trans,
        config=replace(GGDOConfig.released_code(),translation_steps=20,pose_shape_steps=20))
    assert first.translation[2] == trans[2]
    assert not torch.equal(first.translation[:2],trans[:2])
    assert torch.equal(first.translation,both.translation)
    assert torch.allclose(both.vertices,body(both.pose,both.shape,both.translation))


def test_released_objective_scale_matches_code():
    from decon.ggdo import _mask_loss
    pred=torch.tensor([[0.,0.5],[1.,1.]])
    target=torch.tensor([[1.,0.],[1.,0.]])
    config=GGDOConfig.released_code()
    assert torch.allclose(config.mask_weight*_mask_loss(pred,target,config.mask_metric),
                          torch.abs(pred*100-target*100).mean())
