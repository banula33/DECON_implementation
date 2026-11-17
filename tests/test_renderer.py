"""Optional tests against the real PyTorch3D rasterizer (no model assets needed)."""
import pytest
import torch

pytest.importorskip('pytorch3d')

from decon.camera import PinholeCamera
from decon.ggdo import GGDOConfig, optimize_geometry
from decon.official_adapters import PyTorch3DSilhouetteRenderer


def scene():
    camera = PinholeCamera(torch.tensor([[40., 0, 24.], [0, 40., 16.], [0, 0, 1.]]), (32, 48))
    vertices = torch.tensor([[-.4, -.3, 2.], [.4, -.3, 2.], [.4, .3, 2.], [-.4, .3, 2.]])
    renderer = PyTorch3DSilhouetteRenderer(torch.tensor([[0, 1, 2], [0, 2, 3]]), camera.image_size, device='cpu')
    return camera, vertices, renderer


def test_opencv_projection_matches_pytorch3d_screen():
    camera, vertices, _ = scene()
    converted = camera.to_pytorch3d(torch.device('cpu'))
    actual = converted.transform_points_screen(vertices[None], image_size=(camera.image_size,))[0, :, :2]
    assert torch.allclose(actual, camera.project(vertices), atol=1e-4)


def test_real_silhouette_renderer_has_translation_gradients():
    camera, vertices, renderer = scene()
    translation = torch.tensor([.025, .01, .02], requires_grad=True)
    mask = renderer.render_mask(vertices + translation, camera)
    assert mask.shape == (32, 48)
    # Sample inside one triangle, away from the softened shared diagonal.
    assert mask[13, 24] > 0.99
    assert mask[0, 0] < 0.01
    target = renderer.render_mask(vertices + torch.tensor([.1, .03, 0.]), camera).detach()
    ((mask - target) ** 2).mean().backward()
    assert torch.isfinite(translation.grad).all()
    assert translation.grad.norm() > 0


def test_real_renderer_translation_fit_improves_alignment():
    camera, vertices, renderer = scene()
    render = lambda v: renderer.render_mask(v, camera)
    target = render(vertices + torch.tensor([.12, .06, 0.])).detach()
    result = optimize_geometry(vertices, target, render,
                               config=GGDOConfig(translation_steps=25, pose_shape_steps=0, translation_lr=.01))
    assert ((result.geometry_mask - target) ** 2).mean() < result.losses[0] * .5


def test_bounded_render_preserves_projection_resolution_and_gradients():
    camera, vertices, reference = scene()
    renderer = PyTorch3DSilhouetteRenderer(
        torch.tensor([[0, 1, 2], [0, 2, 3]]), camera.image_size,
        device='cpu', max_render_size=24,
    )
    translation = torch.tensor([.025, .01, .02], requires_grad=True)
    mask = renderer.render_mask(vertices + translation, camera)
    assert renderer.raster_size == (16, 24)
    assert mask.shape == camera.image_size
    expected = reference.render_mask(vertices + translation.detach(), camera).detach()
    foreground = mask > .5
    target = expected > .5
    assert (foreground & target).sum() / (foreground | target).sum() > .8
    ((mask - expected) ** 2).mean().backward()
    assert torch.isfinite(translation.grad).all() and translation.grad.norm() > 0


def test_bounded_render_rejects_camera_size_mismatch():
    camera, vertices, renderer = scene()
    wrong_camera = PinholeCamera(camera.intrinsics, (64, 96))
    with pytest.raises(ValueError, match='image sizes differ'):
        renderer.render_mask(vertices, wrong_camera)


@pytest.mark.parametrize('max_render_size', [None, 24])
def test_depth_is_camera_z_with_infinite_background(max_render_size):
    camera, vertices, _ = scene()
    renderer = PyTorch3DSilhouetteRenderer(
        torch.tensor([[0, 1, 2], [0, 2, 3]]), camera.image_size,
        device='cpu', max_render_size=max_render_size,
    )
    depth = renderer.render_depth(vertices, camera)
    assert depth.shape == camera.image_size
    assert depth[13, 24] == pytest.approx(2.0)
    assert torch.isposinf(depth[0, 0])
    assert not torch.isnan(depth).any()
    assert torch.allclose(depth[torch.isfinite(depth)], torch.full_like(depth[torch.isfinite(depth)], 2.0))


def test_depth_uses_frontmost_surface_at_each_pixel():
    camera, vertices, _ = scene()
    # Two rectangles with the same screen projection but different depths.
    # Put the far rectangle first so face order cannot mimic visibility.
    far = vertices * 2
    faces = torch.tensor([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    renderer = PyTorch3DSilhouetteRenderer(faces, camera.image_size, device='cpu')
    depth = renderer.render_depth(torch.cat((far, vertices)), camera)
    assert depth[13, 24] == pytest.approx(2.0)
    finite = depth[torch.isfinite(depth)]
    assert finite.numel() > 0 and torch.allclose(finite, torch.full_like(finite, 2.0))


def test_depth_rejects_camera_size_mismatch():
    camera, vertices, renderer = scene()
    wrong_camera = PinholeCamera(camera.intrinsics, (64, 96))
    with pytest.raises(ValueError, match='image sizes differ'):
        renderer.render_depth(vertices, wrong_camera)
