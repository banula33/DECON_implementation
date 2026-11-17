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
