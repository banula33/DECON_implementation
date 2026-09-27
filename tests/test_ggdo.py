import torch

from decon.ggdo import GGDOConfig, optimize_geometry


def gaussian_renderer(vertices: torch.Tensor) -> torch.Tensor:
    height, width = 24, 24
    y, x = torch.meshgrid(
        torch.arange(height, dtype=vertices.dtype),
        torch.arange(width, dtype=vertices.dtype),
        indexing="ij",
    )
    center = vertices[:, :2].mean(dim=0)
    distance = (x - center[0]) ** 2 + (y - center[1]) ** 2
    return torch.exp(-distance / 8.0)


def test_translation_stage_reduces_mask_error():
    vertices = torch.tensor([[5.0, 8.0, 0.0], [7.0, 8.0, 0.0]])
    target_vertices = vertices + torch.tensor([6.0, 3.0, 0.0])
    target_mask = gaussian_renderer(target_vertices)
    segmented_mask = target_mask > 0.2
    initial_loss = torch.mean(
        (gaussian_renderer(vertices) - segmented_mask.float()) ** 2
    )

    result = optimize_geometry(
        vertices,
        segmented_mask,
        gaussian_renderer,
        config=GGDOConfig(translation_steps=80, pose_shape_steps=20),
    )

    assert result.vertices.shape == vertices.shape
    assert result.geometry_mask.shape == target_mask.shape
    assert result.losses[-1] < float(initial_loss)
