"""Human decoupling orchestration: estimate, segment, GGDO, and inpaint."""

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .adapters import BoundingBoxer, GeometryEstimator, Inpainter, MeshRenderer, Segmenter
from .ggdo import GGDOConfig, optimize_geometry
from .types import DecoupledPerson


class HumanDecouplingPipeline:
    def __init__(
        self,
        geometry_estimator: GeometryEstimator,
        segmenter: Segmenter,
        renderer: MeshRenderer,
        inpainter: Inpainter | None = None,
        *,
        bbox_for_person: BoundingBoxer | None = None,
        ggdo_config: GGDOConfig | None = None,
    ) -> None:
        self.geometry_estimator = geometry_estimator
        self.segmenter = segmenter
        self.renderer = renderer
        self.inpainter = inpainter
        self.bbox_for_person = bbox_for_person or self._default_bbox
        self.ggdo_config = ggdo_config or GGDOConfig()

    def run(self, image: Any) -> list[DecoupledPerson]:
        results: list[DecoupledPerson] = []
        for estimate in self.geometry_estimator.estimate(image):
            box = self.bbox_for_person(estimate.vertices, estimate.camera)
            segmented_mask = self.segmenter.segment(image, box).bool()
            render = lambda vertices: self.renderer.render_mask(vertices, estimate.camera)
            ggdo = optimize_geometry(
                estimate.vertices,
                segmented_mask,
                render,
                config=self.ggdo_config,
            )
            completion_mask = (ggdo.geometry_mask > 0.5) & ~segmented_mask
            completed_image = None
            if self.inpainter is not None:
                completed_image = self.inpainter.inpaint(
                    image, segmented_mask, completion_mask
                )
            results.append(
                DecoupledPerson(
                    estimate=estimate,
                    optimized_vertices=ggdo.vertices,
                    geometry_mask=ggdo.geometry_mask,
                    completion_mask=completion_mask,
                    image=completed_image,
                    losses=ggdo.losses,
                )
            )
        return results

    @staticmethod
    def _default_bbox(vertices: Tensor, camera: Any) -> Tensor:
        del camera
        xy = vertices[:, :2]
        return torch.stack((xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()))
