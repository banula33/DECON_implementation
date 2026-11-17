"""Original-resolution human decoupling with SMPL-guided completion."""

from typing import Any

import torch
from torch import Tensor

from .adapters import BoundingBoxer, GeometryEstimator, Inpainter, MeshRenderer, Segmenter
from .ggdo import BodyModel, GGDOConfig, optimize_geometry
from .images import isolate, load_rgb
from .types import DecoupledPerson


class HumanDecouplingPipeline:
    def __init__(
        self, geometry_estimator: GeometryEstimator, segmenter: Segmenter,
        renderer: MeshRenderer, inpainter: Inpainter | None = None, *,
        body_model: BodyModel | None = None,
        bbox_for_person: BoundingBoxer | None = None,
        ggdo_config: GGDOConfig | None = None,
    ) -> None:
        self.geometry_estimator = geometry_estimator
        self.segmenter = segmenter
        self.renderer = renderer
        self.inpainter = inpainter
        self.body_model = body_model
        self.bbox_for_person = bbox_for_person or self._default_bbox
        self.ggdo_config = ggdo_config or GGDOConfig()

    def segment(self, image: Any):
        """Estimate everyone first so segmentation can reason about occlusion."""
        rgb = load_rgb(image)
        estimates = list(self.geometry_estimator.estimate(rgb))
        if not estimates:
            raise ValueError("SAT-HMR detected no people in the input image")
        boxes = [self.bbox_for_person(p.vertices, p.camera) for p in estimates]
        scene_segment = getattr(self.segmenter, "segment_people", None)
        if callable(scene_segment):
            masks = scene_segment(rgb, estimates, boxes, renderer=self.renderer,
                                  vertex_parts=getattr(self.body_model, "vertex_parts", None))
        else:
            masks = [self.segmenter.segment(rgb, box) for box in boxes]
        if len(masks) != len(estimates):
            raise ValueError("Segmentation must return one mask for each detected person")
        for estimate, box, mask in zip(estimates, boxes, masks):
            mask = mask.to(device=estimate.vertices.device, dtype=torch.bool)
            if tuple(mask.shape) != (rgb.height, rgb.width):
                raise ValueError("Segmentation must retain the original image resolution")
            if not mask.any():
                raise ValueError("SAM returned an empty person mask")
            estimate.mask = mask
            estimate.metadata = {**(estimate.metadata or {}), "bounding_box": box.detach().cpu().tolist()}
        return estimates

    def run(self, image: Any) -> list[DecoupledPerson]:
        rgb = load_rgb(image)
        results = []
        for estimate in self.segment(rgb):
            box = estimate.vertices.new_tensor(estimate.metadata["bounding_box"])
            mask = estimate.mask
            render = lambda vertices: self.renderer.render_mask(vertices, estimate.camera)
            initial_mask = render(estimate.vertices).detach()
            ggdo = optimize_geometry(
                estimate.vertices, mask, render, body_model=self.body_model,
                pose=estimate.pose, shape=estimate.shape, translation=estimate.translation,
                config=self.ggdo_config,
            )
            completion = (ggdo.geometry_mask > 0.5) & ~mask
            segmented_image = isolate(rgb, mask)
            completed = None
            if self.inpainter is not None:
                completed = self.inpainter.inpaint(segmented_image, mask, completion)
            results.append(DecoupledPerson(
                estimate=estimate, optimized_vertices=ggdo.vertices,
                geometry_mask=ggdo.geometry_mask, completion_mask=completion,
                image=completed, losses=ggdo.losses, segmented_image=segmented_image,
                initial_geometry_mask=initial_mask, bounding_box=box,
                optimized_pose=ggdo.pose, optimized_shape=ggdo.shape,
                optimized_translation=ggdo.translation,
            ))
        return results

    @staticmethod
    def _default_bbox(vertices: Tensor, camera: Any) -> Tensor:
        return camera.bounding_box(vertices)
