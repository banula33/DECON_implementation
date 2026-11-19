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
        segmentation_prompts: dict | None = None,
        completion_policy: str = 'occlusion',
    ) -> None:
        self.geometry_estimator = geometry_estimator
        self.segmenter = segmenter
        self.renderer = renderer
        self.inpainter = inpainter
        self.body_model = body_model
        self.bbox_for_person = bbox_for_person or self._default_bbox
        self.ggdo_config = ggdo_config or GGDOConfig()
        self.segmentation_prompts = segmentation_prompts or {}
        if completion_policy not in ('occlusion', 'silhouette'):
            raise ValueError("completion_policy must be occlusion or silhouette")
        self.completion_policy = completion_policy

    def run(self, image: Any) -> list[DecoupledPerson]:
        rgb = load_rgb(image)
        results = []
        for index, estimate in enumerate(self.geometry_estimator.estimate(rgb)):
            box = self.bbox_for_person(estimate.vertices, estimate.camera)
            correction = self.segmentation_prompts.get(str(index), {})
            if 'box' in correction:
                box = torch.as_tensor(correction['box'], device=box.device, dtype=box.dtype)
                if box.shape != (4,) or not torch.isfinite(box).all() or (box[2:] <= box[:2]).any():
                    raise ValueError("Correction box must be finite [left, top, right, bottom]")
            prompts = {k: correction[k] for k in ('points', 'labels', 'regions') if k in correction}
            mask = self.segmenter.segment(rgb, box, **prompts).to(device=estimate.vertices.device, dtype=torch.bool)
            if tuple(mask.shape) != (rgb.height, rgb.width):
                raise ValueError("Segmentation must retain the original image resolution")
            if not mask.any():
                raise ValueError("SAM returned an empty person mask")
            estimate.mask = mask
            render = lambda vertices: self.renderer.render_mask(vertices, estimate.camera)
            initial_mask = render(estimate.vertices).detach()
            ggdo = optimize_geometry(
                estimate.vertices, mask, render, body_model=self.body_model,
                pose=estimate.pose, shape=estimate.shape, translation=estimate.translation,
                config=self.ggdo_config,
            )
            completion = (ggdo.geometry_mask > 0.5) & ~mask
            segmented_image = isolate(rgb, mask)
            results.append(DecoupledPerson(
                estimate=estimate, optimized_vertices=ggdo.vertices,
                geometry_mask=ggdo.geometry_mask, completion_mask=completion,
                image=None, losses=ggdo.losses, segmented_image=segmented_image,
                initial_geometry_mask=initial_mask, bounding_box=box,
                optimized_pose=ggdo.pose, optimized_shape=ggdo.shape,
                optimized_translation=ggdo.translation,
                completion_policy=self.completion_policy, segmentation_prompt=correction or None,
            ))
        if not results:
            raise ValueError("SAT-HMR detected no people in the input image")
        # Finish segmentation/fitting for everyone before loading the large
        # completion model. SAM's model and image embeddings otherwise occupy
        # GPU memory throughout PowerPaint's checkpoint loading.
        release = getattr(self.segmenter, "release", None)
        if release is not None:
            release()
        if self.completion_policy == 'occlusion':
            # Silhouette disagreement alone is not evidence of missing anatomy:
            # loose clothing and pose errors create strips on unobstructed edges.
            # Only synthesize where another segmented person is visible.
            for index, person in enumerate(results):
                occluders = torch.zeros_like(person.completion_mask)
                for other_index, other in enumerate(results):
                    if other_index != index:
                        occluders |= other.estimate.mask.to(occluders.device)
                person.completion_mask &= occluders
        if self.inpainter is not None:
            for person in results:
                person.image = self.inpainter.inpaint(
                    person.segmented_image, person.estimate.mask, person.completion_mask,
                )
        return results

    @staticmethod
    def _default_bbox(vertices: Tensor, camera: Any) -> Tensor:
        return camera.bounding_box(vertices)
