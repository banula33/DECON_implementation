"""Geometry-guided SAM prompts and exclusive visible-person masks.

Geometry supplies sparse hints, never foreground pixels. A visible person can
have several disconnected components; no component is removed for its size.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def load_prompt_corrections(path: str | Path) -> dict:
    """Read {person_index: {positive: [[x,y], ...], negative: [...]}}."""
    with Path(path).open() as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("Segmentation prompts must be an object keyed by person index")
    return data


def validate_corrections(data, count, image_size):
    if data is not None and not isinstance(data, dict):
        raise ValueError("Segmentation prompts must be an object keyed by person index")
    h, w = image_size
    result = [{"positive": [], "negative": []} for _ in range(count)]
    for key, value in (data or {}).items():
        if str(key) not in {str(i) for i in range(count)}:
            raise ValueError(f"Unknown person index in segmentation prompts: {key}")
        if not isinstance(value, dict) or set(value) - {"positive", "negative"}:
            raise ValueError("Each person's prompts must contain positive/negative point lists")
        for label in ("positive", "negative"):
            points = np.asarray(value.get(label, []), dtype=np.float32)
            if points.shape in {(0,), (0, 2)}:
                continue
            if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
                    or (points < 0).any() or (points[:, 0] > w - 1).any()
                    or (points[:, 1] > h - 1).any()):
                raise ValueError("Prompt points must be finite [x, y] pairs inside the original image")
            result[int(key)][label] = points.tolist()
    # The same pixel cannot be a foreground point for two different people.
    owners = {}
    for i, item in enumerate(result):
        negative = {tuple(np.rint(p).astype(int)) for p in item["negative"]}
        for point in item["positive"]:
            pixel = tuple(np.rint(point).astype(int))
            if pixel in negative or (pixel in owners and owners[pixel] != i):
                raise ValueError("Contradictory segmentation prompts at the same pixel")
            owners[pixel] = i
    return result


def visible_geometry(depths, margin=0.01):
    """Only confidently frontmost surface pixels may become positive prompts.

    Depths share OpenCV camera coordinates in metres. Near-ties are omitted
    from automatic prompts, rather than assigned using detection order.
    """
    result = []
    for i, depth in enumerate(depths):
        visible = np.isfinite(depth)
        for j, other in enumerate(depths):
            if i != j:
                visible &= depth + margin < other
        result.append(visible)
    return result


def geometry_points(estimate, depth, visible, vertex_parts=None, max_points=48):
    """Cover disconnected visible islands and SMPL skinning regions (hands too).

    Use interior pixels in each island before sampling body regions. In
    particular, a tiny palm must not disappear through global mask erosion.
    SciPy is part of the optional official-model dependencies.
    """
    from scipy import ndimage

    ys, xs = np.nonzero(visible)
    if not len(xs):
        return np.empty((0, 2), dtype=np.float32)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = visible[y0:y1, x0:x1]
    # Padding makes distances correct even when the foreground fills the crop.
    distance = ndimage.distance_transform_edt(np.pad(crop, 1))[1:-1, 1:-1]
    components, count = ndimage.label(crop, structure=np.ones((3, 3), dtype=int))
    centers = ndimage.maximum_position(distance, components, range(1, count + 1))
    points = [[x + x0, y + y0] for y, x in centers]
    component_budget = max_points if vertex_parts is None else max_points // 2
    extra_components = points.copy()
    if len(points) > component_budget:
        # Keep sampling deterministic and distributed over the image.
        points = [points[i] for i in np.linspace(0, len(points) - 1, component_budget, dtype=int)]

    if vertex_parts is not None:
        parts = torch.as_tensor(vertex_parts).detach().cpu().numpy()
        vertices = estimate.vertices.detach().cpu().numpy()
        if parts.shape != (len(vertices),):
            raise ValueError("SMPL vertex_parts must have one part ID per vertex")
        xy = estimate.camera.project(estimate.vertices).detach().cpu().numpy()
        pixels = np.rint(xy).astype(int)
        h, w = visible.shape
        inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < w)
                  & (pixels[:, 1] >= 0) & (pixels[:, 1] < h))
        candidates = np.flatnonzero(inside)
        px, py = pixels[candidates].T
        # Reject back-facing/hidden vertices, including self-occlusion.
        surface = (visible[py, px]
                   & (np.abs(vertices[candidates, 2] - depth[py, px])
                      <= np.maximum(0.015, vertices[candidates, 2] * 0.01)))
        candidates = candidates[surface]
        for part in np.unique(parts[candidates]):
            ids = candidates[parts[candidates] == part]
            uv = pixels[ids]
            interior = distance[uv[:, 1] - y0, uv[:, 0] - x0]
            # Prefer the part's interior, then the point nearest its median.
            uv = uv[interior >= min(2.0, interior.max())]
            center = np.median(uv, axis=0)
            point = uv[np.argmin(((uv - center) ** 2).sum(1))].tolist()
            if point not in points:
                points.append(point)
    for point in extra_components:
        if point not in points:
            points.append(point)
    return np.asarray(points[:max_points], dtype=np.float32).reshape(-1, 2)


def person_prompts(index, automatic, corrections, *, exclusion_radius=6.0):
    """Explicit points take precedence over nearby automatic suggestions.

    A positive correction for one person is also negative for all others.
    """
    own = corrections[index]
    positive = own["positive"]
    negative = own["negative"] + [p for j, item in enumerate(corrections)
                                   if j != index for p in item["positive"]]
    explicit = positive + negative
    points, labels, sources = [], [], []
    for j, group in enumerate(automatic):
        for point in group:
            if explicit and np.min(np.linalg.norm(np.asarray(explicit) - point, axis=1)) <= exclusion_radius:
                continue
            points.append(point.tolist())
            labels.append(int(j == index))
            sources.append("geometry" if j == index else "other_person")
    for group, label in ((positive, 1), (negative, 0)):
        for point in group:
            points.append(point)
            labels.append(label)
            sources.append("correction")
    return (np.asarray(points, dtype=np.float32).reshape(-1, 2),
            np.asarray(labels, dtype=np.int32), sources)


def exclusive_masks(logits, depths=None, threshold=0.0, corrections=None):
    """Resolve only multiply claimed pixels; never add pixels from geometry.

    At overlapping surfaces use local depth (not a person's average depth)
    when every claimant has geometry. Missing/near-tied depth uses SAM logits.
    These logits are heuristic evidence, not calibrated ownership confidence.
    """
    shape = logits[0].shape
    best_score = np.full(shape, -np.inf, dtype=np.float32)
    owner = np.full(shape, -1, dtype=np.int32)
    nearest = np.full(shape, np.inf, dtype=np.float32)
    runner_up = np.full(shape, np.inf, dtype=np.float32)
    depth_owner = np.full(shape, -1, dtype=np.int32)
    unknown_depth = np.zeros(shape, dtype=bool)
    for i, score in enumerate(logits):
        claimed = score > threshold
        better = claimed & (score > best_score)
        best_score[better], owner[better] = score[better], i
        if depths is not None:
            unknown_depth |= claimed & ~np.isfinite(depths[i])
            depth = np.where(claimed, depths[i], np.inf)
            closer = depth < nearest
            runner_up = np.where(closer, nearest, np.minimum(runner_up, depth))
            nearest = np.minimum(nearest, depth)
            depth_owner[closer] = i
    if depths is not None:
        reliable = ~unknown_depth & np.isfinite(nearest) & (nearest + 0.01 < runner_up)
        owner[reliable] = depth_owner[reliable]
    if corrections and any(item["positive"] for item in corrections):
        from scipy import ndimage

        claims = np.zeros(shape, dtype=np.int32)
        for score in logits:
            claims += score > threshold
        components, _ = ndimage.label(claims > 1)
        anchors = {}
        for i, item in enumerate(corrections):
            for x, y in np.rint(item["positive"]).astype(int).reshape(-1, 2):
                component = components[y, x]
                if component and logits[i][y, x] > threshold:
                    anchors.setdefault(component, set()).add(i)
        for component, people in anchors.items():
            if len(people) != 1:
                continue
            i = next(iter(people))
            # A correction anchors an ambiguous island (e.g. a palm), never
            # inventing foreground or overriding contradictory region hints.
            if any(components[y, x] == component for x, y in
                   np.rint(corrections[i]["negative"]).astype(int).reshape(-1, 2)):
                continue
            region = (components == component) & (logits[i] > threshold)
            owner[region] = i
    return [owner == i for i in range(len(logits))]
