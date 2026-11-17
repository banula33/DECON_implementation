"""Save reviewable decoupling results and parameters for reconstruction/PAPO."""

import json
from pathlib import Path

from PIL import Image, ImageDraw
from torch import Tensor

from .images import isolate, load_rgb, mask_image
from .types import DecoupledPerson, HumanEstimate


def _save_segmentation(rgb, estimate, folder):
    mask = mask_image(estimate.mask)
    mask.save(folder / 'segmentation.png')
    isolate(rgb, estimate.mask).save(folder / 'segmented.png')
    overlay = Image.composite(Image.new('RGB', rgb.size, 'cyan'), rgb, mask)
    Image.blend(rgb, overlay, 0.45).save(folder / 'segmentation_overlay.png')
    metadata = estimate.metadata or {}
    info = metadata.get('segmentation', {})
    prompts = rgb.copy()
    draw = ImageDraw.Draw(prompts)
    radius = max(3, round(max(rgb.size) / 300))
    box = info.get('box', metadata.get('bounding_box'))
    if box is not None:
        draw.rectangle(box, outline='yellow', width=max(1, radius // 2))
    for point in info.get('points', []):
        x, y = point['xy']
        color = 'lime' if point['label'] else 'red'
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline='black', width=1)
        if point['source'] == 'correction':
            draw.rectangle((x-radius-2, y-radius-2, x+radius+2, y+radius+2), outline='white', width=2)
    prompts.save(folder / 'prompts.png')
    diagnostics = {
        **info, 'person_id': metadata.get('person_id', folder.name),
        'image_size': [rgb.height, rgb.width],
        'foreground_pixels': int(estimate.mask.sum().item()),
    }
    (folder / 'segmentation.json').write_text(json.dumps(diagnostics, indent=2) + '\n')


def save_segmentations(image, estimates: list[HumanEstimate], output_dir: str | Path) -> Path:
    """Export masks and prompt diagnostics without fitting or diffusion."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rgb = load_rgb(image)
    rgb.save(output / 'input.png')
    for index, estimate in enumerate(estimates):
        folder = output / f'person_{index:03d}'
        folder.mkdir()
        _save_segmentation(rgb, estimate, folder)
    (output / 'manifest.json').write_text(json.dumps({
        'people': len(estimates), 'stage': 'human_segmentation',
    }, indent=2) + '\n')
    return output


def save_results(image, people: list[DecoupledPerson], output_dir: str | Path, faces: Tensor) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rgb = load_rgb(image)
    rgb.save(output / 'input.png')
    for index, person in enumerate(people):
        folder = output / f'person_{index:03d}'
        folder.mkdir()
        _save_segmentation(rgb, person.estimate, folder)
        mask_image(person.geometry_mask).save(folder / 'geometry_after.png')
        mask_image(person.initial_geometry_mask).save(folder / 'geometry_before.png')
        mask_image(person.completion_mask).save(folder / 'completion_mask.png')
        person.segmented_image.save(folder / 'segmented.png')
        if person.image is not None:
            person.image.save(folder / 'completed.png')
        boxed = rgb.copy()
        ImageDraw.Draw(boxed).rectangle(person.bounding_box.detach().cpu().tolist(), outline='red', width=2)
        boxed.save(folder / 'bounding_box.png')
        for label, mask in [('before', person.initial_geometry_mask), ('after', person.geometry_mask)]:
            overlay = Image.composite(Image.new('RGB', rgb.size, 'cyan'), rgb, mask_image(mask))
            Image.blend(rgb, overlay, 0.4).save(folder / f'overlay_{label}.png')
        vertices = person.optimized_vertices.detach().cpu().numpy()
        triangles = faces.detach().cpu().numpy() + 1
        with (folder / 'optimized.obj').open('w') as stream:
            stream.write('# OpenCV camera coordinates: x right, y down, z forward\n')
            for v in vertices:
                stream.write('v ' + ' '.join(map(str, v)) + '\n')
            for f in triangles:
                stream.write('f ' + ' '.join(map(str, f)) + '\n')
        def values(tensor):
            return tensor.detach().cpu().tolist() if tensor is not None else None
        metadata = {
            'coordinate_system': 'opencv_camera',
            'image_size': list(person.estimate.camera.image_size),
            'intrinsics': values(person.estimate.camera.intrinsics),
            'poses': values(person.optimized_pose), 'betas': values(person.optimized_shape),
            'transl': values(person.optimized_translation),
            'initial_poses': values(person.estimate.pose), 'initial_betas': values(person.estimate.shape),
            'initial_transl': values(person.estimate.translation),
            'bounding_box': values(person.bounding_box), 'losses': person.losses,
            'initial_mse': float(((person.initial_geometry_mask - person.estimate.mask.float()) ** 2).mean()),
            'final_mse': float(((person.geometry_mask - person.estimate.mask.float()) ** 2).mean()),
        }
        (folder / 'parameters.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'manifest.json').write_text(json.dumps({'people': len(people), 'stage': 'human_decoupling'}, indent=2) + '\n')
    return output
