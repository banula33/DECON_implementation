"""Save reviewable decoupling results and parameters for reconstruction/PAPO."""

import json
from pathlib import Path

from PIL import Image, ImageDraw
from torch import Tensor

from .images import load_rgb, mask_image
from .types import DecoupledPerson


def save_results(image, people: list[DecoupledPerson], output_dir: str | Path, faces: Tensor) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rgb = load_rgb(image)
    rgb.save(output / 'input.png')
    for index, person in enumerate(people):
        folder = output / f'person_{index:03d}'
        folder.mkdir()
        mask_image(person.estimate.mask).save(folder / 'segmentation.png')
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
            'completion_policy': person.completion_policy,
            'segmentation_prompt': person.segmentation_prompt,
            'initial_mse': float(((person.initial_geometry_mask - person.estimate.mask.float()) ** 2).mean()),
            'final_mse': float(((person.geometry_mask - person.estimate.mask.float()) ** 2).mean()),
        }
        (folder / 'parameters.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'manifest.json').write_text(json.dumps({'people': len(people), 'stage': 'human_decoupling'}, indent=2) + '\n')
    return output
