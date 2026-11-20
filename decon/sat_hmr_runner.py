"""Subprocess bridge to the official SAT-HMR environment.

Run with cwd set to Human_SMPL_Estimation. Export native camera-space values
and the actual model intrinsics, bypassing upstream's rotated OBJ exports.
"""

import argparse
from pathlib import Path
import sys


def move_target_tensors(value, device):
    """Move nested target tensors while preserving filenames and other metadata."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_target_tensors(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_target_tensors(item, device) for item in value)
    return value


def export_predictions(outputs, targets, destination, threshold, input_size=None):
    import numpy as np
    import torch

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for index, target in enumerate(targets):
        selected = torch.where(outputs['pred_confs'][index].reshape(-1) > threshold)[0]
        for query in selected:
            def array(key):
                return outputs[key][index][query].detach().cpu().numpy()
            extras = {}
            if 'pred_boxes' in outputs:
                if input_size is None:
                    raise ValueError("SAT-HMR input_size is required to export detection boxes")
                cx, cy, width, height = array('pred_boxes') * input_size
                extras['detection_box'] = np.array([cx-width/2, cy-height/2, cx+width/2, cy+height/2])
            np.savez_compressed(
                destination / f'person_{count:03d}.npz',
                vertices=array('pred_verts'), pose=array('pred_poses').reshape(24, 3),
                shape=array('pred_betas').reshape(10), translation=array('pred_transl').reshape(3),
                intrinsics=outputs['pred_intrinsics'][index].detach().cpu().numpy().reshape(3, 3),
                resized_size=target['img_size'].detach().cpu().numpy(),
                **extras,
            )
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--config', default='demo')
    options = parser.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    import torch
    import yaml
    from engines.engine import Engine

    with open(Path('configs/run') / f'{options.config}.yaml') as stream:
        settings = yaml.safe_load(stream)
    with open(Path('configs/models') / f"{settings['model']}.yaml") as stream:
        settings.update(yaml.safe_load(stream))
    settings.update(input_dir=options.input_dir, output_dir=options.output_dir,
                    exp_name='decon_decoupling', mode='infer', distributed_infer=False,
                    infer_batch_size=1, infer_num_workers=0, conf_thresh=[options.threshold])
    engine = Engine(argparse.Namespace(**settings), mode='infer')
    model = engine.unwrapped_model.eval()
    device = next(model.parameters()).device
    # The adapter provides exactly one input image in a private directory.
    for samples, targets in engine.infer_dataloader:
        targets = move_target_tensors(targets, device)
        with torch.inference_mode():
            outputs = model([sample.to(device) for sample in samples], targets)
        export_predictions(outputs, targets, options.output_dir, options.threshold, input_size=model.input_size)


if __name__ == '__main__':
    main()
