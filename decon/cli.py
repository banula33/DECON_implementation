"""Run the complete human decoupling stage with installed official models."""

import argparse
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--official-root', type=Path, required=True,
                        help='DECON/Multi-View_Synthesis/Geo_Gui_Hm_Decou')
    parser.add_argument('--output', type=Path, required=True, help='New directory for per-person results')
    parser.add_argument('--smpl-models', type=Path, help='SMPL model root containing smpl/SMPL_NEUTRAL.pkl')
    parser.add_argument('--sat-python', default=sys.executable)
    parser.add_argument('--sat-config', default='demo')
    parser.add_argument('--sat-threshold', type=float, default=0.3)
    parser.add_argument('--sam-checkpoint', type=Path, required=True)
    parser.add_argument('--sam-config', default='configs/sam2.1/sam2.1_hiera_l.yaml')
    parser.add_argument('--powerpaint-checkpoint', type=Path, required=True)
    parser.add_argument('--allow-model-downloads', action='store_true', help='Allow PowerPaint to download base models')
    parser.add_argument('--segmentation-prompts', type=Path,
                        help='JSON with people keyed by detection index: optional box, points and labels (1=include, 0=exclude), in original pixels')
    parser.add_argument('--legacy-inpainting', action='store_true', help='Use full-frame completion without crop or dilation for comparison')
    parser.add_argument('--completion-policy', choices=['occlusion', 'silhouette'], default='occlusion',
                        help='Only fill geometry hidden by another person (default), or every silhouette disagreement (paper baseline)')
    parser.add_argument('--translation-steps', type=int, default=100)
    parser.add_argument('--pose-shape-steps', type=int, default=300)
    parser.add_argument('--render-max-edge', type=int, default=1024,
                        help='Maximum rasterization edge; masks retain input resolution (0 uses native resolution)')
    args = parser.parse_args(argv)
    if args.render_max_edge < 0:
        parser.error('--render-max-edge must be nonnegative')

    import torch
    from .artifacts import save_results
    from .body_model import SMPLBodyModel
    from .ggdo import GGDOConfig
    from .images import load_rgb
    from .official_adapters import (OfficialSATHMRGeometryEstimator, OfficialSAM2Segmenter,
                                    OfficialPowerPaintInpainter, PyTorch3DSilhouetteRenderer)
    from .pipeline import HumanDecouplingPipeline

    if args.output.exists():
        parser.error('--output must be a new directory (existing results are never overwritten)')
    if not torch.cuda.is_available():
        parser.error('The official PowerPaint pipeline requires CUDA; use a CUDA-enabled environment')
    rgb = load_rgb(args.image)
    root = args.official_root.resolve()
    sat_root = root / 'Human_SMPL_Estimation'
    body = SMPLBodyModel(args.smpl_models or sat_root / 'weights/smpl_data')
    import json
    prompts = json.loads(args.segmentation_prompts.read_text())['people'] if args.segmentation_prompts else None
    pipeline = HumanDecouplingPipeline(
        OfficialSATHMRGeometryEstimator(sat_root, python_executable=args.sat_python, config=args.sat_config, threshold=args.sat_threshold),
        OfficialSAM2Segmenter(root / 'Img_Segmentation', checkpoint=args.sam_checkpoint, model_config=args.sam_config),
        PyTorch3DSilhouetteRenderer(body.faces, (rgb.height, rgb.width),
                                   max_render_size=args.render_max_edge or None),
        OfficialPowerPaintInpainter(root / 'Img_Inpainting', args.powerpaint_checkpoint,
                                   local_files_only=not args.allow_model_downloads,
                                   crop_to_person=not args.legacy_inpainting),
        body_model=body,
        segmentation_prompts=prompts,
        completion_policy=args.completion_policy,
        ggdo_config=GGDOConfig(translation_steps=args.translation_steps, pose_shape_steps=args.pose_shape_steps),
    )
    people = pipeline.run(rgb)
    save_results(rgb, people, args.output, body.faces)
    print(f'Saved {len(people)} decoupled people to {args.output}')


if __name__ == '__main__':
    main()
