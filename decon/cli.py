"""Segment people or run human decoupling with installed official models."""

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
    parser.add_argument('--sam-guidance', choices=('geometry', 'box'), default='geometry',
                        help='Geometry and person prompts (default), or box-only SAM baseline')
    parser.add_argument('--segmentation-prompts', type=Path,
                        help='JSON mapping zero-based person indices to positive/negative [x, y] points')
    parser.add_argument('--segmentation-only', action='store_true',
                        help='Export visible person masks and prompts without GGDO or PowerPaint')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda',
                        help='Device for local models; SAT-HMR also needs a compatible upstream configuration')
    parser.add_argument('--powerpaint-checkpoint', type=Path,
                        help='Required for full decoupling; unused with --segmentation-only')
    parser.add_argument('--allow-model-downloads', action='store_true', help='Allow PowerPaint to download base models')
    parser.add_argument('--translation-steps', type=int, default=100)
    parser.add_argument('--pose-shape-steps', type=int, default=300)
    parser.add_argument('--render-max-edge', type=int, default=1024,
                        help='Maximum rasterization edge; masks retain input resolution (0 uses native resolution)')
    args = parser.parse_args(argv)
    if args.render_max_edge < 0:
        parser.error('--render-max-edge must be nonnegative')
    if not args.segmentation_only and args.powerpaint_checkpoint is None:
        parser.error('--powerpaint-checkpoint is required unless --segmentation-only is set')
    if not args.segmentation_only and args.device != 'cuda':
        parser.error('Full decoupling requires --device cuda for the official PowerPaint controller')
    if args.output.exists():
        parser.error('--output must be a new directory (existing results are never overwritten)')

    import torch
    from .artifacts import save_results, save_segmentations
    from .body_model import SMPLBodyModel
    from .ggdo import GGDOConfig
    from .images import load_rgb
    from .official_adapters import (OfficialSATHMRGeometryEstimator, OfficialSAM2Segmenter,
                                    OfficialPowerPaintInpainter, PyTorch3DSilhouetteRenderer)
    from .pipeline import HumanDecouplingPipeline
    from .segmentation import load_prompt_corrections

    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; segmentation-only can use --device cpu with a compatible SAT-HMR environment')
    corrections = None
    if args.segmentation_prompts is not None:
        try:
            corrections = load_prompt_corrections(args.segmentation_prompts)
        except (OSError, ValueError) as exc:
            parser.error(f'Cannot read --segmentation-prompts: {exc}')
    rgb = load_rgb(args.image)
    root = args.official_root.resolve()
    sat_root = root / 'Human_SMPL_Estimation'
    body = SMPLBodyModel(args.smpl_models or sat_root / 'weights/smpl_data', device=args.device)
    inpainter = None
    if not args.segmentation_only:
        inpainter = OfficialPowerPaintInpainter(
            root / 'Img_Inpainting', args.powerpaint_checkpoint,
            local_files_only=not args.allow_model_downloads,
        )
    pipeline = HumanDecouplingPipeline(
        OfficialSATHMRGeometryEstimator(sat_root, python_executable=args.sat_python, config=args.sat_config,
                                       threshold=args.sat_threshold, device=args.device),
        OfficialSAM2Segmenter(root / 'Img_Segmentation', checkpoint=args.sam_checkpoint,
                              model_config=args.sam_config, device=args.device,
                              guidance=args.sam_guidance, prompt_corrections=corrections),
        PyTorch3DSilhouetteRenderer(body.faces, (rgb.height, rgb.width), device=args.device,
                                   max_render_size=args.render_max_edge or None),
        inpainter,
        body_model=body,
        ggdo_config=GGDOConfig(translation_steps=args.translation_steps, pose_shape_steps=args.pose_shape_steps),
    )
    if args.segmentation_only:
        people = pipeline.segment(rgb)
        save_segmentations(rgb, people, args.output)
        print(f'Saved {len(people)} segmented people to {args.output}')
    else:
        people = pipeline.run(rgb)
        save_results(rgb, people, args.output, body.faces)
        print(f'Saved {len(people)} decoupled people to {args.output}')


if __name__ == '__main__':
    main()
