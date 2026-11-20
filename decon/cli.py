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
    parser.add_argument('--mask-smoothing', type=float, default=0.65, help='Boundary smoothing sigma in image pixels; 0 disables')
    parser.add_argument('--mask-min-area', type=int, default=12, help='Remove islands/fill enclosed holes smaller than this many pixels; 0 disables')
    parser.add_argument('--legacy-inpainting', action='store_true', help='Use full-frame completion without crop or dilation for comparison')
    parser.add_argument('--person-crop', action=argparse.BooleanOptionalAction, default=True, help='Use a person-centered completion crop (default); --no-person-crop disables')
    parser.add_argument('--completion-policy', choices=['occlusion', 'silhouette'], default='occlusion',
                        help='Occlusion-only completion (default), or geometry minus visible mask')
    parser.add_argument('--ggdo-profile', choices=['paper', 'released-code'], default='released-code',
                        help='Released optimizer settings, or MSE objective described in the paper; neither restores missing upstream renderer code')
    parser.add_argument('--inpainting-task', choices=['text-guided','shape-guided'], default='text-guided')
    parser.add_argument('--inpainting-prompt', default='a complete person')
    parser.add_argument('--translation-steps', type=int)
    parser.add_argument('--pose-shape-steps', type=int)
    parser.add_argument('--render-max-edge', type=int, default=512,
                        help='Maximum rasterization edge; masks retain input resolution (0 uses native resolution)')
    args = parser.parse_args(argv)
    import math
    if not math.isfinite(args.mask_smoothing) or args.mask_smoothing < 0 or args.mask_min_area < 0:
        parser.error('Mask smoothing and minimum area must be finite and nonnegative')
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
    from dataclasses import replace
    ggdo = GGDOConfig.released_code() if args.ggdo_profile == 'released-code' else GGDOConfig()
    overrides = {name: getattr(args, name) for name in ('translation_steps','pose_shape_steps')
                 if getattr(args, name) is not None}
    ggdo = replace(ggdo, **overrides)
    pipeline = HumanDecouplingPipeline(
        OfficialSATHMRGeometryEstimator(sat_root, python_executable=args.sat_python, config=args.sat_config, threshold=args.sat_threshold),
        OfficialSAM2Segmenter(root / 'Img_Segmentation', checkpoint=args.sam_checkpoint, model_config=args.sam_config),
        PyTorch3DSilhouetteRenderer(body.faces, (rgb.height, rgb.width),
                                   max_render_size=args.render_max_edge or None),
        OfficialPowerPaintInpainter(root / 'Img_Inpainting', args.powerpaint_checkpoint,
                                   local_files_only=not args.allow_model_downloads,
                                   crop_to_person=args.person_crop and not args.legacy_inpainting,
                                   task=args.inpainting_task, prompt=args.inpainting_prompt),
        body_model=body,
        segmentation_prompts=prompts,
        completion_policy=args.completion_policy,
        mask_smoothing=args.mask_smoothing, mask_min_area=args.mask_min_area,
        ggdo_config=ggdo,
    )
    people = pipeline.run(rgb)
    save_results(rgb, people, args.output, body.faces)
    from dataclasses import asdict
    (args.output/'run_settings.json').write_text(json.dumps({
        'ggdo_profile':args.ggdo_profile,'ggdo':asdict(ggdo),
        'completion_policy':args.completion_policy,'render_max_edge':args.render_max_edge,
        'inpainting_task':args.inpainting_task,'inpainting_prompt':args.inpainting_prompt,
        'person_crop':args.person_crop and not args.legacy_inpainting,
        'segmentation_prompts':prompts,
        'mask_smoothing':args.mask_smoothing,'mask_min_area':args.mask_min_area,
        'exact_upstream_reproduction':False,
    },indent=2)+'\n')
    print(f'Saved {len(people)} decoupled people to {args.output}')


if __name__ == '__main__':
    main()
