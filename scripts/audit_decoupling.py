"""Reproducible, staged audit against recoverable DECON release settings.

Run geometry first, then completion in a separate process to bound GPU memory.
This does not claim to restore missing upstream modules or undocumented settings.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['geometry', 'completion'], required=True)
    parser.add_argument('--people', type=int, default=0, help='Select largest detection boxes; zero keeps all')
    parser.add_argument('--task', choices=['text-guided','shape-guided'], default='text-guided')
    parser.add_argument('--crop', action='store_true', help='Explicit local experiment, not a documented DECON setting')
    parser.add_argument('--prompt', default='a complete person')
    args = parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from decon.artifacts import save_results
    from decon.body_model import SMPLBodyModel
    from decon.ggdo import GGDOConfig
    from decon.images import load_rgb
    from decon.official_adapters import (OfficialSATHMRGeometryEstimator, OfficialSAM2Segmenter,
                                         OfficialPowerPaintInpainter, PyTorch3DSilhouetteRenderer)
    from decon.pipeline import HumanDecouplingPipeline

    official = ROOT/'vendor/DECON/Multi-View_Synthesis/Geo_Gui_Hm_Decou'
    rgb = load_rgb(args.image)
    if args.stage == 'geometry':
        if args.output.exists():
            parser.error('Geometry output must be a new directory')
        estimator = OfficialSATHMRGeometryEstimator(official/'Human_SMPL_Estimation')
        estimates = estimator.estimate(rgb)
        if not estimates or any('detection_box' not in e.metadata for e in estimates):
            raise RuntimeError('Audit requires fresh SAT-HMR detection boxes')
        if args.people:
            def area(e):
                x0,y0,x1,y1=e.metadata['detection_box']
                return (x1-x0)*(y1-y0)
            estimates = sorted(estimates,key=area,reverse=True)[:args.people]
        class Cached:
            def estimate(self, _):
                return estimates
        body = SMPLBodyModel(official/'Human_SMPL_Estimation/weights/smpl_data')
        config = GGDOConfig.released_code()
        pipeline = HumanDecouplingPipeline(
            Cached(), OfficialSAM2Segmenter(official/'Img_Segmentation',
                checkpoint=official/'Img_Segmentation/checkpoints/sam2.1_hiera_large.pt'),
            PyTorch3DSilhouetteRenderer(body.faces,(rgb.height,rgb.width),max_render_size=512),
            body_model=body,ggdo_config=config,completion_policy='silhouette',
        )
        results = pipeline.run(rgb)
        save_results(rgb,results,args.output,body.faces)
        (args.output/'audit_settings.json').write_text(json.dumps({
            'source_commit':'992364634b73bba05d68d5930ff20627e0487b4b',
            'exact_upstream_reproduction':False,'ggdo':asdict(config),
            'render_max_edge':512,'box_source':'SAT-HMR pred_boxes; 2.5% padding per side',
            'manual_prompts':False,'completion_policy':'silhouette',
            'sam_precision':'float32 image encoding; bfloat16 mask prediction',
            'selection':[e.metadata for e in estimates],
            'limitations':['Missing upstream renderer; local calibrated pinhole renderer retained',
                           'Coherent best parameter state retained instead of upstream mixed best/final state'],
        },indent=2)+'\n')
        print(f'Saved {len(results)} fits to {args.output}',flush=True)
    else:
        if not (args.output/'manifest.json').exists():
            parser.error('Run geometry first')
        label = args.task + ('-crop' if args.crop else '-full')
        dest = args.output/label
        dest.mkdir(exist_ok=False)
        inpainter = OfficialPowerPaintInpainter(official/'Img_Inpainting',
            official/'Img_Inpainting/checkpoints/ppt-v2',crop_to_person=args.crop,
            mask_dilation=0,task=args.task,prompt=args.prompt,fitting_degree=0.9)
        checks=[]
        for folder in sorted(args.output.glob('person_*')):
            inpainter.debug_directory = dest/folder.name
            human=np.array(Image.open(folder/'segmentation.png'))>127
            completion=np.array(Image.open(folder/'completion_mask.png'))>127
            result=inpainter.inpaint(Image.open(folder/'segmented.png'),
                                    torch.from_numpy(human),torch.from_numpy(completion))
            result.save(dest/f'{folder.name}.png')
            a=np.asarray(result)
            assert a.shape==np.asarray(rgb).shape and np.array_equal(a[human],np.asarray(rgb)[human])
            assert (a[~(human|completion)]==255).all()
            checks.append({'person':folder.name,'visible_pixels_preserved':True,
                           'resolution_preserved':True,'outside_region_white':True})
            (dest/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
            print(folder.name,'completed',flush=True)
        (dest/'settings.json').write_text(json.dumps({'task':args.task,'crop':args.crop,
            'mask_dilation':0,'prompt':args.prompt,'fitting_degree':0.9,'seed':42,'steps':50,
            'paper_specifies_these_inpainting_settings':False},indent=2)+'\n')


if __name__=='__main__':
    main()
