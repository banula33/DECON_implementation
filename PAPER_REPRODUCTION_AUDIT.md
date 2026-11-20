# DECON decoupling audit — 20 November 2026

This audit reads the supplied `03916-AAAI26.JiangY-CV (1).pdf`, especially
section 3.2 (PDF page 3), implementation details (pages 4–5), Figure 7 and
limitations (page 7), together with official source commit
`992364634b73bba05d68d5930ff20627e0487b4b`.

The earlier local results were not an exact reproduction. The changes below
correct identifiable deviations; they do not establish perfect reconstruction.

| Stage | Evidence in the supplied sources | Earlier local behavior | Correction/status |
|---|---|---|---|
| SAM box | `Human_SMPL_Estimation/engines/funcs/infer_funcs.py`, lines 153–190, saves SAT-HMR `pred_boxes`, expanded 2.5% per side | Discarded detection boxes; used projected mesh bounds | Export learned boxes, rescale from model input to original image, pad and clip; prefer them for SAM |
| SAM execution | `Img_Segmentation/sam.py`: one box, `multimask_output=False`, bfloat16 prediction after image encoding | Float32 mask prediction; later manual prompts/crops | Restore the released automatic box path and prediction precision; manual corrections remain explicitly optional |
| GGDO loss | Paper prose calls it MSE; released `src/ggdo/ggdo.py` uses mean absolute difference on masks scaled by 100 | MSE only | Separate `paper` and `released-code` optimizer profiles |
| GGDO pose preservation | Released stage 2 adds `50 * mean((theta_initial-theta)^2)` | No pose-preservation penalty | Include it in the released-code profile |
| GGDO motion | Released code keeps original depth, optimizes XY translation first, then pose/shape with translation fixed | Optimized XYZ and continued translation in stage 2 | Lock depth and stage-2 translation in released-code profile |
| GGDO schedule | 200 iterations split equally; Adam AMSGrad; LR .03/.002; ReduceLROnPlateau factor .5, patience 5, minimum 1e-5 | 100/300 steps; LR .01/.001; plain Adam | Implement the released settings |
| Completion support | Geometry defines completion scope; no intersection with other person masks is documented | Added an occlusion-only intersection as a workaround | Restore silhouette-minus-visible-mask as default; retain occlusion-only as a labeled experiment |
| PowerPaint | Repository bundles v2; application supports multiple task modes | Fixed text-guided task and a generic prompt | Expose task and prompt; test modes explicitly, without claiming authors used an undocumented choice |

## Why exact execution of the released GGDO entry point is blocked

The checked entry point refers to unavailable modules/assets, including
`lib.options`, `lib.data.SmplOptDataset_Multi`, `rendering_script.Render`,
`evaluate_model`, and tetrahedral SMPL support data. A similarly named
`GGDODataset_Multi.py` exists but itself references `apps.generatesmpl_opt_Multi`;
renaming it would not restore a complete runnable package.

The local port retains calibrated pinhole SMPL rendering instead of inventing the
missing renderer's camera settings. It also saves a coherent best parameter state
(pose, shape and translation together), unlike the released script's combination
of best pose and final shape/translation. The released-code profile now evaluates
the objective on native raster-resolution masks, using nearest-neighbor resized
segmentation targets, and restores original resolution only for output. The first
diagnostic run (`outputs/paper_audit/football`) predates this correction and also
used bfloat16 image encoding; use `final_football` and `final_dancing` for the
corrected precision/resolution runs. The missing renderer and coherent-state
handling remain explicit differences in the audit settings.

The paper and release do not specify a complete decoupling invocation: PowerPaint
task, positive/negative prompts, seeds, mask preparation and final compositing
settings for the published examples are not supplied. Shape-guided PowerPaint
is an available experiment, not a verified claim about DECON's published setup.

## What additional information would help

More arbitrary photos or newly generated photos will not resolve reproduction
ambiguity. Request the following from the DECON authors:

1. A runnable decoupling package, including the missing GGDO renderer, options,
   dataset adapter and required support assets, plus its environment specification.
2. One original example image with SAT-HMR parameters/boxes, SAM masks, optimized
   SMPL mask, exact PowerPaint input mask and expected completed images.
3. The corresponding commands, checkpoint revisions, optimizer settings,
   prompts, task mode, fitting degree, seed and mask/compositing procedure.

For objective segmentation accuracy on the user's football photo, manually
annotated per-person visible masks would provide ground truth. For verifying
hidden anatomy, additional views or ground-truth geometry are needed; one image
does not reveal that anatomy. These are validation aids, not prerequisites for
continuing implementation fixes.

The paper itself reports failure with severe (>50%) occlusion in its limitations
section. Faithful execution therefore cannot guarantee perfect output on every
single-image scene.

## Reproducible local checks

`scripts/audit_decoupling.py` separates geometry and completion processes to keep
GPU memory bounded. It saves box selection, optimizer settings and pixel checks.
Audit outputs are under `outputs/paper_audit/`; previous outputs remain intact.

```bash
HF_HOME="$PWD/checkpoints/huggingface" .venv-gpu/bin/python scripts/audit_decoupling.py \
  outputs/football/input/football.jpg --output outputs/paper_audit/new_run \
  --stage geometry --people 3
HF_HOME="$PWD/checkpoints/huggingface" .venv-gpu/bin/python scripts/audit_decoupling.py \
  outputs/football/input/football.jpg --output outputs/paper_audit/new_run \
  --stage completion
```

Official source: [DECON](https://github.com/IridescentJiang/DECON).
PowerPaint task descriptions: [official README](https://github.com/open-mmlab/PowerPaint#inference).
