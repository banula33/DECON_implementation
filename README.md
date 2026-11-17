# DECON human decoupling

Implementation of the human-decoupling stage in **DECON: Reconstruction of
Clothed-Geometric Multiple Humans from a Single Image via Geometry-Guided
Decoupling** (AAAI 2026, section 3.2).

The code connects SAT-HMR → geometry-guided SAM 2 person segmentation →
two-stage SMPL GGDO → isolated PowerPaint completion. Segmentation can also run
on its own for inspection and correction. Multi-view synthesis, NeuS
reconstruction, and PAPO are not implemented here.

The default segmentation extends the paper's box prompts with visible SMPL
surface points and other-person exclusion points. This helps SAM associate
separated visible regions, such as a palm beyond an occluded arm, with one
person. `--sam-guidance box` retains a box-prompt baseline for comparison.

## Status and validation

The pipeline, adapters, command-line runner, and result export are implemented.
Regression tests cover disconnected palms, hand-region prompts, crossing-person
depth, prompt corrections, exclusive mask ownership, parametric optimization,
the SAT-HMR subprocess bridge, image isolation, and orchestration. Optional
PyTorch3D tests cover camera projection, visibility, silhouette gradients, and
translation fitting; they are skipped when PyTorch3D is not installed.
Foundation-model outputs and body geometry in the main suite are synthetic
fixtures. Passing these tests does **not** establish segmentation or
reconstruction quality on real images, or validate the pretrained models together.

A real-image run still requires licensed SMPL assets, pretrained checkpoints,
compatible upstream environments, and CUDA for the official PowerPaint
controller. The current workspace does not contain the vendor checkout or
runtime checkpoints referenced by earlier setup notes. The current CPU test
environment uses PyTorch 2.6.0+cpu; real-model validation remains outstanding.

## Install

Use Python 3.10 or newer for this package. Follow the upstream model projects'
Python/PyTorch/CUDA requirements for production; their environments may need
separate versions. SAT-HMR can use a separate interpreter with `--sat-python`.

```bash
python3 -m venv .venv
source .venv/bin/activate
# Install an appropriate PyTorch build for your CPU or CUDA setup first.
pip install -e '.[test,official]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Disabling pytest plugin autoload avoids unrelated globally configured plugins
(e.g. ROS launch-testing) affecting this repository's tests.

Fetch the [official DECON source](https://github.com/IridescentJiang/DECON) into
`vendor/DECON` (ignored by Git):

```bash
git clone https://github.com/IridescentJiang/DECON.git vendor/DECON
```

Adapters were checked against upstream commit
`992364634b73bba05d68d5930ff20627e0487b4b`. The SAT-HMR bridge imports its
`Engine` and exports predictions directly, so upstream API changes may require
adjustments. It does not alter the official checkout or delete demo results.

Required assets and dependencies:

- **SAT-HMR:** install its dependencies and provide the checkpoint selected by
  `configs/run/demo.yaml` (currently `weights/sat_hmr/sat_644.pth`).
- **SMPL:** obtain licensed female, male, and neutral models using the official
  README links. Put them under `Human_SMPL_Estimation/weights/smpl_data/smpl`.
  SAT-HMR also needs the support files distributed by its authors, including
  `body_verts_smpl.npy`, `J_regressor_h36m_correct.npy`, and
  `smpl_mean_params.npz`. The optimizer uses the neutral SMPL model.
- **SAM 2:** install its package and use a checkpoint/config pair that match
  (the CLI defaults to the SAM 2.1 Hiera large config).
- **PowerPaint v2 (full decoupling only):** install its dependencies; supply the `ppt-v2` directory
  containing `realisticVisionV60B1_v51VAE` and `PowerPaint_Brushnet`. Its
  Stable Diffusion base models must also be cached, or pass
  `--allow-model-downloads` to allow the controller to fetch them.
- **PyTorch3D:** install a build compatible with your PyTorch/CUDA versions.
  Camera conversion uses its official `cameras_from_opencv_projection` API.

The helper can arrange downloaded SMPL and SAT-HMR files:

```bash
python scripts/prepare_sat_hmr_weights.py /path/to/downloads \
  --destination vendor/DECON/Multi-View_Synthesis/Geo_Gui_Hm_Decou/Human_SMPL_Estimation/weights
python scripts/check_official_setup.py \
  --official-root vendor/DECON/Multi-View_Synthesis/Geo_Gui_Hm_Decou
```

The helper only arranges files; it does not acquire licensed models or the
additional support arrays. Preflight reports file/package availability and
CUDA status; it cannot prove that upstream environments are compatible.

Legacy SMPL pickles may contain Chumpy objects. To use them without Chumpy
at runtime, first arrange `SMPL_FEMALE.pkl`, `SMPL_MALE.pkl`, and
`SMPL_NEUTRAL.pkl` in a staging directory, then run the converter in a Python
environment containing NumPy, SciPy, and Chumpy 0.70:

```bash
python scripts/convert_smpl_models.py /path/to/staged-models /path/to/new-converted-models
```

Alternatively, pass `--chumpy-source /path/to/extracted/chumpy-0.70` to use an
extracted Chumpy source distribution. Copy the converted models into
`Human_SMPL_Estimation/weights/smpl_data/smpl`. Conversion preserves the
source files and checks the converted Chumpy array values. Only use trusted
model pickles obtained from the model providers.

## Run human decoupling

```bash
python -m decon.cli input.jpg \
  --official-root vendor/DECON/Multi-View_Synthesis/Geo_Gui_Hm_Decou \
  --sam-checkpoint /path/to/sam2.1_hiera_large.pt \
  --powerpaint-checkpoint /path/to/ppt-v2 \
  --output outputs/example
```

Optional arguments:

- `--sat-python /path/to/sat-env/bin/python`: SAT-HMR environment.
- `--sat-config demo`: SAT-HMR run configuration name.
- `--sat-threshold 0.3`: person detection confidence threshold.
- `--smpl-models /path/to/smpl_data`: directory containing `smpl/`.
- `--sam-config configs/sam2.1/sam2.1_hiera_l.yaml`: matching SAM configuration.
- `--sam-guidance geometry`: visible geometry and person prompts (default).
  Use `box` for box prompts without automatic geometry points.
- `--segmentation-prompts prompts.json`: positive/negative correction points;
  see the segmentation-only workflow below.
- `--segmentation-only`: stop after exporting person masks and diagnostics.
- `--device cuda`: device for local models and prediction tensors. For
  segmentation-only, `cpu` is available if the separate SAT-HMR environment and
  its run configuration support CPU inference. This flag does not rewrite
  upstream SAT-HMR configuration. Full PowerPaint decoupling requires CUDA.
- `--translation-steps 100 --pose-shape-steps 300`: optimization iterations.
- `--render-max-edge 1024`: cap the differentiable rasterization resolution
  to limit GPU memory use. Intrinsics are scaled for rasterization and soft
  masks are bilinearly restored to original resolution. This approximates
  fine silhouette detail; `0` uses native resolution and requires more memory.
  Depth prompts use this same cap with nearest-neighbor restoration. Increase
  it if a very small hand disappears from the projected geometry.

The output directory must be new. Each `person_000/`, `person_001/`, etc.
contains:

- `segmentation.png`, `segmented.png`, and `bounding_box.png`;
- `segmentation_overlay.png`, `prompts.png`, and `segmentation.json` for review;
- `geometry_before.png`, `geometry_after.png`, and corresponding overlays;
- `completion_mask.png` and `completed.png`;
- `optimized.obj` and `parameters.json` (initial/fitted SMPL parameters,
  original-resolution camera intrinsics, bounding box, and loss history).

Images retain the original resolution. Meshes use **OpenCV camera coordinates**
(x right, y down, z forward), without the 180-degree OBJ rotation applied by
the upstream visualization exporter. Parameters and segmentation artifacts
are retained for subsequent reconstruction and scene positioning.

## Inspect and correct segmentation first

This path uses SAT-HMR, SMPL, SAM 2, and PyTorch3D. It does not load PowerPaint
or run geometry fitting or image completion:

```bash
python -m decon.cli input.jpg \
  --official-root vendor/DECON/Multi-View_Synthesis/Geo_Gui_Hm_Decou \
  --sam-checkpoint /path/to/sam2.1_hiera_large.pt \
  --segmentation-only \
  --output outputs/segmentation
```

Inspect each `person_000/`, `person_001/`, etc. The output includes the binary
`segmentation.png`, isolated `segmented.png`, `segmentation_overlay.png`, prompt
locations in `prompts.png`, and prompt/mask diagnostics in `segmentation.json`.
Green points include a region, red points exclude it, and white outlines mark
explicit corrections. Review `unmet_points` and `overlap_pixels_removed` in the
JSON; `needs_review: false` is not a segmentation quality guarantee.
Masks represent each person's **visible pixels**, including disconnected
components. They are whole-person instance masks, not semantic labels such as
"left palm" or "right forearm". Occluded anatomy is handled later by completion.

If a visible palm is missing or assigned to the wrong person, add a positive
point inside it for the correct person. A positive point automatically becomes
a negative point for other people. Use negative points to reject unwanted
regions. For example, save `prompts.json` as:

```json
{
  "0": {
    "positive": [[420, 180], [610, 260]],
    "negative": [[500, 220]]
  },
  "1": {
    "positive": [[510, 170]]
  }
}
```

Replace these example points with coordinates from your image. Points use
original-image pixels `[x, y]`, with the origin at the top left. Person IDs are
zero-based detection indices matching the output folders; confirm the intended
person in the first run and keep the input and SAT-HMR settings unchanged.
Then run again with a new output directory:

```bash
python -m decon.cli input.jpg \
  --official-root vendor/DECON/Multi-View_Synthesis/Geo_Gui_Hm_Decou \
  --sam-checkpoint /path/to/sam2.1_hiera_large.pt \
  --segmentation-only \
  --segmentation-prompts prompts.json \
  --output outputs/segmentation_corrected
```

The same prompt file can be used for full decoupling. Compare an uncorrected run
with `--sam-guidance box` in another output directory to inspect the effect of
automatic geometry prompts.

Automatic association depends on the estimated pose and camera alignment.
Incorrect hand placement, heavy occlusion, very small visible regions, and
occluders absent from the estimated scene can still produce wrong prompts or
masks. Geometry prompts and corrections are hints to SAM, not a guarantee of
correct ownership. Inspect the masks before using them for reconstruction;
there is no image evidence to recover a completely invisible palm during
visible-pixel segmentation.

## Implementation details

- SAT-HMR runs in a subprocess with a private input/output directory. The bridge
  exports the model's actual intrinsics, camera-space vertices, pose, shape,
  and translation. Intrinsics are rescaled to the original image resolution.
- SAM 2 receives projected, padded, clipped pixel-space boxes and, by default,
  points sampled from visible SMPL surface regions, including hands and
  disconnected visible islands. Local rendered depth suppresses prompts hidden
  by other people and avoids near-tied depth assignments. Other-person points
  provide negative prompts, and explicit corrections override nearby automatic
  points. The image embedding is shared across people.
- Overlapping SAM masks are assigned exclusively using local surface depth
  when every claimant has a surface and depth ordering is unambiguous, and SAM
  logits otherwise. A uniquely corrected overlap island follows its positive
  correction unless that conflicts with another explicit region hint.
  Geometry never creates foreground
  pixels, and small disconnected components are not removed by size. Empty
  detections and empty person masks fail explicitly.
- Prompt refinement uses SAM's original-resolution points and previous
  low-resolution mask logits; see the
  [official SAM 2 predictor interface](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py).
- GGDO first optimizes translation, then pose, shape, and translation through a
  differentiable SMPL model. The objective is silhouette MSE, following paper
  equations 3–4; the best encountered fit is retained. This follows the paper's
  formulation rather than copying all heuristics in the released GGDO script.
  The iteration counts and learning rates are configurable engineering defaults,
  not claimed paper hyperparameters. No unconstrained vertex deformation is used.
- The SMPL forward result must reproduce SAT-HMR's vertices before fitting.
  This catches body-model or coordinate-system mismatches early.
- Completion is the optimized silhouette minus the visible person mask. Other
  people and background are replaced with white before PowerPaint. Generated
  pixels are composited only inside the completion mask; visible pixels are
  preserved exactly. PowerPaint's resized output is restored to input size.
  An empty completion mask skips the diffusion model.
- A lower fitting loss alone does not establish anatomical correctness under
  occlusion. Inspect overlays and completed images on representative examples.
  Severe occlusion remains a limitation of the paper's approach.

## Acceptance check for a real run

Use a two-person image with moderate overlap. Check that both people are
segmented separately, initial mesh projections match the camera, optimized
silhouettes improve alignment, and completed images contain only the intended
person with plausible missing anatomy. Verify original resolution and visible
pixel preservation. Repeat with different poses before calling this stage
validated on real data.

After decoupling is validated, implement geometry-conditioned multi-view
RGB/normal generation, NeuS reconstruction, then PAPO (paper section 3.3).
PAPO describes a target built from original-resolution segmented **images**;
its exact projection representation should be verified before reducing it to
binary silhouettes.
