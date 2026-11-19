# DECON human decoupling

Implementation of the human-decoupling stage in **DECON: Reconstruction of
Clothed-Geometric Multiple Humans from a Single Image via Geometry-Guided
Decoupling** (AAAI 2026, section 3.2).

The code now connects SAT-HMR → camera-projected bounding boxes → SAM 2 →
two-stage SMPL GGDO → isolated PowerPaint completion. Multi-view synthesis,
NeuS reconstruction, and PAPO are not implemented here.

## Status and validation

The pipeline, adapters, command-line runner, and result export are implemented.
The regression suite covers real PyTorch3D camera projection,
silhouette gradients, and translation fitting on CPU. Tests also cover parametric
optimization, a SAT-HMR subprocess bridge, image isolation, and two-person
orchestration. Foundation-model outputs and body geometry are synthetic fixtures;
these tests do **not** establish reconstruction quality or validate the pretrained
models together. Renderer tests are skipped if PyTorch3D is not installed.
The checked environment uses PyTorch 2.6.0+cpu and PyTorch3D 0.7.8
(`stable` commit `75ebeeaea0908c5527e7b1e305fbc7681382db47`).

A real-image run still requires licensed SMPL assets, pretrained checkpoints,
compatible upstream environments, and CUDA for the official PowerPaint
controller. The development `.venv` is a CPU testing environment.

The supplied football photo has now passed through SAT-HMR, SAM 2, GGDO, and
PowerPaint. Outputs preserve the visible-mask pixels and original resolution,
but the central players have hand/edge artifacts, so visual acceptance has not
passed. See [current run notes](SETUP_STATUS.md).

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

A reference checkout is available locally in `vendor/DECON` (ignored by Git).
For a fresh workspace, fetch the [official DECON source](https://github.com/IridescentJiang/DECON):

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
- **PowerPaint v2:** install its dependencies; supply the `ppt-v2` directory
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
- `--segmentation-prompts prompts.json`: optional box corrections and positive/
  negative points for missed visible limbs. Coordinates use the original image.
- `--completion-policy occlusion`: conservative default, filling only predicted
  body regions covered by another person's segmentation. `silhouette` restores
  the paper-style geometry-minus-mask baseline, including exposed edge errors.
- `--legacy-inpainting`: disable person-centered crop and generation-mask dilation
  for comparison; combine with `--completion-policy silhouette` for the old path.
- `--translation-steps 100 --pose-shape-steps 300`: optimization iterations.
- `--render-max-edge 1024`: cap the differentiable rasterization resolution
  to limit GPU memory use. Intrinsics are scaled for rasterization and soft
  masks are bilinearly restored to original resolution. This approximates
  fine silhouette detail; `0` uses native resolution and requires more memory.

The output directory must be new. Each `person_000/`, `person_001/`, etc.
contains:

- `segmentation.png`, `segmented.png`, and `bounding_box.png`;
- `geometry_before.png`, `geometry_after.png`, and corresponding overlays;
- `completion_mask.png` and `completed.png`;
- `optimized.obj` and `parameters.json` (initial/fitted SMPL parameters,
  original-resolution camera intrinsics, bounding box, and loss history).

Images retain the original resolution. Meshes use **OpenCV camera coordinates**
(x right, y down, z forward), without the 180-degree OBJ rotation applied by
the upstream visualization exporter. Parameters and segmentation artifacts
are retained for subsequent reconstruction and scene positioning.

## Implementation details

- SAT-HMR runs in a subprocess with a private input/output directory. The bridge
  exports the model's actual intrinsics, camera-space vertices, pose, shape,
  and translation. Intrinsics are rescaled to the original image resolution.
- SAM 2 receives projected, padded, clipped pixel-space boxes. Empty detections
  and empty masks fail explicitly.
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
  The default intersects this difference with other people's visible masks to
  avoid generating anatomy from exposed silhouette errors. This is an engineering
  refinement, not the paper's original mask rule. It does not address occlusion by
  objects or out-of-frame anatomy, and depends on the quality of the other masks.
  An empty completion mask skips the diffusion model.
  PowerPaint receives a square person crop with 12% context and a generation mask
  dilated by 12 pixels at its 640-pixel working size. Only the original, undilated
  completion region is composited back; visible pixels remain exact.
- PowerPaint v2 retains the official prediction code and weights, but constructs
  BrushNet without redundant initialized model copies and loads components onto
  the GPU before enabling CPU offloading. Checkpoint keys and shapes are checked
  before materialization. A same-seed first-player comparison produced identical
  pixels while reducing observed peak resident RAM from 10.45 to 7.53 GiB.
- The pipeline finishes segmentation and geometry fitting for all people, then
  releases SAM's model and image embeddings before loading PowerPaint. This
  avoids keeping both large models on the GPU during completion loading.
- A lower fitting loss alone does not establish anatomical correctness under
  occlusion. Inspect overlays and completed images on representative examples.
  Severe occlusion remains a limitation of the paper's approach.

## Correcting missing visible parts

If a hand, foot, or disconnected visible limb is missing, add a positive point
inside that region and negative points on nearby people. Expand the box if a
limb lies outside it. Do not mark an occluding person's pixels as positive:
segmentation can recover visible pixels, but hidden anatomy needs completion.

```json
{
  "people": {
    "0": {
      "box": [100, 40, 700, 1000],
      "points": [[300, 300], [620, 550], [680, 350]],
      "labels": [1, 1, 0]
    }
  }
}
```

Use `--segmentation-prompts prompts.json` with a new output directory. The
numbers above are illustrative; use coordinates from your image. Keys are
zero-based SAT-HMR detection indices in the output order; `1` includes a point,
`0` excludes it. A correction can supply only a box or only points and labels.
Keep disconnected visible pieces: a hand behind another person can be correctly
segmented as a separate component. Saved parameters record the prompts and
completion policy for review.

For small parts in large photographs, a person entry may also contain `regions`:
`[{"box": [600, 400, 800, 650], "points": [[700, 500]], "labels": [1]}]`.
These boxes select local crops for a higher-detail SAM pass. Points still use
original-image coordinates. An optional `prompt_box` inside a region specifies
a tighter object box, also in original pixels, while retaining the larger crop
for context. Use negative points on nearby faces, clothing and background.
Their masks are added to the main mask, so each crop
must be checked for inclusion of a neighboring person. Region prompts are manual
corrections, not an automatic guarantee of anatomically correct segmentation.

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
