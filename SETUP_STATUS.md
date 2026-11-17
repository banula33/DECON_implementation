# Real-image setup status — 17 November 2026

The licensed body-model blocker is resolved. Full real-image reconstruction
has not yet been validated.

## Models installed and checked

Runtime models are in:

`vendor/DECON/Multi-View_Synthesis/Geo_Gui_Hm_Decou/Human_SMPL_Estimation/weights/smpl_data/smpl/`

- `SMPL_FEMALE.pkl`: supplied SMPL 1.1.0 female model.
- `SMPL_MALE.pkl`: supplied SMPL 1.1.0 male model.
- `SMPL_NEUTRAL.pkl`: supplied SMPLify 1.0.0 neutral model, as specified by
  the upstream setup instructions.
- Public support files: `body_verts_smpl.npy`,
  `J_regressor_h36m_correct.npy`, and `smpl_mean_params.npz`.

The licensed source folders `SMPL_python_v.1.1.0/` and
`mpips_smplify_public_v2/` are preserved and have Git ignore entries.
The SMPLify folder was already tracked when this work started; ignore entries
do not untrack those files or remove them from existing history.
Runtime copies were converted from legacy Chumpy objects to NumPy arrays
using `scripts/convert_smpl_models.py`. Converted Chumpy array values were
checked against the originals. All three models load without Chumpy and
produce finite `(1, 6890, 3)` meshes. The neutral model passed CPU and CUDA
pose, shape, and translation gradient checks. The existing suite now passes 25 tests, including bounded-rendering checks.

The conversion used verified Chumpy 0.70 sources extracted under
`/tmp/decon-chumpy/chumpy-0.70`, with system Python's NumPy/SciPy.
Converted copies are also retained in `outputs/setup/converted-smpl/`.
The converter requires a new destination and does not overwrite source files.
SciPy is now included in the project's official dependency list because
SMPL model pickles contain sparse matrices.

## GPU and Python environments

- NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB VRAM, driver 580.178.04.
- `.venv` remains the working CPU test environment.
- `.venv-gpu` now has PyTorch 2.8.0+cu128 and torchvision 0.23.0+cu128.
  A CUDA tensor operation passed on the actual GPU.
- `nvidia-smi` and CUDA require execution outside the Codex sandbox here.
  Failure inside the sandbox does not imply a driver problem.
- NumPy 1.26.4, SciPy, smplx, Hydra, iopath, and supporting build packages
  are installed in `.venv-gpu`. SAT-HMR dependencies and xFormers are now
  downloading; the pinned PyTorch3D source has been fetched and its GPU wheel is building.
- The body-model CPU smoke check used `.venv` with
  `PYTHONPATH=/usr/lib/python3/dist-packages` to access the already installed
  system NumPy/SciPy while dedicated dependencies downloaded.
- The CUDA body-model smoke check used `.venv-gpu` with that same system
  dependency path, plus `.venv`'s installed smplx package. It verified actual
  GPU execution, but does not establish a self-contained GPU environment.
- PyTorch3D 0.7.8 was compiled for CUDA 12.8 and the RTX 5050 (`sm_120`).
  CUDA rasterization and translation gradients passed. A synthetic SMPL
  forward/backward check at the photo's 6000 × 3376 output dimensions used
  about 1.14 GiB peak allocated GPU memory with a 1024-pixel raster edge.
  All 25 tests pass in both CPU and GPU environments.
- SAT-HMR inference and PowerPaint loading/inference remain unvalidated.
  The separate GPU environment is not yet ready for the full pipeline.

## Checkpoints

- SAT-HMR `sat_644.pth` downloaded, revision
  `6b40d93a8ffec552a2098b15996e116d9e32efe5`.
- SAM 2.1 Hiera large has finished downloading and passed GPU inference on
  the supplied photo. PowerPaint v2 downloads remain incomplete and resumed
  from partial files after connection resets/timeouts.
- PowerPaint revision: `49ce8271db56ea91e7e9412789ba7faa2e7c83f1`.
- Download progress/errors: `outputs/setup/checkpoint-downloads.log`.
- Resume script: `outputs/setup/download_public_checkpoints.py`.
  Do not launch a second copy while a download is still running.
- Files ending in `.partial` must not be used for inference.

The published PowerPaint U-Net file is named
`diffusion_pytorch_model-002.safetensors`. After download, create a
`diffusion_pytorch_model.safetensors` symlink in the same directory for the
loader. PowerPaint's additional `runwayml/stable-diffusion-v1-5` base-model
cache has not been prepared. Loading and memory use on this 8 GB GPU remain
untested.

## Remaining acceptance work

The supplied football photo is staged under `outputs/football/input/`.
SAM-only diagnostic previews are available under `outputs/football/sam_preview/`;
see `outputs/football/STATUS.md` for their limitations.
After dependencies and checkpoints are ready, run the CLI from README.md.
Inspect separate person masks, camera projections, optimized silhouettes,
completion quality, resolution, and preservation of visible pixels.
No full reconstruction or quality claim has been made.

Multi-view synthesis, NeuS, and PAPO remain downstream work after decoupling
is validated on real images.
