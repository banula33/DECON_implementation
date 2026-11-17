# Real-image setup status — 17 November 2026

Real-image validation is not complete. The existing CPU suite passes, but that
does not validate pretrained-model inference.

## Confirmed on this machine

- NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB VRAM, driver 580.178.04.
- `nvidia-smi` works outside the Codex sandbox. Its failure inside the sandbox
  is not evidence that the driver needs reinstalling.
- `.venv` is the working CPU test environment; `.venv-gpu` is a separate,
  incomplete GPU environment.
- Downloaded the three public support files from the SAT-HMR authors' linked
  folder and checked that NumPy can open them:
  `body_verts_smpl.npy`, `J_regressor_h36m_correct.npy`, `smpl_mean_params.npz`.

## Downloads started, not yet validated

CUDA PyTorch 2.8.0 and torchvision 0.23.0 are being installed into `.venv-gpu`
using the official CUDA 12.8 wheel index. The initial attempt timed out;
the retry uses two concurrent downloads and a 600-second HTTP timeout.
Check `outputs/setup/gpu-install.log` for the result. Other upstream packages
and a compatible PyTorch3D build still need installation after this succeeds.

Public model downloads were started for:

- SAT-HMR `sat_644.pth`, revision `6b40d93a8ffec552a2098b15996e116d9e32efe5`.
- SAM 2.1 Hiera large, official `092824` checkpoint release.
- PowerPaint v2, revision `49ce8271db56ea91e7e9412789ba7faa2e7c83f1`.
  Only safetensors variants are selected where interchangeable binary weights
  also exist; the BrushNet text encoder still requires its `.bin` file.

Files ending in `.partial` are incomplete and must not be used for inference.
The download script and PowerPaint manifest are saved under `outputs/setup/`.
Do not launch a second copy while the original download is running.

The published PowerPaint U-Net file is named
`diffusion_pytorch_model-002.safetensors`. Once downloaded, its directory needs
a `diffusion_pytorch_model.safetensors` symlink to that file for the default
Diffusers loader. PowerPaint's additional `runwayml/stable-diffusion-v1-5`
base-model cache has **not** been prepared. Loading and memory usage on this
8 GB GPU remain untested.

## Required from you

Obtain the licensed female and male models from <https://smpl.is.tue.mpg.de/>
and the neutral model from <https://smplify.is.tue.mpg.de/>. Registration and
license acceptance must be completed by you. Extract these files into a local
folder:

```text
basicModel_f_lbs_10_207_0_v1.0.0.pkl
basicModel_m_lbs_10_207_0_v1.0.0.pkl
basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
```

Then arrange them using:

```bash
.venv/bin/python scripts/prepare_sat_hmr_weights.py /path/to/extracted/models \
  --destination vendor/DECON/Multi-View_Synthesis/Geo_Gui_Hm_Decou/Human_SMPL_Estimation/weights
```

An input image of two moderately overlapping people is also needed for the
acceptance run. No personal photo has been selected or sent anywhere.

## Remaining validation

After the downloads and upstream dependency installation finish, verify CUDA
tensor operations, SAM 2 and PowerPaint loading, SAT-HMR inference with the
licensed assets, and the full CLI run described in README.md. Inspect masks,
camera projections, optimized geometry, completion quality, original image
resolution, and visible-pixel preservation before calling decoupling validated.
Multi-view synthesis, NeuS, and PAPO remain downstream work.
