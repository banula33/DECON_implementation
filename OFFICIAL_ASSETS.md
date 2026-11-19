# Official DECON assets — checked 19 November 2026

Fetched `main` from https://github.com/IridescentJiang/DECON.git. The existing
`vendor/DECON` checkout already matches upstream commit
`992364634b73bba05d68d5930ff20627e0487b4b`; no source replacement was necessary.
Existing model assets and local adapter changes were preserved.

## Available locally

Paths below are relative to `vendor/DECON`.

| Component | Source directory | Status |
|---|---|---|
| SAT-HMR | `Multi-View_Synthesis/Geo_Gui_Hm_Decou/Human_SMPL_Estimation` | Source, inference checkpoint, licensed SMPL and support arrays present |
| SAM 2 | `Multi-View_Synthesis/Geo_Gui_Hm_Decou/Img_Segmentation` | Source and SAM 2.1 large checkpoint present |
| PowerPaint | `Multi-View_Synthesis/Geo_Gui_Hm_Decou/Img_Inpainting` | Source, v2 weights and cached base components present; inference already exercised |
| GGDO | `Multi-View_Synthesis/src/ggdo` | Official source present; local adapter implements fitting |
| Multi-view synthesis | `Multi-View_Synthesis` | Source present; trained stage checkpoints missing |
| NeuS reconstruction | `Multi-View_Synthesis/src/recon` | Source present; downstream integration not validated |
| PAPO | `Multi-View_Synthesis/src/papo` | Source present; downstream integration not validated |
| SMPL part replacement | `Post-processing/SMPL_parts_replacement` | Source present |
| Mesh refinement | `Post-processing/mesh_refinement` | Source present |

The current-stage preflight confirms CUDA, SMPL-X, PyTorch3D, NumPy and Pillow
are available, with no missing SMPL/support files. The recent two-photo runs
exercise decoupling end to end, but fail visual acceptance; see
`outputs/synthetic_validation/REPORT.md`.

## Missing multi-view assets

The checked repository has no GitHub releases or tags and its README does not
provide a DECON-trained multi-view checkpoint download. The checked-in
`Multi-View_Synthesis/configs/inference/inference_wild.yaml` instead refers to
absolute paths on the training machine. Its corresponding `inference_wild.py`
expects:

- Stage 1: `denoising_unet-31000.pth`, `reference_unet-31000.pth`,
  `normal_guider-31000.pth`, `depth_guider-31000.pth`.
- Stage 2: `denoising_unet-16695.pth`.
- Base components: `CosmicMan-SD`, `sd-vae-ft-mse`, and
  `sd-image-variations-diffusers/image_encoder` under `pretrained_weights`.

These filenames describe the wild-image config, not a confirmed recommended
release. Other supplied configs reference other steps and architectures.
Obtain the authors' intended inference configuration and matching trained
checkpoints together. Generic base models or MagicMan weights should not be
represented as DECON-trained replacements. Base components alone cannot unblock
this stage, so no speculative bulk download or environment downgrade was made.

Evidence: `outputs/setup/decon_source_audit.json` and
`outputs/setup/decon_remote_assets.json`. Upstream source references:
[README](https://github.com/IridescentJiang/DECON/blob/992364634b73bba05d68d5930ff20627e0487b4b/README.md),
[wild config](https://github.com/IridescentJiang/DECON/blob/992364634b73bba05d68d5930ff20627e0487b4b/Multi-View_Synthesis/configs/inference/inference_wild.yaml),
[weight loader](https://github.com/IridescentJiang/DECON/blob/992364634b73bba05d68d5930ff20627e0487b4b/Multi-View_Synthesis/inference_wild.py).
