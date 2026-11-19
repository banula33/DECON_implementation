# Real-image setup status — 19 November 2026

The real-image pipeline has completed SAT-HMR, geometry-prompted SAM 2, and
both GGDO optimization stages for the three foreground players in the supplied
football photo. PowerPaint has generated all three completed images, with exact
visible-mask pixel preservation and original resolution. Visual acceptance failed
on the central players because of hand and edge artifacts.

## Installed and verified

- Licensed female/male SMPL 1.1.0 models and SMPLify's neutral 1.0.0 model,
  converted from Chumpy to NumPy without changing the array values.
- The three public SAT-HMR support arrays.
- Original licensed source folders are preserved. Both have ignore entries,
  but the SMPLify folder was already tracked; ignore rules do not remove history.
- `.venv` remains the CPU environment. `.venv-gpu` uses PyTorch 2.8.0+cu128,
  torchvision 0.23.0+cu128, NumPy 1.26.4, and xFormers 0.0.32.post2.
- PyTorch3D 0.7.8, commit `75ebeeaea0908c5527e7b1e305fbc7681382db47`,
  was compiled for CUDA 12.8 and RTX 5050 (`sm_120`). GPU rasterization and
  gradients work. The 30-test suite passes in the GPU environment; the CPU environment passes
  28 tests with one optional test module skipped.
- SAT-HMR and SAM 2 dependencies are installed. PowerPaint dependencies are
  installed, and its controller imports successfully.
- Transformers 4.37.2 supplies Python 3.12-compatible tokenizers;
  huggingface-hub 0.25.2 retains APIs needed by Diffusers 0.27.0.
  Exact runtime constraints are in `outputs/setup/gpu-constraints.txt`.
- SAT-HMR checkpoint SHA-256 and xFormers wheel SHA-256 were checked against
  their upstream digests.

CUDA access requires execution outside the Codex sandbox on this machine.
Inside-sandbox CUDA failures do not imply a broken driver.

## Real photo results

Input: `/home/kavin/Downloads/pexels-franco-monsalvo-252430633-32179295.jpg`
(6000 × 3376). The original image is unchanged.

SAT-HMR detected four people. The geometry pass selected the three largest
projected boxes to focus on the foreground players. Results:

| Player | Geometry folder | Initial silhouette MSE | Fitted silhouette MSE |
| --- | --- | ---: | ---: |
| Central blue | `outputs/football/geometry/person_000` | 0.017611 | 0.014439 |
| Central white | `outputs/football/geometry/person_001` | 0.014382 | 0.006033 |
| Right blue, refined | `outputs/football/right_refined/person_000` | 0.018011 | 0.005886 |

Each folder includes a mesh, parameters, masks, and before/after overlays.
The original right-player result is preserved as `geometry/person_002`.
Its tight box missed lower-leg pixels; a visually reviewed 20% projected-box
margin recovered the legs and feet, so the right player was refitted.
The other two players retain the default 2.5% margin. This is per-image
prompt tuning, not an automatically validated general box-selection rule.

The geometry pass uses 100 translation steps and 300 pose/shape steps.
Rasterization is capped at 1024 pixels on the longest edge, with scaled camera
intrinsics and differentiable restoration to full-resolution masks. A synthetic
SMPL forward/backward memory check at the photo's output size used about
1.14 GiB of allocated GPU memory. Fine silhouette boundaries are approximate.

The SAT-HMR bridge now moves target metadata to the GPU alongside the image;
without this, the real run failed during camera-intrinsics calculation.

Lower silhouette loss does not establish anatomical correctness. The central
blue player remains substantially occluded, and SAM omits some disconnected
visible arm regions. The completed central blue player has detached hand shapes and dark contours;
the white player has dark edge artifacts. The right player is cleaner. These
outputs are not yet accepted for multi-view reconstruction.

## Completion results and memory fix

All three completed images are available in `outputs/football/completion/`:
`person_000_completed.png`, `person_001_completed.png`, and
`person_002_completed.png`. Each is 6000 × 3376. `review.jpg` shows cropped
previews together. `quality_review.json` records the failed visual acceptance.

The kernel journal confirms that the original worker (PID 142574) was killed by
the system OOM killer on 17 November at 17:58:32 +05:30, during model loading.
It was not merely waiting for more files. All checkpoints and base components
have now downloaded, and the controller imports and performs inference.

`decon/powerpaint_loader.py` avoids redundant initialized UNet copies and loads
complete BrushNet weights into an empty model after checking keys and shapes.
Components materialize on the GPU before CPU offloading is enabled. The official
controller prediction method and trained weights are retained. The new loader
also avoids the upstream move back to CUDA immediately after enabling offloading.

Measured comparison:

- Reference run, original loader, three players: peak RSS 10.45 GiB.
- Revised loader, first player regenerated with seed 42: peak RSS 7.53 GiB.
- The regenerated image is pixel-for-pixel identical to the reference first player.
- The two runs have different image counts; this is a recorded run comparison,
  not a general memory guarantee for all workloads or machine states.

Evidence: `outputs/football/completion_memory_v2/comparison.json`,
`outputs/football/logs/completion_retry.log`, and
`outputs/football/logs/completion_memory_v2.log`.

PowerPaint weights use revision `49ce8271db56ea91e7e9412789ba7faa2e7c83f1`.
The canonical U-Net symlink exists. For local cached base models, use
`HF_HOME=/home/kavin/DECON_implementation/checkpoints/huggingface`.
No downloads or completion jobs need to remain running for these saved results.
Historical PID files may refer to exited workers; the current result status is
`outputs/football/completion/status.json`.

## Remaining validation

The official repository was fetched and checked against upstream on 19 November;
the local checkout already matches commit `992364634b73bba05d68d5930ff20627e0487b4b`.
All downstream source directories are present, but the trained multi-view stage
checkpoints are missing and no download link is provided in the checked source.
See [OFFICIAL_ASSETS.md](OFFICIAL_ASSETS.md) for the inventory and exact filenames.

Two additional AI-generated photos (walking and overlapping dancers) have now
completed the full CLI with two people each. All four outputs preserve visible
pixels exactly and retain 1536 × 1024 resolution; silhouette MSE decreases by
27.1%, 28.0%, 6.3%, and 51.5%. Both cases fail visual acceptance: completion adds
dark fragments/strips, and the rear dancer's reconstructed foot is misaligned.
Review sheets, exact generation prompts, measurements and findings are saved in
`outputs/synthetic_validation/REPORT.md`. These synthetic images have no
ground-truth meshes and do not replace real-photo accuracy validation.

The initial integrated runs exposed CUDA OOM during PowerPaint loading while SAM
was resident. The pipeline now fits all people, releases SAM and its cached image
embeddings, then performs completion. Both full reruns succeeded; the GPU suite
still passes all 30 tests, including the segmentation/release/completion order.

Improve disconnected-arm segmentation and completion-mask geometry, and evaluate
person-centered inpainting crops separately. Additional images with different
poses/occlusion are still needed for broader real-photo validation; the football
photo remains the only real-photo case processed. Multi-view synthesis, NeuS, and PAPO remain downstream work
until visual acceptance improves.
