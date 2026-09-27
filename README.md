# DECON Implementation Guide

## Current implementation

The first implementation slice is available in `decon/`:

```text
decon/
├── adapters.py   # SAT-HMR, SAM 2, renderer, and PowerPaint interfaces
├── ggdo.py       # two-stage differentiable mask optimization
├── pipeline.py   # estimate -> segment -> GGDO -> inpaint
└── types.py      # typed per-person inputs and outputs
```

Install the base dependency with `pip install -r requirements.txt`. The
external model adapters are intentionally injected: SAT-HMR, SAM 2, and
PowerPaint each require their own checkpoints and installation instructions.
The numerical GGDO contract is covered by `tests/test_ggdo.py` using a small
differentiable renderer. A production renderer should implement
`MeshRenderer.render_mask` with PyTorch3D and return a soft `[height, width]`
mask so gradients can reach translation and SMPL parameters.

The current optimizer uses a vertex-space delta for the second stage. This is
an adapter-neutral stand-in for SMPL pose and shape optimization; the SAT-HMR
adapter should expose pose/shape parameters when it is connected so those
parameters can replace the delta directly.

## Official model adapters

`decon.official_adapters` connects the pipeline to the official DECON
checkout. The checkout cloned for this workspace is `../DECON_official`.
The adapters are lazy, so importing `decon` does not require all model
packages or checkpoints to be installed.

Required official assets:

- SAT-HMR: place the official SAT-HMR weights and SMPL files under
    `Multi-View_Synthesis/Geo_Gui_Hm_Decou/Human_SMPL_Estimation/weights`.

After downloading the licensed SMPL files and SAT-HMR `.pth` checkpoint into
one local folder, prepare the official layout with:

```powershell
python scripts\prepare_sat_hmr_weights.py C:\path\to\downloaded\weights
```

The script renames the three SMPL files to `SMPL_FEMALE.pkl`,
`SMPL_MALE.pkl`, and `SMPL_NEUTRAL.pkl`, copies SAT-HMR `.pth` files into
`weights/sat_hmr`, and refuses to overwrite existing files.
- SAM 2: install the dependencies described in its `INSTALL.md` and download
    the checkpoint with `Img_Segmentation/checkpoints/download_ckpts.sh`.
- PowerPaint: install its environment from `Img_Inpainting` and download the
    `ppt-v2` checkpoint directory described in the official README.
- PyTorch3D: install a build matching the installed PyTorch and CUDA versions.

Example wiring:

```python
from pathlib import Path

from decon import (
        HumanDecouplingPipeline,
        OfficialPowerPaintInpainter,
        OfficialSATHMRGeometryEstimator,
        OfficialSAM2Segmenter,
        PyTorch3DSilhouetteRenderer,
)

official = Path("../DECON_official/Multi-View_Synthesis/Geo_Gui_Hm_Decou")
sat = OfficialSATHMRGeometryEstimator(official / "Human_SMPL_Estimation")
sam = OfficialSAM2Segmenter(official / "Img_Segmentation")
paint = OfficialPowerPaintInpainter(
        official / "Img_Inpainting", official / "Img_Inpainting/checkpoints/ppt-v2"
)
# Load SAT-HMR's SMPL face tensor and create the original-view camera here.
renderer = PyTorch3DSilhouetteRenderer(faces, (image_height, image_width))
pipeline = HumanDecouplingPipeline(sat, sam, renderer, paint)
persons = pipeline.run(input_image_path)
```

SAT-HMR's official `demo.yaml` expects the input image in its `demo/` folder
and writes `demo_results/`; the adapter follows that contract and parses its
`*_smpl_para.json` and `*_smpl_mesh.obj` outputs. PowerPaint v2 and the
official renderer require a CUDA device.

Run the local setup check before downloading or installing model assets:

```powershell
python scripts/check_official_setup.py
```

This reports checkpoint presence, PyTorch/CUDA status, and the exact reason
inference cannot run on a CPU-only installation.

This guide provides a step-by-step roadmap to implement the **DECON** (DEcouple-and-reCONstruct) framework from scratch, based on the provided research paper. 

DECON reconstructs clothed-geometric multiple humans from a single image. The pipeline is broken down into three main stages: Decoupling, Reconstruction, and Position Optimization.

> **Tip:** The original authors plan to release their code at [https://github.com/IridescentJiang/DECON](https://github.com/IridescentJiang/DECON). It is highly recommended to reference their implementation details (like hyperparameters and training configurations) once it becomes available.

---

## Phase 1: Environment & Foundation Models Setup

DECON heavily relies on several state-of-the-art pre-trained models. Your first step is to set up a robust environment and integrate these components.

1.  **Core Environment**:
    *   Python 3.9+
    *   PyTorch and torchvision
    *   [PyTorch3D](https://pytorch3d.org/) (Crucial for differentiable rendering in later steps)
2.  **Foundation Models to Integrate**:
    *   **Geometry Estimator**: [SAT-HMR](https://github.com/Zhengyuliu/SAT-HMR) (estimates SMPL parameters from multi-human images).
    *   **Segmentation**: [SAM 2](https://github.com/facebookresearch/segment-anything-2) (Segment Anything Model 2).
    *   **Inpainting**: [PowerPaint](https://github.com/open-mmlab/PowerPaint) (for completing occluded human parts).

---

## Phase 2: Geometry-Guided Human Decoupling

This stage isolates individuals from the input image and fills in missing visual information caused by occlusions.

### Step 2.1: Initial Geometry Estimation & Segmentation
1.  Pass the input RGB image through **SAT-HMR** to obtain initial SMPL parameters (pose $\theta$, shape $\beta$, and camera/translation $t$) for all individuals.
2.  Extract 2D bounding boxes from the projected SMPL models.
3.  Use these bounding boxes as prompts for **SAM 2** to extract the initial 2D segmented mask ($m_s$) for each person.

### Step 2.2: Geometry-Guided Decoupling Optimization (GGDO)
Because the initial SMPL prediction might not perfectly align with the image, you need to optimize it.
1.  Set up a differentiable renderer using **PyTorch3D**.
2.  Render the current SMPL mesh to generate a 2D geometric mask ($m_{geo}$).
3.  **Stage 1 (Translation):** Freeze pose and shape. Optimize the translation parameter $t$ using Mean Squared Error (MSE) loss between $m_{geo}$ and $m_s$.
4.  **Stage 2 (Pose & Shape):** Unfreeze and optimize pose $\theta$ and shape $\beta$ using the same MSE loss to tightly align the SMPL silhouette with the SAM 2 mask.

### Step 2.3: Image Inpainting
1.  Use the optimized SMPL mask ($m_{geo}$) to define the exact region where the human should exist.
2.  Identify missing regions (where $m_{geo}$ exists but the SAM 2 mask $m_s$ does not).
3.  Pass the segmented image and the missing region mask to **PowerPaint** to generate a fully complete, unoccluded image of each individual.

---

## Phase 3: Generative Multi-View Synthesis Reconstruction

Now that you have isolated, complete images of each person, you need to reconstruct their 3D geometry.

### Step 3.1: Multi-View Synthesis
1.  Implement a diffusion-based novel-view synthesizer (similar to [Wonder3D](https://github.com/xxlong0/Wonder3D) or [MagicMan](https://magicman-project.github.io/)).
2.  Condition the generation on:
    *   The decoupled, inpainted image from Phase 2 (as color/style reference).
    *   The optimized SMPL model (as geometric guidance using a ControlNet-style architecture).
3.  Generate images and normal maps of the individual from multiple viewpoints (e.g., 6 views).

### Step 3.2: 3D Mesh Reconstruction
1.  Use a Neural Implicit Surface framework like **NeuS** (integrated within Wonder3D's paradigm).
2.  Train the NeuS representation using the generated multi-view RGB images and normal maps to extract a high-fidelity 3D mesh with clothing details.

---

## Phase 4: Perspective-Aware Position Optimization (PAPO)

The final step is to assemble the individual 3D meshes back into a coherent scene, fixing floating or intersecting bodies.

1.  **Initialization**: Place each reconstructed 3D mesh at the optimized translation $t$ found in Phase 2 (GGDO).
2.  **Learnable Parameters**: Define learnable XYZ-axis offsets ($o_{xyz}$) for each subject.
3.  **Target Generation**: Create a target mask ($p_{tar}$) by performing a pixel-wise summation of all the original SAM 2 segmented masks from Step 2.1.
4.  **Differentiable Rendering**: Render the combined projection ($p_{ini}$) of all 3D meshes simultaneously using PyTorch3D from the original camera viewpoint.
5.  **Optimization Loop**: 
    *   Calculate the MSE loss between the combined projection $p_{ini}$ and the target mask $p_{tar}$.
    *   Backpropagate to update the $o_{xyz}$ offsets until the models correctly overlap/occlude each other according to the original 2D image.

---

## Suggested Git Repository Structure

When you initialize your repository, a good starting structure would be:

```text
DECON/
│
├── data/                  # Input images and ground truth
├── models/                # Checkpoints (SAM2, PowerPaint, SAT-HMR)
├── modules/               # Core pipeline components
│   ├── geometry_estimator.py # SAT-HMR wrapper
│   ├── segmenter.py          # SAM 2 wrapper
│   ├── inpainter.py          # PowerPaint wrapper
│   ├── ggdo.py               # PyTorch3D optimization logic
│   ├── multiview_synth.py    # Diffusion model for novel views
│   ├── neus_recon.py         # NeuS reconstruction
│   └── papo.py               # Position optimization logic
│
├── pipeline.py            # Main script chaining the modules together
├── requirements.txt       
└── README.md
```
