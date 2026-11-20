import json
from pathlib import Path

import torch
from PIL import Image

from decon.artifacts import save_results
from decon.body_model import SMPLBodyModel
from decon.ggdo import GGDOConfig
from decon.official_adapters import (
    OfficialSATHMRGeometryEstimator,
    OfficialSAM2Segmenter,
    PyTorch3DSilhouetteRenderer,
)
from decon.pipeline import HumanDecouplingPipeline


root = Path("vendor/DECON/Multi-View_Synthesis/Geo_Gui_Hm_Decou")
image_path = Path("outputs/football/input/football.jpg")
output = Path("outputs/refinement/football_segmentation_rerun")
image = Image.open(image_path).convert("RGB")
prompts = json.loads(Path("outputs/refinement/football_cli_prompts.json").read_text())["people"]
body = SMPLBodyModel(root / "Human_SMPL_Estimation/weights/smpl_data")
pipeline = HumanDecouplingPipeline(
    OfficialSATHMRGeometryEstimator(root / "Human_SMPL_Estimation"),
    OfficialSAM2Segmenter(
        root / "Img_Segmentation",
        checkpoint=root / "Img_Segmentation/checkpoints/sam2.1_hiera_large.pt",
    ),
    PyTorch3DSilhouetteRenderer(body.faces, (image.height, image.width), max_render_size=1024),
    body_model=body,
    segmentation_prompts=prompts,
    inpainter=None,
    ggdo_config=GGDOConfig(translation_steps=100, pose_shape_steps=300),
)
people = pipeline.run(image)
save_results(image, people, output, body.faces)
print(f"Saved {len(people)} segmentation and geometry results to {output}")
