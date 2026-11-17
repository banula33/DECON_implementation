"""Per-person segmentation: YOLOv8-pose person boxes + limb keypoints -> SAM 2.1.

Fallback for when SAT-HMR/SMPL/PyTorch3D are unavailable. It approximates the
DECON geometry-guided prompts with 2D pose keypoints instead of SMPL geometry:
  * each detected person is segmented from its box prompt;
  * merged "group" boxes are dropped and only used to recover leftover limbs;
  * confident wrist/elbow keypoints (``--auto-limbs``) and manual
    ``--prompts`` points segment local parts that are forced onto their owner,
    so a hand separated from its body (e.g. behind another person) is kept;
  * overlapping pixels are owned exclusively (highest SAM logit).

Usage:
    python scripts/segment_people.py data/football.jpg --output outputs/football
    python scripts/segment_people.py data/ --output outputs/batch          # folder
    python scripts/segment_people.py img.jpg --prompts p.json --output out  # manual fixes
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
          (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60)]
LIMB_KPTS = [7, 8, 9, 10]  # COCO: elbows, wrists
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ioa(inner, outer):
    ix = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return ix * iy / max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))


class Segmenter:
    def __init__(self, args):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from ultralytics import YOLO

        self.args = args
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = YOLO(args.yolo)
        self.predictor = SAM2ImagePredictor(
            build_sam2(args.sam_config, args.sam_checkpoint, device=self.device))
        print(f"Device: {self.device}"
              + (f" ({torch.cuda.get_device_name(0)})" if self.device == "cuda" else ""))

    def _sam(self, **kw):
        if self.device == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.predictor.predict(**kw)
        return self.predictor.predict(**kw)

    @torch.inference_mode()
    def run(self, image_path: Path, out: Path, prompts: dict) -> list[dict]:
        a = self.args
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        h, w = rgb.shape[:2]

        det = self.yolo(rgb, classes=[0], conf=a.conf, device=self.device, verbose=False)[0]
        boxes = det.boxes.xyxy.cpu().numpy()
        scores = det.boxes.conf.cpu().numpy()
        kpts = (det.keypoints.data.cpu().numpy() if det.keypoints is not None
                else np.zeros((len(boxes), 17, 3)))
        if len(boxes) == 0:
            print(f"{image_path.name}: no people detected")
            return []
        order = np.argsort(-(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
        boxes, scores, kpts = boxes[order], scores[order], kpts[order]

        is_group = np.array([sum(ioa(boxes[j], boxes[i]) > 0.85
                                 for j in range(len(boxes)) if j != i) >= 2
                             for i in range(len(boxes))])
        group_boxes = boxes[is_group]
        boxes, scores, kpts = boxes[~is_group], scores[~is_group], kpts[~is_group]

        # Drop partial duplicate boxes (e.g. an upper-body box inside a full-body box
        # for the same person with lower confidence).
        keep = [i for i in range(len(boxes))
                if not any(j != i and scores[j] > scores[i] and ioa(boxes[i], boxes[j]) > 0.7
                           for j in range(len(boxes)))]
        boxes, scores, kpts = boxes[keep], scores[keep], kpts[keep]
        n = len(boxes)

        padded = []
        for x0, y0, x1, y1 in boxes:
            px, py = (x1 - x0) * a.pad, (y1 - y0) * a.pad
            padded.append([max(0, x0 - px), max(0, y0 - py),
                           min(w - 1, x1 + px), min(h - 1, y1 + py)])
        padded = np.array(padded, dtype=np.float32)

        self.predictor.set_image(rgb)
        logits = []
        for box in padded:
            _, _, low = self._sam(box=box, multimask_output=False, return_logits=True)
            m, _, _ = self._sam(box=box, mask_input=low, multimask_output=False, return_logits=True)
            logits.append(np.asarray(m[0], np.float32))
        logits = np.stack(logits)
        group_masks = [self._sam(box=b.astype(np.float32), multimask_output=False)[0][0] > 0
                       for b in group_boxes]

        fg = logits > 0
        owner = np.where(fg.any(0), np.argmax(np.where(fg, logits, -np.inf), 0), -1)

        # Leftover group-mask regions go to the person they touch most.
        kernel = np.ones((7, 7), np.uint8)
        for gm in group_masks:
            nc, labels = cv2.connectedComponents((gm & (owner < 0)).astype(np.uint8))
            for c in range(1, nc):
                comp = labels == c
                ring = cv2.dilate(comp.astype(np.uint8), kernel).astype(bool) & ~comp
                contact = [int((ring & (owner == i)).sum()) for i in range(n)]
                if max(contact) > 0:
                    owner[comp] = int(np.argmax(contact))

        # Part prompts: manual (strong) and automatic limb keypoints.
        pos = {i: [] for i in range(n)}  # (x, y, source)
        if a.auto_limbs:
            for i in range(n):
                for k in LIMB_KPTS:
                    x, y, c = kpts[i, k]
                    if c >= a.kpt_conf and 0 <= x < w and 0 <= y < h:
                        pos[i].append((int(x), int(y), "auto"))
        manual_neg = {}
        for i, spec in prompts.items():
            if i < n:
                pos[i] += [(int(x), int(y), "manual") for x, y in spec.get("positive", [])]
                manual_neg[i] = spec.get("negative", [])

        part_log = []
        for src in ("auto", "manual"):  # manual applied last so it wins
            for i in range(n):
                others = [(x, y) for j in range(n) if j != i for x, y, _ in pos[j]]
                others += [tuple(map(int, kp[:2])) for j in range(n) if j != i
                           for kp in kpts[j] if kp[2] >= a.kpt_conf]
                neg = others + [tuple(p) for p in manual_neg.get(i, [])]
                for x, y, s in pos[i]:
                    if s != src:
                        continue
                    # Auto points already inside this person's mask need no help.
                    if s == "auto" and owner[y, x] == i:
                        continue
                    coords = np.array([(x, y)] + neg, np.float32)
                    labels = np.array([1] + [0] * len(neg), np.int32)
                    masks, sc, _ = self._sam(point_coords=coords, point_labels=labels,
                                             multimask_output=True)
                    cap = (0.08 if s == "manual" else 0.03) * h * w
                    ok = [k for k in range(len(masks))
                          if 0 < masks[k].sum() < cap and masks[k][y, x] > 0]
                    if not ok:
                        part_log.append({"person": i, "point": [x, y], "source": s, "used": False})
                        continue
                    part = masks[max(ok, key=lambda k: sc[k])] > 0
                    if s == "auto":
                        # Don't steal another person's confident core pixels.
                        part &= ~((owner >= 0) & (owner != i) & (logits.max(0) > a.steal_logit))
                    owner[part] = i
                    part_log.append({"person": i, "point": [x, y], "source": s, "used": True,
                                     "pixels": int(part.sum())})

        return self._export(rgb, out, boxes, padded, scores, owner, fg, pos, manual_neg, part_log)

    def _export(self, rgb, out, boxes, padded, scores, owner, fg, pos, manual_neg, part_log):
        out.mkdir(parents=True, exist_ok=False)
        overlay = rgb.astype(np.float32).copy()
        summary = []
        for i in range(len(boxes)):
            mask = owner == i
            d = out / f"person_{i:03d}"
            d.mkdir()
            Image.fromarray((mask * 255).astype(np.uint8)).save(d / "segmentation.png")
            seg = np.full_like(rgb, 255)
            seg[mask] = rgb[mask]
            Image.fromarray(seg).save(d / "segmented.png")
            Image.fromarray(np.dstack([rgb, (mask * 255).astype(np.uint8)]), "RGBA").save(
                d / "segmented_rgba.png")
            ys, xs = np.nonzero(mask)
            if len(xs):
                Image.fromarray(seg[ys.min():ys.max() + 1, xs.min():xs.max() + 1]).save(
                    d / "bounding_box.png")
            c = np.array(COLORS[i % len(COLORS)], np.float32)
            overlay[mask] = overlay[mask] * 0.45 + c * 0.55
            info = {"person": i, "detection_score": round(float(scores[i]), 3),
                    "box_xyxy": padded[i].round(1).tolist(), "mask_pixels": int(mask.sum()),
                    "overlap_pixels_removed": int((fg[i] & ~mask).sum()),
                    "part_prompts": [p for p in part_log if p["person"] == i]}
            (d / "segmentation.json").write_text(json.dumps(info, indent=2))
            summary.append(info)

        ov = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(ov)
        for i, b in enumerate(padded):
            draw.rectangle(b.tolist(), outline=COLORS[i % len(COLORS)], width=3)
            draw.text((b[0] + 4, b[1] + 4), f"person_{i:03d}", fill=(255, 255, 255))
            for x, y, s in pos[i]:
                r = 4 if s == "manual" else 3
                draw.ellipse([x - r, y - r, x + r, y + r],
                             fill=(0, 255, 0) if s == "manual" else (255, 255, 0), outline=(0, 0, 0))
        for pts in manual_neg.values():
            for x, y in pts:
                draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 0, 0), outline=(0, 0, 0))
        ov.save(out / "segmentation_overlay.png")
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", help="image file or folder of images")
    p.add_argument("--output", required=True)
    p.add_argument("--yolo", default="yolov8m-pose.pt", help="YOLOv8 (pose) weights")
    p.add_argument("--sam-checkpoint", default="checkpoints/sam2.1_hiera_base_plus.pt")
    p.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--conf", type=float, default=0.4, help="person detection threshold")
    p.add_argument("--pad", type=float, default=0.03, help="box padding fraction")
    p.add_argument("--no-auto-limbs", dest="auto_limbs", action="store_false",
                   help="disable wrist/elbow keypoint prompts")
    p.add_argument("--kpt-conf", type=float, default=0.6, help="keypoint confidence for auto prompts")
    p.add_argument("--steal-logit", type=float, default=4.0,
                   help="auto prompts never take pixels another person claims above this logit")
    p.add_argument("--prompts", help='JSON {"<person>": {"positive": [[x,y]], "negative": [[x,y]]}}'
                                     " (single image only)")
    args = p.parse_args()

    src, out = Path(args.input), Path(args.output)
    images = sorted(f for f in src.iterdir() if f.suffix.lower() in IMAGE_EXT) if src.is_dir() else [src]
    if args.prompts and len(images) != 1:
        raise SystemExit("--prompts only applies to a single image")
    prompts = ({int(k): v for k, v in json.loads(Path(args.prompts).read_text()).items()}
               if args.prompts else {})

    seg = Segmenter(args)
    for img in images:
        t = time.perf_counter()
        dest = out / img.stem if src.is_dir() else out
        summary = seg.run(img, dest, prompts)
        print(f"{img.name}: {len(summary)} people -> {dest} ({time.perf_counter() - t:.1f}s)")


if __name__ == "__main__":
    main()
