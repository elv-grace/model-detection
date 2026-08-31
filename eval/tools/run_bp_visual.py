#!/usr/bin/env python3
"""Visual-prompted YOLOE, one exemplar at a time (instance-level image prompting).

Why one at a time
-----------------
The 8-class sweep built a `logo` prototype by averaging 90 pool exemplars across 30 brands and
got F1 0.44, against 0.80 for text. Averaging distinct brands produces a prototype for "some
graphic thing" rather than for any brand, so that number says little about what visual
prompting is actually for. The real use case is the N=1 case: the user hands over ONE
reference image and asks for that thing.

So each exemplar gets its own complete run here. That also matches the intended API -- the
caller passes a list of image paths which may have length one -- and it produces a
distribution over exemplars rather than a single number, which is what tells you whether
visual prompting is reliable or merely lucky on a good reference.

Where exemplars come from
-------------------------
    brand   the model-logo pool (tight crops) plus the test-files/brand_*.jpg images
    person  the test-files/person_*.jpg images

Cropping exemplars out of the 100 frozen frames would be leakage -- the prompt would contain
the answers -- so every exemplar is external to the evaluation set. Brands and people that DO
appear in the footage are included deliberately, since that is the targeted case; ones that do
not are included so the false-positive rate is measured too.

Prompt boxes
------------
Pool images are tight crops, so the whole image is the box. The test-files images are not: a
photo of a Mercedes contains road, sky and background, and prompting with the whole frame
would build a prototype of the scene rather than of the car. For those, --box auto runs one
text-prompted pass over the exemplar and takes its highest-confidence box for the intended
class, falling back to the full image if nothing fires.

Usage
-----
    python eval/run_bp_visual.py --weights yoloe-11l-seg.pt
    python eval/run_bp_visual.py --weights yoloe-26l-seg.pt --pool-brands 6
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402
REPO = os.path.dirname(HERE)
POOL = os.environ.get("ELV_LOGO_POOL", "/pool")
WEIGHTS_DIR = os.environ.get("ELV_WEIGHTS_DIR", "/root/.cache/detection")
TEST_FILES = os.environ.get("ELV_TEST_FILES", os.path.join(REPO, "test-files"))

# Brands visible in the frozen frames, included on purpose: this is the targeted use case.
SEEDED = ["NBA", "Gap", "New Era Cap Company", "Nike", "Adidas"]

# Text prompt used to find the object inside a non-crop exemplar (--box auto).
AUTO_BOX_PROMPT = {"brand": ["car", "logo", "sports car"], "person": ["person"]}


def collect_exemplars(pool_brands: int, per_brand: int, seed: int):
    """[(cls, name, path, is_tight_crop)]"""
    out = []
    for path in sorted(glob.glob(os.path.join(TEST_FILES, "brand_*"))):
        out.append(("brand", os.path.splitext(os.path.basename(path))[0], path, False))
    for path in sorted(glob.glob(os.path.join(TEST_FILES, "person_*"))):
        out.append(("person", os.path.splitext(os.path.basename(path))[0], path, False))

    if pool_brands and os.path.isdir(POOL):
        import random
        brands = sorted(d for d in os.listdir(POOL) if os.path.isdir(os.path.join(POOL, d)))
        chosen = [b for b in SEEDED if b in brands]
        rest = [b for b in brands if b not in chosen]
        chosen += random.Random(seed).sample(rest, max(0, pool_brands - len(chosen)))
        for brand in chosen[:pool_brands]:
            files = sorted(f for f in os.listdir(os.path.join(POOL, brand))
                           if f.lower().endswith((".jpg", ".jpeg", ".png")))
            for name in files[:per_brand]:
                out.append(("brand", f"pool_{brand.replace(' ', '_')}_{name.split('.')[0]}",
                            os.path.join(POOL, brand, name), True))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="yoloe-11l-seg.pt")
    parser.add_argument("--tag", default=None, help="short name for the output dir")
    parser.add_argument("--pool-brands", type=int, default=6)
    parser.add_argument("--per-brand", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--box", default="auto", choices=["auto", "full"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.005)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--out", default=os.path.join(paths.EXPERIMENTS, "02_brand_person_101", "runs_visual"))
    args = parser.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    tag = args.tag or args.weights.replace("yoloe-", "yoloe").replace("l-seg.pt", "")
    with open(paths.FRAMES_JSON) as handle:
        frames = json.load(handle)["frames"]

    exemplars = collect_exemplars(args.pool_brands, args.per_brand, args.seed)
    print(f"{tag}: {len(exemplars)} exemplars, {len(frames)} frames")

    os.chdir(WEIGHTS_DIR)
    summary = []

    for index, (cls, name, path, tight) in enumerate(exemplars, 1):
        run_name = f"{tag}__{cls}__{name}"
        try:
            width, height = Image.open(path).size
            box = [0.0, 0.0, float(width), float(height)]

            if args.box == "auto" and not tight:
                # Locate the object inside the exemplar first, so the prototype is built from
                # the object rather than from the scene around it.
                finder = YOLOE(args.weights)
                terms = AUTO_BOX_PROMPT[cls]
                finder.set_classes(terms, finder.get_text_pe(terms))
                found = finder.predict(path, imgsz=args.imgsz, conf=0.10, verbose=False)
                boxes = found[0].boxes
                if boxes is not None and len(boxes):
                    best = int(boxes.conf.argmax())
                    box = [float(v) for v in boxes.xyxy[best].tolist()]

            model = YOLOE(args.weights)
            prompts = {"bboxes": np.array([box]), "cls": np.array([0])}
            model.predict(path, visual_prompts=prompts, predictor=YOLOEVPSegPredictor,
                          verbose=False)
            vpe = model.predictor.get_vpe(path)
            # YOLOE compares prompt embeddings to region embeddings by cosine, so the
            # prototype must be unit length exactly like a text PE.
            vpe = torch.nn.functional.normalize(vpe, p=2, dim=-1)
            model.set_classes([cls], vpe)

            # Clear the cached predictor. ultralytics reuses model.predictor across calls
            # (Model.predict: `if not self.predictor: self.predictor = ...`), so without this
            # the YOLOEVPSegPredictor installed by the VPE-extraction call above would be
            # reused for every frame -- and it still holds self.prompts / self.visuals
            # rasterised from the REFERENCE image, so it would keep matching against those
            # instead of the class embedding just installed by set_classes.
            #
            # It also keeps the comparison honest. Once the VPE is installed as a class
            # embedding, the only remaining difference between a visual run and a text run is
            # where that embedding came from -- the image encoder rather than MobileCLIP's
            # text tower. Backbone, detection head, NMS and post-processing are identical.
            # Leaving the VP predictor in place would compare two different inference paths,
            # confounding the prompt type with the code path.
            model.predictor = None

            out_path = os.path.join(args.out, run_name, "out.jsonl")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            written = 0
            with open(out_path, "w") as handle:
                for start in range(0, len(frames), 8):
                    chunk = frames[start : start + 8]
                    results = model.predict(
                        [os.path.join(paths.FRAMESET, f["frame"]) for f in chunk],
                        imgsz=args.imgsz, conf=args.conf, iou=0.7,
                        max_det=args.max_det, verbose=False)
                    for frame, result in zip(chunk, results):
                        h, w = result.orig_shape
                        if result.boxes is None:
                            continue
                        for xyxy, score in zip(result.boxes.xyxy.tolist(),
                                               result.boxes.conf.tolist()):
                            x1, y1, x2, y2 = xyxy
                            handle.write(json.dumps({"type": "tag", "data": {
                                "start_time": 0, "end_time": 0,
                                "source_media": os.path.join("/elv/test", frame["id"] + ".png"),
                                "tag": cls, "track": "",
                                "frame_info": {"frame_idx": frame["frame_idx"], "box": {
                                    "x1": round(max(0.0, x1 / w), 4),
                                    "y1": round(max(0.0, y1 / h), 4),
                                    "x2": round(min(1.0, x2 / w), 4),
                                    "y2": round(min(1.0, y2 / h), 4)}},
                                "additional_info": {
                                    "kind": "crop", "prompt": f"visual:{cls}",
                                    "score": round(float(score), 4), "detector": run_name,
                                    "exemplar": name, "exemplar_path": path,
                                    "exemplar_box": [round(v, 1) for v in box],
                                    "tight_crop": tight},
                            }}) + "\n")
                            written += 1
            summary.append({"run": run_name, "cls": cls, "exemplar": name,
                            "detections": written, "tight_crop": tight})
            print(f"  [{index}/{len(exemplars)}] {run_name}: {written} detections")
        except Exception as exc:
            print(f"  [{index}/{len(exemplars)}] {run_name} FAILED: {type(exc).__name__}: {exc}")
            summary.append({"run": run_name, "cls": cls, "exemplar": name,
                            "error": f"{type(exc).__name__}: {exc}"})

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"summary_{tag}.json")
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nwrote {args.out}/<run>/out.jsonl and {os.path.basename(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
