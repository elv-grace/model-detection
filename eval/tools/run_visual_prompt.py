#!/usr/bin/env python3
"""Visual-prompted YOLOE: build a `logo` prototype from model-logo pool exemplars (step 6c).

The question this answers
-------------------------
Text prompting reached logo F1 0.80 (recall 0.74) once prompts were mined from YOLOE's own
vocabulary. Visual prompting is the alternative for targets that resist description — can
reference *images* of brand marks beat the word "logo"?

Method
------
The pool images are already tight crops of a logo filling the frame, so each exemplar's
visual-prompt box is the whole image. YOLOE returns a (1, 1, 512) prompt embedding per
exemplar; averaging them and re-normalising gives a single class prototype, the standard
few-shot construction. That prototype is installed with set_classes(["logo"], vpe), after
which detection runs on the normal path — so everything downstream is unchanged.

Why exemplars come from the pool, not the frames
------------------------------------------------
Cropping exemplars out of the 100 frozen frames and then measuring on those same frames would
be leakage: the prompt would contain the answers. /ml/pools/logo_pool is an independent
source, which also matches the real deployment (a curated brand pool already exists).

Brands present in the evaluation footage (NBA, Gap, New Era) are included deliberately — that
is the instance-level use case visual prompting is *for*. A diverse random sample is included
alongside them so the false-positive rate is measured on brands that are not present, not just
recall on brands that are.

Usage
-----
    python eval/run_visual_prompt.py --brands 30 --per-brand 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = os.environ.get("ELV_LOGO_POOL", "/pool")
WEIGHTS_DIR = os.environ.get("ELV_WEIGHTS_DIR", "/root/.cache/detection")

# Brands visible in the frozen frames. Present on purpose: this is the targeted use case.
SEEDED = ["NBA", "Gap", "New Era Cap Company", "Nike", "Adidas"]


def pick_exemplars(n_brands: int, per_brand: int, seed: int):
    brands = sorted(d for d in os.listdir(POOL) if os.path.isdir(os.path.join(POOL, d)))
    rng = random.Random(seed)
    chosen = [b for b in SEEDED if b in brands]
    pool = [b for b in brands if b not in chosen]
    chosen += rng.sample(pool, max(0, n_brands - len(chosen)))

    exemplars = []
    for brand in chosen:
        files = sorted(
            f for f in os.listdir(os.path.join(POOL, brand))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        for name in files[:per_brand]:
            exemplars.append((brand, os.path.join(POOL, brand, name)))
    return chosen, exemplars


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="yoloe-11l-seg.pt")
    parser.add_argument("--only", nargs="*", default=None,
                        help="use only these brands (instance-level test; a single brand "
                             "with one image is the N=1 case)")
    parser.add_argument("--brands", type=int, default=30)
    parser.add_argument("--per-brand", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.005)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--name", default="yoloe11-visual")
    parser.add_argument("--out", default=os.path.join(HERE, "runs"))
    args = parser.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    with open(os.path.join(HERE, "frames.json")) as handle:
        frames = json.load(handle)["frames"]

    if args.only:
        # Averaging distinct brands dilutes the prototype toward 'some graphic thing'.
        # Restricting to one brand keeps it instance-specific, which is what visual
        # prompting is actually for.
        global SEEDED
        SEEDED = list(args.only)
        args.brands = len(args.only)
    brands, exemplars = pick_exemplars(args.brands, args.per_brand, args.seed)
    print(f"{len(brands)} brands, {len(exemplars)} exemplars")
    print(f"  seeded (present in footage): {', '.join(b for b in SEEDED if b in brands)}")

    os.chdir(WEIGHTS_DIR)
    model = YOLOE(args.weights)

    vectors = []
    for i, (brand, path) in enumerate(exemplars, 1):
        try:
            width, height = Image.open(path).size
            # Pool images are tight crops, so the exemplar box is the entire image.
            prompts = {"bboxes": np.array([[0.0, 0.0, float(width), float(height)]]),
                       "cls": np.array([0])}
            model.predict(path, visual_prompts=prompts, predictor=YOLOEVPSegPredictor,
                          verbose=False)
            vectors.append(model.predictor.get_vpe(path))
        except Exception as exc:
            print(f"  skip {brand}/{os.path.basename(path)}: {type(exc).__name__}: {exc}")
        print(f"  embedded {i}/{len(exemplars)}", end="\r", flush=True)

    if not vectors:
        print("no exemplars embedded", file=sys.stderr)
        return 1

    # Prototype: mean of exemplar embeddings, re-normalised. YOLOE compares prompt embeddings
    # to region embeddings by cosine, so the prototype must be unit length like a text PE.
    prototype = torch.cat(vectors, dim=1).mean(dim=1, keepdim=True)
    prototype = torch.nn.functional.normalize(prototype, p=2, dim=-1)
    print(f"\nprototype {tuple(prototype.shape)} from {len(vectors)} exemplars")

    model.set_classes(["logo"], prototype)
    # Drop the visual-prompt predictor so prediction runs on the normal path with the
    # prototype installed as a class embedding.
    model.predictor = None

    path = os.path.join(args.out, args.name, "out.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    written = 0
    with open(path, "w") as handle:
        for start in range(0, len(frames), 8):
            chunk = frames[start : start + 8]
            results = model.predict([os.path.join(HERE, f["frame"]) for f in chunk],
                                    imgsz=args.imgsz, conf=args.conf, iou=0.7,
                                    max_det=args.max_det, verbose=False)
            for frame, result in zip(chunk, results):
                h, w = result.orig_shape
                if result.boxes is None:
                    continue
                for xyxy, score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
                    x1, y1, x2, y2 = xyxy
                    handle.write(json.dumps({"type": "tag", "data": {
                        "start_time": 0, "end_time": 0,
                        "source_media": os.path.join("/elv/test", frame["id"] + ".png"),
                        "tag": "logo", "track": "",
                        "frame_info": {"frame_idx": frame["frame_idx"], "box": {
                            "x1": round(max(0.0, x1 / w), 4), "y1": round(max(0.0, y1 / h), 4),
                            "x2": round(min(1.0, x2 / w), 4), "y2": round(min(1.0, y2 / h), 4)}},
                        "additional_info": {"kind": "crop", "prompt": "visual:logo",
                                            "score": round(float(score), 4),
                                            "detector": args.name,
                                            "exemplars": len(vectors),
                                            "brands": len(brands)},
                    }}) + "\n")
                    written += 1
            print(f"  {min(start + 8, len(frames))}/{len(frames)} frames, {written} dets",
                  end="\r", flush=True)

    print(f"\n{written} detections -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
