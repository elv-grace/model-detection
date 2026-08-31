#!/usr/bin/env python3
"""How much does crop_padding change a vector, and does a mismatch break retrieval?

The README states that vectors built at different `crop_padding` are not comparable. That was a
mechanism argument, not a measurement: padding changes both the pixels in the crop and the scale
the object occupies within it (at 0.06 the object fills 1/1.12 = 89% of the crop's linear extent
rather than 100%), and SigLIP encodes the whole crop, so the vector must move.

"Must move" is not the same as "breaks". The operational question is whether a padding mismatch
between a stored vector and a query vector costs a retrieval, and that has an answer:

    self-similarity   cos(same object @ padding A, same object @ padding B)
    distractor        cos(different objects, same padding) -- the noise floor to beat
    recall@1          for each object embedded at A, is the SAME object at B its nearest
                      neighbour among all objects at B? This is the number that matters.
                      Matched padding is the control; the gap to it is the real cost.

Ground-truth boxes are used rather than detections so object identity is known exactly and a
retrieval failure is a real failure rather than a labelling artefact.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paths  # noqa: E402

PADDINGS = [0.0, 0.06, 0.12, 0.25]


def crop(img, box, padding):
    h, w = img.shape[:2]
    x1, y1 = box["x1"] * w, box["y1"] * h
    x2, y2 = box["x2"] * w, box["y2"] * h
    pw, ph = (x2 - x1) * padding, (y2 - y1) * padding
    cx1, cy1 = max(0, int(round(x1 - pw))), max(0, int(round(y1 - ph)))
    cx2, cy2 = min(w, int(round(x2 + pw))), min(h, int(round(y2 + ph)))
    if cx2 - cx1 < 4 or cy2 - cy1 < 4:
        return None
    return np.ascontiguousarray(img[cy1:cy2, cx1:cx2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/siglip2-base-patch16-naflex")
    parser.add_argument("--cls", default="brand", choices=["brand", "person"])
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    from transformers import AutoModel, AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    frames = {f["id"]: f for f in json.load(open(paths.FRAMES_JSON))["frames"]}
    labels = json.load(open(paths.BOX_LABELS))["frames"]

    objects = []   # (frame_id, box) for each ground-truth object of the wanted class
    for frame_id, entry in labels.items():
        if not entry.get("done"):
            continue
        for box in entry["boxes"]:
            if box["cls"] == args.cls:
                objects.append((frame_id, box))

    images = {fid: np.array(Image.open(os.path.join(paths.FRAMESET, frames[fid]["frame"]))
                            .convert("RGB")) for fid, _ in objects}

    # Keep only objects croppable at EVERY padding, so each row compares like with like.
    vectors = {}
    usable = [i for i, (fid, box) in enumerate(objects)
              if all(crop(images[fid], box, p) is not None for p in PADDINGS)]
    print(f"{args.model}\n{len(usable)} ground-truth `{args.cls}` objects, "
          f"paddings {PADDINGS}\n")

    for padding in PADDINGS:
        crops = [Image.fromarray(crop(images[objects[i][0]], objects[i][1], padding))
                 for i in usable]
        out = []
        for i in range(0, len(crops), args.batch):
            with torch.no_grad():
                inputs = processor(images=crops[i:i + args.batch], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                feats = model.get_image_features(**inputs)
            # transformers 5.x returns BaseModelOutputWithPooling here rather than a tensor.
            feats = getattr(feats, "pooler_output", feats)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append(feats.cpu())
        vectors[padding] = torch.cat(out).numpy()

    base = 0.06   # the shipped default; everything is compared against it
    ref = vectors[base]
    # Distractor floor: similarity between DIFFERENT objects at the shipped padding. If
    # cross-padding self-similarity is not clearly above this, padding is a dominant axis.
    sims = ref @ ref.T
    np.fill_diagonal(sims, np.nan)
    floor = np.nanmean(sims)
    print(f"distractor floor (different objects, both @ {base}): {floor:.3f}\n")

    print(f"{'padding':>9}{'self-sim vs 0.06':>19}{'margin':>9}"
          f"{'recall@1':>11}{'mean rank':>11}")
    print("-" * 59)
    for padding in PADDINGS:
        other = vectors[padding]
        self_sim = float(np.mean(np.sum(ref * other, axis=1)))
        cross = ref @ other.T                     # query @ 0.06 against index @ padding
        order = np.argsort(-cross, axis=1)
        rank = np.array([np.where(order[i] == i)[0][0] for i in range(len(usable))])
        print(f"{padding:>9}{self_sim:>19.3f}{self_sim - floor:>9.3f}"
              f"{float(np.mean(rank == 0)):>11.3f}{float(np.mean(rank)) + 1:>11.1f}")

    print("\nrecall@1 at padding 0.06 is the CONTROL (query and index built identically).\n"
          "Every other row is the cost of a mismatch between them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
