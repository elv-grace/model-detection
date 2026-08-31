#!/usr/bin/env python3
"""Does the tagger's embedder produce the vectors the evaluation measured?

Every number in eval/ was produced by a benchmark script calling
`AutoModel.get_image_features` in fp32. The tagger does something different on both counts: it
loads `Siglip2VisionModel` -- the vision tower alone, discarding the text half of the checkpoint
-- and it runs in bfloat16 on GPU. Neither choice is wrong, but both are opportunities for the
shipped vectors to quietly stop being the vectors that were graded, so both are checked here
rather than assumed.

It also checks the crop geometry, because `crop_padding` is the parameter that changes a vector
without changing anything visible about it: the crop must be the padded box, and the tag must
report the UN-padded one, or the overlay and the index disagree.

Run it inside the container so the versions under test are the shipped ones:

    podman run --rm --entrypoint /opt/conda/envs/mlpod/bin/python \\
        --volume=$PWD/eval:/elv/eval:ro --volume=detection_cache:/root/.cache \\
        --network host -e HF_HOME=/root/.cache -e MD_ROOT=/elv \\
        --device nvidia.com/gpu=0 localhost/general_detection \\
        -u /elv/eval/tools/verify_embedder_wiring.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = os.environ.get("MD_ROOT",
                      os.path.dirname(os.path.dirname(os.path.dirname(
                          os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from general_detection.config import RuntimeConfig       # noqa: E402
from general_detection.detector import BaseDetector      # noqa: E402
from general_detection.embedder import Siglip2CropEmbedder  # noqa: E402


def production_crops(cfg, limit):
    """Ground-truth boxes cropped through the production path -- BaseDetector._crop, gated by
    the shipped min_crop_pixels, padded by the shipped crop_padding."""
    frameset = os.path.join(ROOT, "eval", "frameset")
    frames = {f["id"]: f
              for f in json.load(open(os.path.join(frameset, "frames.json")))["frames"]}
    labels = json.load(open(os.path.join(ROOT, "eval", "box_gt",
                                         "box_labels.json")))["frames"]
    out = []
    for frame_id, entry in sorted(labels.items()):
        if not entry.get("done"):
            continue
        image = np.array(Image.open(os.path.join(frameset, frames[frame_id]["frame"]))
                         .convert("RGB"))
        height, width = image.shape[:2]
        for box in entry["boxes"]:
            if box.get("cls") != "brand":
                continue
            x1, y1 = box["x1"] * width, box["y1"] * height
            x2, y2 = box["x2"] * width, box["y2"] * height
            if min(x2 - x1, y2 - y1) < cfg.min_crop_pixels:
                continue
            crop = BaseDetector._crop(image, x1, y1, x2, y2, cfg.crop_padding)
            if crop is not None:
                out.append((crop, (x1, y1, x2, y2), (height, width)))
        if len(out) >= limit:
            break
    return out


def nn_agreement(a, b):
    """Fraction of crops whose nearest neighbour is the same under both embeddings."""
    sa, sb = a @ a.T, b @ b.T
    np.fill_diagonal(sa, -np.inf)
    np.fill_diagonal(sb, -np.inf)
    return float((sa.argmax(1) == sb.argmax(1)).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=160)
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="max cosine deviation allowed between the two fp32 paths")
    args = parser.parse_args()

    cfg = RuntimeConfig()
    settings = yaml.safe_load(open(os.path.join(ROOT, "config.yml")))["model"]["embedder"]
    model_id, revision = settings["model_id"], settings.get("revision")
    print(f"config.yml embedder : {model_id}  revision={revision}")
    print(f"crop_padding {cfg.crop_padding}   min_crop_pixels {cfg.min_crop_pixels}   "
          f"max_num_patches {cfg.max_num_patches}\n")

    failures = []

    # ---- crop geometry -------------------------------------------------------------------
    picked = production_crops(cfg, args.limit)
    print(f"{len(picked)} production crops\n")
    print(f"{'un-padded box':>18}{'crop':>14}{'expected':>14}{'ratio':>8}")
    print("-" * 54)
    bad_geometry = 0
    for crop, (x1, y1, x2, y2), (height, width) in picked:
        box_w, box_h = x2 - x1, y2 - y1
        expected_w = (min(width, round(x2 + box_w * cfg.crop_padding))
                      - max(0, round(x1 - box_w * cfg.crop_padding)))
        expected_h = (min(height, round(y2 + box_h * cfg.crop_padding))
                      - max(0, round(y1 - box_h * cfg.crop_padding)))
        if crop.shape[1] != expected_w or crop.shape[0] != expected_h:
            bad_geometry += 1
    for crop, (x1, y1, x2, y2), _ in picked[:4]:
        box_w, box_h = x2 - x1, y2 - y1
        print(f"{f'{box_w:.1f}x{box_h:.1f}':>18}"
              f"{f'{crop.shape[1]}x{crop.shape[0]}':>14}"
              f"{f'{1 + 2 * cfg.crop_padding:.2f}x':>14}{crop.shape[1] / box_w:>8.3f}")
    if bad_geometry:
        failures.append(f"{bad_geometry} crops do not match the crop_padding arithmetic")
    print(f"\ncrop == padded box on all {len(picked)} crops: "
          f"{'FAIL' if bad_geometry else 'ok'}   "
          f"(ratio is under {1 + 2 * cfg.crop_padding:.2f} only where a box is clipped at a "
          f"frame edge)\n")

    crops = [c for c, _, _ in picked]

    # ---- the vision tower against the benchmark's own call -------------------------------
    embedder = Siglip2CropEmbedder(model_id, revision=revision, dtype=torch.float32)
    vec_fp32, upscales = embedder.embed(crops, cfg)
    device = embedder.device
    del embedder
    torch.cuda.empty_cache()

    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    reference = AutoModel.from_pretrained(model_id, revision=revision).to(device).eval()
    chunks = []
    for i in range(0, len(crops), 32):
        with torch.no_grad():
            inputs = processor(images=[Image.fromarray(c) for c in crops[i:i + 32]],
                               return_tensors="pt", max_num_patches=cfg.max_num_patches)
            feats = reference.get_image_features(
                **{k: v.to(device) for k, v in inputs.items()})
        feats = getattr(feats, "pooler_output", feats)
        chunks.append((feats / feats.norm(dim=-1, keepdim=True)).float().cpu())
    vec_reference = torch.cat(chunks).numpy()
    del reference
    torch.cuda.empty_cache()

    deviation = float(np.abs((vec_fp32 * vec_reference).sum(1) - 1.0).max())
    print(f"Siglip2VisionModel fp32 vs AutoModel.get_image_features fp32")
    print(f"  max |cos - 1| : {deviation:.2e}   "
          f"NN agreement {nn_agreement(vec_fp32, vec_reference):.3f}   "
          f"{'FAIL' if deviation > args.tolerance else 'ok'}")
    if deviation > args.tolerance:
        failures.append("the vision-tower path diverges from the benchmarked path in fp32")

    # ---- the shipped dtype ---------------------------------------------------------------
    embedder = Siglip2CropEmbedder(model_id, revision=revision)
    vec_shipped, _ = embedder.embed(crops, cfg)
    dtype = embedder.dtype
    del embedder
    torch.cuda.empty_cache()

    cos = (vec_shipped * vec_fp32).sum(1)
    similarities = vec_fp32 @ vec_fp32.T
    np.fill_diagonal(similarities, -np.inf)
    top2 = np.sort(similarities, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    shipped_sims = vec_shipped @ vec_shipped.T
    np.fill_diagonal(shipped_sims, -np.inf)
    flipped = np.where(similarities.argmax(1) != shipped_sims.argmax(1))[0]

    norms = np.linalg.norm(vec_shipped, axis=1)
    print(f"\nshipped {dtype} vs fp32")
    print(f"  cos           : min {cos.min():.6f}  mean {cos.mean():.6f}")
    print(f"  NN agreement  : {nn_agreement(vec_shipped, vec_fp32):.3f}")
    if len(flipped):
        # A flip between two crops that fp32 could barely separate is a tie broken differently,
        # not a wrong answer. The margin comparison is what distinguishes the two.
        print(f"  where it flips (n={len(flipped)}): fp32 top-1/top-2 margin median "
              f"{np.median(margin[flipped]):.5f}, against "
              f"{np.median(np.delete(margin, flipped)):.5f} where it does not")
    print(f"  unit norm     : [{norms.min():.5f}, {norms.max():.5f}]   "
          f"{'FAIL' if abs(norms - 1).max() > 1e-3 else 'ok'}")
    if abs(norms - 1).max() > 1e-3:
        failures.append("shipped vectors are not unit length")
    print(f"  upscale       : median {np.median(upscales):.1f}x  "
          f"range {min(upscales):.1f}-{max(upscales):.1f}x")

    print("\n" + ("FAILED:\n  " + "\n  ".join(failures) if failures
                  else "all checks passed"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
