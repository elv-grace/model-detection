#!/usr/bin/env python3
"""The production query: a detected crop against the brand pool.

Why this is the realistic test
------------------------------
compare_embedders.py measures pool-to-pool retrieval -- a clean reference image against other
clean reference images. That is same-domain, and it is not what the tagger does. The tagger
queries a DETECTION against the pool: a small, motion-blurred, off-angle, partly occluded crop
from broadcast video, matched to clean reference art. Cross-domain is where encoders actually
differ, so this is the more honest measurement of the two.

The catch, stated up front
--------------------------
It needs brand identity for each crop, which the box ground truth does not carry -- it labels
`brand`, not *which* brand. Identity was assigned by hand from the ground-truth crops, and only
where BOTH the mark is unambiguous at full resolution AND the brand exists in the pool.

That is a hard limit on this frame set. The footage is NBA and NFL broadcast, and its marks are
mostly leagues, teams and US insurers -- State Farm, ESPN, NBC, the Chiefs arrowhead, the
Buccaneers wordmark, USA Basketball -- none of which are in the 2960-brand pool. Only NBA, NFL,
Nike, KIA and Gap overlap, giving 15 queries.

So this is UNDERPOWERED as a model comparison: at n=15 the binomial standard error is ~0.13, and
only a difference of roughly 0.26 or more would be real. It answers "does cross-domain retrieval
work at all, and does either model fall over" -- not "which model is better". The powered
comparison remains the pool-to-pool one, and this is reported beside it rather than instead.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paths  # noqa: E402
from compare_embedders import embed  # noqa: E402

POOL = "/ml/pools/logo_pool/2960brands"

# index into eval/box_gt/gt_brand_index.json -> pool brand directory. Hand-assigned from the
# full-resolution crops, and deliberately conservative: a mark that could not be read with
# certainty was left out rather than guessed.
LABELS = {
    15: "NBA", 62: "NBA", 71: "NBA", 128: "NBA",
    159: "NFL", 161: "NFL", 168: "NFL", 169: "NFL", 176: "NFL", 179: "NFL",
    164: "Nike", 181: "Nike",
    96: "KIA", 113: "KIA",
    187: "Gap",
}


def crop_for(index_entry, labels, frames, padding):
    frame_id, box_i = index_entry.split("#")
    box = labels[frame_id]["boxes"][int(box_i)]
    image = Image.open(os.path.join(paths.FRAMESET, frames[frame_id]["frame"])).convert("RGB")
    width, height = image.size
    x1, y1 = box["x1"] * width, box["y1"] * height
    x2, y2 = box["x2"] * width, box["y2"] * height
    pw, ph = (x2 - x1) * padding, (y2 - y1) * padding
    return image.crop((int(max(0, x1 - pw)), int(max(0, y1 - ph)),
                       int(min(width, x2 + pw)), int(min(height, y2 + ph))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*",
                        default=["google/siglip2-base-patch16-naflex",
                                 "google/siglip2-large-patch16-384"])
    parser.add_argument("--pool", default=POOL)
    parser.add_argument("--distractors", type=int, default=800,
                        help="brands in the gallery besides the target ones, as a haystack")
    parser.add_argument("--per-brand", type=int, default=8)
    parser.add_argument("--padding", type=float, default=0.06)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out",
                        default=os.path.join(paths.EXPERIMENTS, "08_embedders",
                                             "crop_to_pool.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = {f["id"]: f for f in json.load(open(paths.FRAMES_JSON))["frames"]}
    labels = json.load(open(paths.BOX_LABELS))["frames"]
    index = json.load(open(os.path.join(paths.BOX_GT, "gt_brand_index.json")))

    queries, q_brand = [], []
    for i, brand in sorted(LABELS.items()):
        queries.append(crop_for(index[str(i)], labels, frames, args.padding))
        q_brand.append(brand)
    q_brand = np.array(q_brand)

    targets = sorted(set(q_brand))
    rng = random.Random(args.seed)
    others = [b for b in sorted(os.listdir(args.pool))
              if os.path.isdir(os.path.join(args.pool, b)) and b not in targets]
    rng.shuffle(others)
    gallery_brands = targets + others[:args.distractors]

    g_paths, g_brand = [], []
    for brand in gallery_brands:
        files = sorted(os.listdir(os.path.join(args.pool, brand)))
        rng.shuffle(files)
        for f in files[:args.per_brand]:
            g_paths.append(os.path.join(args.pool, brand, f))
            g_brand.append(brand)
    g_brand = np.array(g_brand)

    print(f"{len(queries)} detected crops across {len(targets)} brands ({', '.join(targets)})")
    print(f"gallery: {len(g_paths)} images from {len(gallery_brands)} brands "
          f"({args.distractors} distractor brands)\n")

    tmp = os.path.join("/tmp", "cropq")
    os.makedirs(tmp, exist_ok=True)
    q_paths = []
    for i, image in enumerate(queries):
        path = os.path.join(tmp, f"{i}.png")
        image.save(path)
        q_paths.append(path)

    header = f"{'model':<34}{'dim':>5}{'r@1':>7}{'r@5':>7}{'MRR':>7}   per-brand r@1"
    print(header)
    print("-" * (len(header) + 6))
    results = {}
    for model_id in args.models:
        qv, _, dim = embed(model_id, q_paths, args.batch, device)
        gv, _, _ = embed(model_id, g_paths, args.batch, device)
        sims = qv @ gv.T
        order = np.argsort(-sims, axis=1)
        hits = g_brand[order] == q_brand[:, None]

        r1 = float(hits[:, 0].mean())
        r5 = float(hits[:, :5].any(axis=1).mean())
        first = np.argmax(hits, axis=1)
        found = hits.any(axis=1)
        mrr = float(np.mean(np.where(found, 1.0 / (first + 1), 0.0)))
        per = {b: float(hits[q_brand == b, 0].mean()) for b in targets}

        results[model_id] = {"dim": dim, "recall_at_1": r1, "recall_at_5": r5, "mrr": mrr,
                             "per_brand": per, "queries": len(queries),
                             "gallery": len(g_paths)}
        detail = "  ".join(f"{b} {per[b]:.2f}" for b in targets)
        print(f"{model_id.split('/')[-1]:<34}{dim:>5}{r1:>7.2f}{r5:>7.2f}{mrr:>7.2f}   {detail}")

    n = len(queries)
    print(f"\nn={n}: binomial SE ~{(0.25 / n) ** 0.5:.2f}, so only a gap above "
          f"~{2 * (0.25 / n) ** 0.5:.2f} is real.\nThis measures whether cross-domain retrieval "
          f"works, not which model is better.")
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w") as handle:
        json.dump({"pool": args.pool, "distractors": args.distractors,
                   "padding": args.padding, "results": results}, handle, indent=1)
    print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
