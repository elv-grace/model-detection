#!/usr/bin/env python3
"""Phase B: compare SigLIP 2 checkpoints on brand-logo retrieval.

The task, and why this one
--------------------------
The tagger's vectors exist to answer "which brand is this crop" by cosine similarity against a
pool. So the benchmark is that exact query: embed a held-out image of a brand, and ask whether
the nearest neighbours in a large gallery are the same brand.

An earlier version of this script tried to build pairs from the box ground truth by matching
boxes across frames of one clip at IoU >= 0.3. That proxy is invalid and was discarded: the
frames are sampled seconds to a minute apart, so in crowd and sports footage a box at the same
position is a DIFFERENT person. It scored person recall@1 at 0.047 and produced only 24 brand
pairs -- too few to separate two models (binomial SE ~0.10) -- and both candidates returned
identical numbers, which is the signature of a metric measuring its own construction rather
than the models.

The logo pool has real identity labels and tens of thousands of images, so it replaces the
proxy entirely.

    /ml/pools/logo_pool/2960brands/<brand>/<n>.jpg

The candidates
--------------
    siglip2-base-patch16-naflex    768-d, aspect-preserving, variable patch count
    siglip2-large-patch16-384     1024-d, fixed 384x384 square

They differ in TWO ways at once -- a bigger encoder and a different resolution policy -- so a
win for -384 does not by itself refute the aspect-ratio argument for NaFlex. so400m is excluded:
it emits 1152-d and the vectorstore caps at 1024.

Protocol
--------
`--brands` brands are sampled, `--per-brand` images each. One image per brand is the QUERY, the
rest go in the GALLERY. Every query is ranked against the whole gallery, so the difficulty scales
with the gallery rather than with the brand count alone.

    recall@1 / @5     is a same-brand image the top hit
    mean reciprocal rank   rewards getting it near the top when not first
    wide-crop subset  recall@1 restricted to queries with aspect ratio >= --wide. This is the
                      cut where NaFlex and fixed-384 should diverge if the squash argument is
                      real: a 4:1 wordmark is squashed to a square by -384 and preserved by
                      NaFlex, so it is the discriminating subset rather than a curiosity.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paths  # noqa: E402

POOL = "/ml/pools/logo_pool/2960brands"
CANDIDATES = [
    "google/siglip2-base-patch16-naflex",
    "google/siglip2-large-patch16-384",
]


def load_pool(root, n_brands, per_brand, seed):
    rng = random.Random(seed)
    brands = sorted(b for b in os.listdir(root) if os.path.isdir(os.path.join(root, b)))
    rng.shuffle(brands)
    queries, gallery = [], []          # (path, brand)
    for brand in brands:
        files = sorted(os.listdir(os.path.join(root, brand)))
        if len(files) < 2:
            continue
        rng.shuffle(files)
        chosen = files[:per_brand]
        queries.append((os.path.join(root, brand, chosen[0]), brand))
        for f in chosen[1:]:
            gallery.append((os.path.join(root, brand, f), brand))
        if len(queries) >= n_brands:
            break
    return queries, gallery


def embed(model_id, paths_, batch, device):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    vectors, aspects = [], []
    for i in range(0, len(paths_), batch):
        chunk = []
        for p in paths_[i:i + batch]:
            image = Image.open(p).convert("RGB")
            aspects.append(max(image.size) / max(1, min(image.size)))
            chunk.append(image)
        with torch.no_grad():
            inputs = processor(images=chunk, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            feats = model.get_image_features(**inputs)
        feats = getattr(feats, "pooler_output", feats)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vectors.append(feats.float().cpu())
        if i and i % (batch * 40) == 0:
            print(f"    {i}/{len(paths_)}".ljust(40), end="\r", flush=True)
    print(" " * 40, end="\r", flush=True)
    dim = vectors[0].shape[1]
    del model
    torch.cuda.empty_cache()
    return torch.cat(vectors).numpy(), np.array(aspects), dim


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=CANDIDATES)
    parser.add_argument("--pool", default=POOL)
    parser.add_argument("--brands", type=int, default=400)
    parser.add_argument("--per-brand", type=int, default=4)
    parser.add_argument("--wide", type=float, default=2.5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out",
                        default=os.path.join(paths.EXPERIMENTS, "08_embedders", "scores.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    queries, gallery = load_pool(args.pool, args.brands, args.per_brand, args.seed)
    q_paths = [p for p, _ in queries]
    g_paths = [p for p, _ in gallery]
    q_brand = np.array([b for _, b in queries])
    g_brand = np.array([b for _, b in gallery])
    print(f"{len(queries)} query images, {len(gallery)} gallery images, "
          f"{len(set(q_brand))} brands\n")

    header = (f"{'model':<34}{'dim':>5}{'r@1':>8}{'r@5':>8}{'MRR':>8}"
              f"{'wide r@1':>10}{'n wide':>8}{'ms/img':>9}")
    print(header)
    print("-" * len(header))
    results = {}
    for model_id in args.models:
        start = time.perf_counter()
        qv, q_aspect, dim = embed(model_id, q_paths, args.batch, device)
        # Timed on the query pass only, and it includes PIL decode plus the processor, which is
        # what the tagger actually pays per crop -- not a bare forward pass.
        ms_per_image = (time.perf_counter() - start) * 1000 / max(1, len(q_paths))
        gv, _, _ = embed(model_id, g_paths, args.batch, device)

        sims = qv @ gv.T
        order = np.argsort(-sims, axis=1)
        hits = g_brand[order] == q_brand[:, None]

        r1 = float(hits[:, 0].mean())
        r5 = float(hits[:, :5].any(axis=1).mean())
        first = np.argmax(hits, axis=1)
        found = hits.any(axis=1)
        mrr = float(np.mean(np.where(found, 1.0 / (first + 1), 0.0)))

        wide = q_aspect >= args.wide
        wide_r1 = float(hits[wide, 0].mean()) if wide.any() else float("nan")

        results[model_id] = {"dim": dim, "recall_at_1": r1, "recall_at_5": r5, "mrr": mrr,
                             "wide_recall_at_1": wide_r1, "n_wide": int(wide.sum()),
                             "ms_per_image": ms_per_image,
                             "queries": len(queries), "gallery": len(gallery)}
        print(f"{model_id.split('/')[-1]:<34}{dim:>5}{r1:>8.3f}{r5:>8.3f}{mrr:>8.3f}"
              f"{wide_r1:>10.3f}{int(wide.sum()):>8}{ms_per_image:>9.1f}")

    n = len(queries)
    print(f"\nBinomial standard error at n={n} is about "
          f"{(0.25 / n) ** 0.5:.3f}; treat differences below ~{2 * (0.25 / n) ** 0.5:.3f} "
          f"as a tie.")
    print(f"`wide` = query aspect ratio >= {args.wide}, the subset where a fixed square resize "
          f"squashes hardest\nand where NaFlex should show an advantage if the argument for it "
          f"holds.")

    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w") as handle:
        json.dump({"pool": args.pool, "brands": args.brands, "per_brand": args.per_brand,
                   "wide_threshold": args.wide, "seed": args.seed, "results": results},
                  handle, indent=1)
    print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
