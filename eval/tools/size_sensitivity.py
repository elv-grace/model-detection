#!/usr/bin/env python3
"""What does `min_crop_pixels` buy, and what does it cost?

The parameter drops a detection whose un-padded box has a shorter side below N source pixels.
It ships at 32, and the box ground truth says 32 discards TWO THIRDS of the brand marks the
detector is asked to find (66/192 survive). That is a large price, and it was never measured
against a benefit -- the 32 came from an upscale-factor argument, not from data:

    SigLIP 2's NaFlex processor binary-searches a scale to FILL max_num_patches, with no cap at
    1.0 (scale_max = 100.0 in transformers' get_image_size_for_max_num_patches). So a 16px crop
    is upscaled ~16x at budget 256. The claim was that the vision tower then encodes
    interpolation blur as texture, and heavily-upscaled crops cluster with each other rather
    than by content -- which would make small crops worse than useless, since they would pollute
    the index with vectors keyed on blur.

That claim is testable, and this script tests both halves of the trade.

    (1) BENEFIT, powered and labelled. Pool-to-pool retrieval with the QUERY downscaled to a
        target short side, the gallery left at native. This is the production query shape --
        a small crop against clean reference art -- with real brand identity, so recall@1 as a
        function of query size is measured rather than argued. Downscaling clean art is
        OPTIMISTIC relative to a real broadcast crop (no motion blur, no compression at that
        scale), so where this curve breaks is a LOWER bound on where a real crop breaks.

    (2) MECHANISM, on real crops. If heavy upscaling makes size the dominant axis, then a
        ground-truth crop's nearest neighbour among other ground-truth crops should be closer
        to it in SIZE than chance. Measured as median |log2 size ratio| between a crop and its
        nearest neighbour, against the same statistic for random pairs.

    (3) COST. The fraction of ground-truth marks each threshold discards, per class.

Uses the shipped embedder and the shipped crop_padding by default, since the question is about
the configuration that actually runs.
"""
from __future__ import annotations

import argparse
import collections
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

POOL = "/ml/pools/logo_pool/2960brands"
MODEL = "google/siglip2-base-patch16-naflex"
# Bracketing the ground truth: the brand median short side is 22px, so the decision lives
# between 12 and 32 and the ends are there to show the curve's shape.
SIZES = [8, 12, 16, 20, 24, 32, 48, 64, 96]
THRESHOLDS = [0, 8, 12, 16, 20, 24, 32, 48, 64, 96]


def load_embedder(model_id, device):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    return processor, model


def embed_images(images, processor, model, device, batch):
    """L2-normalized pooled vectors for a list of PIL images."""
    out = []
    for i in range(0, len(images), batch):
        with torch.no_grad():
            inputs = processor(images=images[i:i + batch], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            feats = model.get_image_features(**inputs)
        # transformers 5.x returns BaseModelOutputWithPooling rather than a bare tensor.
        feats = getattr(feats, "pooler_output", feats)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        out.append(feats.float().cpu())
    return torch.cat(out).numpy()


def downscale(image, short_side):
    """Resize so the shorter side is `short_side`, preserving aspect. Never upscales:
    an image already smaller than the target is returned unchanged, so a bin never contains
    a crop that was invented rather than reduced."""
    width, height = image.size
    if min(width, height) <= short_side:
        return image, False
    scale = short_side / min(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.BICUBIC), True


# ---- (1) benefit: retrieval vs query size -------------------------------------------------


def load_pool(root, n_brands, per_brand, seed, min_short):
    """One query + (per_brand - 1) gallery images per brand.

    Queries must be at least `min_short` px on the short side natively, so that every size in
    the sweep is a genuine downscale of the SAME image rather than a mix of downscaled and
    left-alone ones -- otherwise the largest bins would silently contain smaller images and
    flatten the curve.
    """
    rng = random.Random(seed)
    brands = sorted(b for b in os.listdir(root) if os.path.isdir(os.path.join(root, b)))
    rng.shuffle(brands)
    queries, gallery = [], []
    for brand in brands:
        directory = os.path.join(root, brand)
        files = sorted(os.listdir(directory))
        if len(files) < 2:
            continue
        rng.shuffle(files)
        chosen = files[:per_brand]
        try:
            with Image.open(os.path.join(directory, chosen[0])) as probe:
                if min(probe.size) < min_short:
                    continue
        except Exception:
            continue
        queries.append((os.path.join(directory, chosen[0]), brand))
        for f in chosen[1:]:
            gallery.append((os.path.join(directory, f), brand))
        if len(queries) >= n_brands:
            break
    return queries, gallery


def benefit(args, processor, model, device):
    queries, gallery = load_pool(args.pool, args.brands, args.per_brand, args.seed,
                                 max(SIZES))
    q_brand = np.array([b for _, b in queries])
    g_brand = np.array([b for _, b in gallery])
    print(f"(1) BENEFIT -- retrieval vs query crop size\n"
          f"    {len(queries)} queries / {len(gallery)} gallery images, "
          f"{len(set(q_brand))} brands\n"
          f"    gallery at native resolution; query downscaled to each short side\n")

    gv = embed_images([Image.open(p).convert("RGB") for p, _ in gallery],
                      processor, model, device, args.batch)

    rows = {}
    for size in SIZES + ["native"]:
        images, upscales = [], []
        for path, _ in queries:
            image = Image.open(path).convert("RGB")
            if size != "native":
                image, _ = downscale(image, size)
            images.append(image)
            # The linear scale NaFlex applies to reach the patch budget, so the number the
            # mechanism argument is actually about is reported beside the outcome.
            upscales.append((args.patches * 16 * 16 / (image.size[0] * image.size[1])) ** 0.5)
        qv = embed_images(images, processor, model, device, args.batch)

        sims = qv @ gv.T
        order = np.argsort(-sims, axis=1)
        hits = g_brand[order] == q_brand[:, None]
        r1 = float(hits[:, 0].mean())
        r5 = float(hits[:, :5].any(axis=1).mean())
        first = np.argmax(hits, axis=1)
        found = hits.any(axis=1)
        mrr = float(np.mean(np.where(found, 1.0 / (first + 1), 0.0)))

        # Is a small crop's WRONG answer self-identifying? A miss that comes back at a low
        # cosine is harmless -- a downstream similarity gate drops it. A miss that comes back
        # as confidently as a hit is poison, because no threshold separates them, and that is
        # the difference between "gate small crops out" and "let them through and gate on
        # similarity instead".
        top1 = sims[np.arange(len(qv)), order[:, 0]]
        hit_mask = hits[:, 0]
        sim_hit = float(top1[hit_mask].mean()) if hit_mask.any() else float("nan")
        sim_miss = float(top1[~hit_mask].mean()) if (~hit_mask).any() else float("nan")

        rows[str(size)] = {"recall_at_1": r1, "recall_at_5": r5, "mrr": mrr,
                           "median_upscale": float(np.median(upscales)),
                           "top1_sim_hit": sim_hit, "top1_sim_miss": sim_miss}
        print(f"    measured {str(size):>7} ...", end="\r", flush=True)

    # Printed in a second pass: `native` is the control every row is compared against and it
    # is measured last, so the delta column cannot be filled in as the sweep runs.
    base = rows["native"]["recall_at_1"]
    for row in rows.values():
        row["delta_vs_native"] = row["recall_at_1"] - base

    header = (f"{'query short side':>17}{'r@1':>8}{'r@5':>8}{'MRR':>8}"
              f"{'upscale':>10}{'vs native':>11}{'sim hit':>10}{'sim miss':>10}{'gap':>8}")
    print(header)
    print("    " + "-" * (len(header) - 4))
    for size in SIZES + ["native"]:
        row = rows[str(size)]
        label = f"{size}px" if size != "native" else "native"
        delta = "" if size == "native" else f"{row['delta_vs_native']:>+11.3f}"
        print(f"    {label:>13}{row['recall_at_1']:>8.3f}{row['recall_at_5']:>8.3f}"
              f"{row['mrr']:>8.3f}{row['median_upscale']:>9.1f}x{delta or '':>11}"
              f"{row['top1_sim_hit']:>10.3f}{row['top1_sim_miss']:>10.3f}"
              f"{row['top1_sim_hit'] - row['top1_sim_miss']:>8.3f}")

    n = len(queries)
    print(f"\n    Binomial SE at n={n} is {(0.25 / n) ** 0.5:.3f}; a drop under "
          f"~{2 * (0.25 / n) ** 0.5:.3f} is a tie with native.")
    print("    Clean art downscaled cleanly, so this is an OPTIMISTIC bound: a real broadcast\n"
          "    crop at the same pixel size carries motion blur and compression this does not.\n")
    return rows


# ---- (2) mechanism: is size the dominant axis on real crops? ------------------------------


def gt_crops(cls, padding):
    """(crop, short side) for every ground-truth object of `cls`, cropped exactly as the
    tagger crops -- padded box, un-padded box gating the size."""
    frames = {f["id"]: f for f in json.load(open(paths.FRAMES_JSON))["frames"]}
    labels = json.load(open(paths.BOX_LABELS))["frames"]
    out = []
    for frame_id, entry in sorted(labels.items()):
        if not entry.get("done"):
            continue
        boxes = [b for b in entry["boxes"] if b.get("cls") == cls]
        if not boxes:
            continue
        image = Image.open(os.path.join(paths.FRAMESET, frames[frame_id]["frame"])).convert("RGB")
        width, height = image.size
        for box in boxes:
            x1, y1 = box["x1"] * width, box["y1"] * height
            x2, y2 = box["x2"] * width, box["y2"] * height
            short = min(x2 - x1, y2 - y1)
            pw, ph = (x2 - x1) * padding, (y2 - y1) * padding
            crop = image.crop((int(max(0, x1 - pw)), int(max(0, y1 - ph)),
                               int(min(width, x2 + pw)), int(min(height, y2 + ph))))
            if min(crop.size) < 2:
                continue
            out.append((crop, short))
    return out


def mechanism(args, processor, model, device):
    crops = gt_crops(args.cls, args.padding)
    sizes = np.array([s for _, s in crops])
    vectors = embed_images([c for c, _ in crops], processor, model, device, args.batch)
    print(f"(2) MECHANISM -- does upscaling make SIZE the dominant axis?\n"
          f"    {len(crops)} ground-truth `{args.cls}` crops, padding {args.padding}\n")

    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -np.inf)
    nn = np.argmax(sims, axis=1)
    log_ratio = np.abs(np.log2(sizes / sizes[nn]))

    rng = np.random.default_rng(args.seed)
    random_partner = rng.permutation(len(sizes))
    random_ratio = np.abs(np.log2(sizes / sizes[random_partner]))

    print(f"    median |log2 size ratio| to nearest neighbour : {np.median(log_ratio):.2f}")
    print(f"    median |log2 size ratio| to a random crop     : {np.median(random_ratio):.2f}")
    print("    (equal => the embedding is not keyed on size; much smaller => it is)\n")

    # Same question by bin: mean similarity WITHIN a size bin against the overall floor.
    np.fill_diagonal(sims, np.nan)
    floor = float(np.nanmean(sims))
    edges = [0, 16, 24, 32, 48, 10 ** 6]
    print(f"    {'size bin':>14}{'n':>6}{'mean sim within':>18}{'vs floor':>10}")
    print("    " + "-" * 44)
    bins = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.where((sizes >= lo) & (sizes < hi))[0]
        if len(idx) < 3:
            continue
        within = float(np.nanmean(sims[np.ix_(idx, idx)]))
        label = f"{lo}-{hi}px" if hi < 10 ** 6 else f"{lo}px+"
        bins[label] = {"n": int(len(idx)), "mean_sim_within": within,
                       "vs_floor": within - floor}
        print(f"    {label:>14}{len(idx):>6}{within:>18.3f}{within - floor:>+10.3f}")
    print(f"\n    overall floor (all pairs): {floor:.3f}\n")

    return {"n": len(crops), "cls": args.cls,
            "median_log2_ratio_nn": float(np.median(log_ratio)),
            "median_log2_ratio_random": float(np.median(random_ratio)),
            "floor": floor, "bins": bins}


# ---- (3) cost: what each threshold discards -----------------------------------------------


def cost():
    frames = {f["id"]: f for f in json.load(open(paths.FRAMES_JSON))["frames"]}
    labels = json.load(open(paths.BOX_LABELS))["frames"]
    sizes = collections.defaultdict(list)
    for frame_id, entry in labels.items():
        if not entry.get("done"):
            continue
        with Image.open(os.path.join(paths.FRAMESET, frames[frame_id]["frame"])) as image:
            width, height = image.size
        for box in entry["boxes"]:
            if box.get("cls") not in ("brand", "person"):
                continue   # `ignore` regions are not marks the detector is graded on
            sizes[box["cls"]].append(min((box["x2"] - box["x1"]) * width,
                                         (box["y2"] - box["y1"]) * height))

    print("(3) COST -- ground-truth objects surviving each threshold\n")
    print(f"    {'min_crop_pixels':>16}" + "".join(f"{c:>12}" for c in sorted(sizes)))
    print("    " + "-" * (16 + 12 * len(sizes)))
    out = {}
    for threshold in THRESHOLDS:
        row = {c: float((np.array(v) >= threshold).mean()) for c, v in sizes.items()}
        out[str(threshold)] = row
        print(f"    {threshold:>16}" + "".join(f"{row[c]:>12.3f}" for c in sorted(sizes)))
    print()
    for cls, values in sorted(sizes.items()):
        values = np.array(values)
        print(f"    {cls}: n={len(values)}, median short side "
              f"{np.median(values):.0f}px, p10 {np.percentile(values, 10):.0f}px, "
              f"p90 {np.percentile(values, 90):.0f}px")
    print()
    return out


# ---- (4) decision: the two curves composed over the real size distribution ----------------


def decision(benefit_rows, args):
    """Compose (1) and (3): what a threshold actually delivers on THIS footage.

    A threshold is not free in either direction, and the two errors are not symmetric:

        identification RECALL     of all ground-truth marks, the fraction that survive the gate
                                  AND retrieve the right brand. Gating can only lower this.
        identification PRECISION  of the marks that survive the gate, the fraction that retrieve
                                  the right brand. Gating raises this.

    Precision is the one that matters more than its name suggests, because a small crop does not
    fail quietly -- see the sim hit/miss columns in (1). Its wrong answer comes back at a cosine
    a downstream similarity gate cannot distinguish from a right one, so an ungated index does
    not merely miss marks, it asserts wrong brands with confidence.
    """
    frames = {f["id"]: f for f in json.load(open(paths.FRAMES_JSON))["frames"]}
    labels = json.load(open(paths.BOX_LABELS))["frames"]
    sizes = []
    for frame_id, entry in labels.items():
        if not entry.get("done"):
            continue
        with Image.open(os.path.join(paths.FRAMESET, frames[frame_id]["frame"])) as image:
            width, height = image.size
        for box in entry["boxes"]:
            if box.get("cls") == "brand":
                sizes.append(min((box["x2"] - box["x1"]) * width,
                                 (box["y2"] - box["y1"]) * height))
    sizes = np.array(sizes)

    grid = np.array(SIZES, dtype=float)
    r1 = np.array([benefit_rows[str(int(s))]["recall_at_1"] for s in SIZES])
    gap = np.array([benefit_rows[str(int(s))]["top1_sim_hit"]
                    - benefit_rows[str(int(s))]["top1_sim_miss"] for s in SIZES])
    # Interpolated in log size, which is the axis the measured curve is smooth in. Below the
    # smallest measured size the curve is clamped rather than extrapolated -- 8px is already
    # near chance and extrapolating past it would invent numbers.
    def at(values, s):
        return np.interp(np.log2(np.clip(s, grid[0], grid[-1])), np.log2(grid), values)

    per_mark = at(r1, sizes)
    print("(4) DECISION -- the two curves composed over the ground-truth brand marks\n"
          f"    {len(sizes)} marks, r@1 interpolated from (1) at each mark's own size\n")
    header = (f"{'min_crop_pixels':>16}{'kept':>8}{'ident recall':>14}"
              f"{'ident precision':>17}{'hit/miss gap':>14}")
    print(header)
    print("    " + "-" * (len(header) - 4))
    out = {}
    for threshold in THRESHOLDS:
        keep = sizes >= threshold
        kept = float(keep.mean())
        recall = float(per_mark[keep].sum() / len(sizes)) if keep.any() else 0.0
        precision = float(per_mark[keep].mean()) if keep.any() else float("nan")
        mean_gap = float(at(gap, sizes[keep]).mean()) if keep.any() else float("nan")
        out[str(threshold)] = {"kept": kept, "ident_recall": recall,
                               "ident_precision": precision, "mean_hit_miss_gap": mean_gap}
        print(f"    {threshold:>16}{kept:>8.3f}{recall:>14.3f}"
              f"{precision:>17.3f}{mean_gap:>14.3f}")
    print("\n    `kept` is the fraction of marks that reach the embedder at all.\n"
          "    `ident precision` is optimistic twice over: the curve it interpolates was\n"
          "    measured on cleanly downscaled art, and it credits a retrieval that a real\n"
          "    broadcast crop of that size would more often miss.\n")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--pool", default=POOL)
    parser.add_argument("--brands", type=int, default=800)
    parser.add_argument("--per-brand", type=int, default=4)
    parser.add_argument("--cls", default="brand", choices=["brand", "person"])
    parser.add_argument("--padding", type=float, default=0.06,
                        help="crop_padding, matching the shipped default")
    parser.add_argument("--patches", type=int, default=256,
                        help="max_num_patches, for reporting the upscale NaFlex applies")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["benefit", "mechanism", "cost", "decision"])
    parser.add_argument("--json-out",
                        default=os.path.join(paths.EXPERIMENTS, "09_min_crop", "scores.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{args.model} on {device}\n")
    processor, model = load_embedder(args.model, device)

    results = {"model": args.model, "pool": args.pool, "padding": args.padding,
               "max_num_patches": args.patches, "seed": args.seed}
    if "benefit" not in args.skip:
        results["benefit"] = benefit(args, processor, model, device)
    if "mechanism" not in args.skip:
        results["mechanism"] = mechanism(args, processor, model, device)
    if "cost" not in args.skip:
        results["cost"] = cost()
    if "decision" not in args.skip and "benefit" in results:
        results["decision"] = decision(results["benefit"], args)

    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w") as handle:
        json.dump(results, handle, indent=1)
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
