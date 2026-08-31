#!/usr/bin/env python3
"""Choose a shippable score threshold, and measure how much the choice costs when it is wrong.

The problem with every other number in this repo
------------------------------------------------
score_boxes.py sweeps each detector over its own thresholds and reports it at its best. That is
the right way to RANK models -- their scores are on different scales, so any fixed threshold
would rank them by score calibration instead of by detection quality.

It is the wrong way to SHIP one. The reported threshold was chosen by looking at the same 25
frames it is then evaluated on, so the reported F1 is an in-sample maximum and is optimistic by
an unknown amount. Production gets one threshold in a config file and then meets frames nobody
tuned on.

This measures that gap directly: pick the threshold on some clips, evaluate it on the others.

Why the split is by CLIP and not by frame
-----------------------------------------
The 25 frames come from 11 clips, three frames per clip in most cases, sampled seconds apart.
Two frames from the same NBA possession share lighting, camera, jerseys and hoardings -- and the
same physical logos, often at the same scale. A frame-level split would put near-duplicates on
both sides, so the held-out set would not be held out in any meaningful sense and the estimate
would inherit most of the optimism it exists to measure.

Leave-one-clip-out is therefore the unit. Every clip takes a turn as the test set with the
threshold chosen on the other ten, which gives 11 honest evaluations and, just as usefully, 11
independently chosen thresholds -- if those disagree wildly, no single config value is safe.

Two ways to pick, and they answer different questions
-----------------------------------------------------
    best-F1     the balanced choice, and what the ranking tables use.
    at-precision  the lowest threshold whose precision on the training clips is at least some
                  target. This is usually what a tagger actually wants: a wrong tag is visible
                  to a user and a missed one is not, so precision is worth buying with recall.

Usage
-----
    python eval/box_gt/operating_point.py
    python eval/box_gt/operating_point.py --backends gdino owlv2 --target-precision 0.6
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
sys.path.insert(0, HERE)

import paths  # noqa: E402
import score_boxes as SB  # noqa: E402


def clip_of(frame_id: str) -> str:
    """`NBA33min__001` -> `NBA33min`. The frame set names frames <clip>__<index>."""
    return frame_id.rsplit("__", 1)[0]


def load_gt(labels_path):
    with open(labels_path) as handle:
        raw = json.load(handle)["frames"]
    gt = {c: defaultdict(list) for c in SB.CLASSES}
    ignores = defaultdict(list)
    frame_ids = set()
    for frame_id, entry in raw.items():
        if not entry.get("done"):
            continue
        frame_ids.add(frame_id)
        for box in entry["boxes"]:
            if box["cls"] == "ignore":
                ignores[frame_id].append(box)
            elif box["cls"] in SB.CLASSES:
                gt[box["cls"]][frame_id].append(box)
    return gt, ignores, frame_ids


def curve(dets, gt, ignores, cls, frames, iou_thr):
    """Score-ordered (score, tp) records restricted to `frames`, plus the positive count."""
    records = []
    positives = sum(len(v) for f, v in gt.get(cls, {}).items() if f in frames)
    for frame_id, rows in dets.get(cls, {}).items():
        if frame_id not in frames:
            continue
        truth = list(gt.get(cls, {}).get(frame_id, []))
        claimed = [False] * len(truth)
        regions = ignores.get(frame_id, [])
        for det in sorted(rows, key=lambda r: -r["score"]):
            best, best_iou = -1, 0.0
            for i, box in enumerate(truth):
                if claimed[i]:
                    continue
                value = SB.iou(det, box)
                if value > best_iou:
                    best, best_iou = i, value
            if best >= 0 and best_iou >= iou_thr:
                claimed[best] = True
                records.append((det["score"], True))
            else:
                if any(SB.centre_in(det, region) for region in regions):
                    continue
                records.append((det["score"], False))
    return records, positives


def measure(records, positives, threshold):
    tp = sum(1 for score, hit in records if score >= threshold and hit)
    fp = sum(1 for score, hit in records if score >= threshold and not hit)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / positives if positives else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp}


def pick(records, positives, mode, target):
    """Choose a threshold from the training records alone."""
    grid = sorted({score for score, _ in records}, reverse=True)
    if not grid:
        return float("inf")
    if mode == "f1":
        return max(grid, key=lambda t: measure(records, positives, t)["f1"])
    # Lowest threshold still meeting the precision target -- lowest because precision rises as
    # the threshold rises, so the lowest qualifying one keeps the most recall.
    ok = [t for t in grid if measure(records, positives, t)["precision"] >= target]
    return min(ok) if ok else grid[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default=paths.BOX_LABELS)
    parser.add_argument("--runs", default="04_brand_person_mark")
    parser.add_argument("--backends", nargs="*", default=["owlv2", "gdino", "yoloe26-text",
                                                          "yolo11", "yoloe11-text"])
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--nms", type=float, default=0.6)
    parser.add_argument("--mode", choices=["f1", "precision"], default="f1")
    parser.add_argument("--target-precision", type=float, default=0.6)
    parser.add_argument("--json-out", default=os.path.join(HERE, "operating_point.json"))
    args = parser.parse_args()

    gt, ignores, frame_ids = load_gt(args.labels)
    clips = sorted({clip_of(f) for f in frame_ids})
    target = args.target_precision if args.mode == "precision" else None
    print(f"{len(frame_ids)} frames from {len(clips)} clips | leave-one-clip-out | "
          f"IoU {args.iou} | nms {args.nms} | pick by "
          f"{'best F1' if args.mode == 'f1' else f'precision >= {target}'}\n")

    results = {}
    for cls in SB.CLASSES:
        header = (f"{'run':<15}{'in-sample F1':>13}{'held-out F1':>13}{'drop':>7}"
                  f"{'held P':>8}{'held R':>8}{'threshold':>22}")
        print(f"--- {cls} ---\n{header}\n{'-' * len(header)}")
        for backend in args.backends:
            # Mirror paths.runs_glob: an experiment dir may hold runs/ or be a run dir itself
            # (06_resolution/runs_ul_1280), so probe both rather than assuming the former.
            base = paths.experiment(args.runs)
            path = os.path.join(base, "runs", backend, "out.jsonl")
            if not os.path.exists(path):
                path = os.path.join(base, backend, "out.jsonl")
            if not os.path.exists(path):
                continue
            dets, _ = SB.read_detections(path, frame_ids, nms=args.nms)

            everything, all_pos = curve(dets, gt, ignores, cls, frame_ids, args.iou)
            in_sample = measure(everything, all_pos,
                                pick(everything, all_pos, args.mode, target))

            # Leave-one-clip-out: threshold from the other clips, measured on this one. The
            # folds are pooled rather than averaged, so a clip with many boxes counts for more
            # -- averaging per-clip F1 would let a clip holding two marks outvote one holding
            # forty.
            pooled: List = []
            pooled_positives = 0
            chosen = []
            for held in clips:
                test = {f for f in frame_ids if clip_of(f) == held}
                train = frame_ids - test
                tr_records, tr_pos = curve(dets, gt, ignores, cls, train, args.iou)
                threshold = pick(tr_records, tr_pos, args.mode, target)
                chosen.append(threshold)
                te_records, te_pos = curve(dets, gt, ignores, cls, test, args.iou)
                pooled.extend([(s, h) for s, h in te_records if s >= threshold])
                pooled_positives += te_pos
            held_out = measure(pooled, pooled_positives, float("-inf"))

            finite = [t for t in chosen if t != float("inf")]
            spread = (f"{min(finite):.3f}-{max(finite):.3f}" if finite else "n/a")
            drop = in_sample["f1"] - held_out["f1"]
            print(f"{backend:<15}{in_sample['f1']:>13.3f}{held_out['f1']:>13.3f}"
                  f"{drop:>+7.3f}{held_out['precision']:>8.2f}{held_out['recall']:>8.2f}"
                  f"{spread:>22}")
            results.setdefault(cls, {})[backend] = {
                "in_sample": in_sample, "held_out": held_out,
                "thresholds": chosen, "spread": spread}
        print()

    print("in-sample F1 is the optimistic number the ranking tables report: its threshold was\n"
          "chosen on the same frames it is scored on. held-out F1 is what a config file would\n"
          "actually deliver. `threshold` is the range chosen across the 11 folds -- a wide range\n"
          "means no single value is safe and the detector needs recalibrating per content type.")

    with open(args.json_out, "w") as handle:
        json.dump({"iou": args.iou, "nms": args.nms, "mode": args.mode,
                   "target_precision": target, "clips": clips, "results": results},
                  handle, indent=1)
    print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
