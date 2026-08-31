#!/usr/bin/env python3
"""Score detector runs against box-level ground truth.

What this measures that frame presence cannot
---------------------------------------------
Frame presence asks "did the detector fire somewhere on a frame containing the class", which
saturated on this schema: brand is present in 58 frames of 100 and person in 97, so the field
landed inside a narrow F1 band. Boxes ask "did it find the object", which separates a
detector that localises from one that merely fires on the right frames.

Three numbers come out, and they answer different questions:

    AP          the ranking quality of the whole score-ordered detection list. The headline.
    P/R @ best  precision and recall at the operating point you would actually ship.
    mean IoU    tightness of the matched boxes. Not a detection metric at all -- it is a
                direct read on the crop quality the embedder inherits, since a loose box
                embeds background alongside the object and a clipped one loses the mark.

Matching
--------
Greedy by descending score, standard COCO practice: each detection takes the highest-IoU
unclaimed ground-truth box of the same class above the IoU threshold. One ground-truth box can
be matched once, so duplicate detections on the same object correctly count as false positives
-- which matters here, because the runs are ungated and Grounding DINO emits well over a
hundred detections per frame.

Ignore regions
--------------
Frames containing dense crowds cannot be exhaustively boxed, and pretending otherwise would
punish detectors for finding real people the labeller did not draw. A detection whose centre
falls inside an `ignore` region is dropped before scoring: it counts neither as a true positive
nor a false positive. Ground-truth `ignore` boxes are never counted as positives.

Cross-prompt de-duplication
---------------------------
On by default at IoU 0.6, because it is part of the pipeline rather than a scoring trick: six
near-synonymous brand terms return the same mark several times, and every repeat after the first
is both a false positive here and a wasted SigLIP forward pass in production. It lifts Grounding
DINO's brand AP from 0.166 to 0.205 and cuts the crops it emits by around a third, with the gain
coming entirely from removed duplicates rather than from found marks.

The threshold is chosen by a constraint, not by fitting AP. Class-agnostic coverage -- how many
ground-truth marks are hit by any detection -- must not fall, because if it does the suppression
is merging marks that are genuinely distinct: adjacent logos on one hoarding, or a wordmark
inside an emblem. Coverage does fall below about 0.5 (gdino 0.609 -> 0.542 at 0.3, owlv2 0.542 ->
0.484), and AP hides that because both boxes were being counted anyway. 0.6 is the smallest
value at which coverage is intact for every backend; it is also where person AP peaks. Fitting
AP instead would have chosen 0.45, which quietly destroys real marks.

Pass --nms 0 to measure the raw detector.

Class-aware and class-agnostic, and why the second one matters
--------------------------------------------------------------
Two numbers are reported per class:

    AP / P / R      class-aware. The detection must carry a tag that maps to the class. This
                    is the right measure for the default and text-target modes, where the
                    detector was told what to look for.

    coverage        CLASS-AGNOSTIC recall: of the ground-truth boxes of this class, how many
                    are hit at the IoU threshold by ANY detection, whatever it was labelled.

Coverage is the number that retires the tag map. A prompt-free backend answers from its own
4585-term vocabulary, so scoring it class-aware needs a mapping from "sports ball" to our
schema -- without one it scores zero by construction rather than by measurement, and that
mapping is a large hand-maintained artefact that nothing in the production pipeline consumes.
Coverage asks the question the pipeline actually cares about instead: did a crop land on the
object. What the detector called it is metadata; the embedding is the retrieval key.

Precision is deliberately not reported class-agnostically. A prompt-free run that boxes a chair
is not WRONG, it is just not boxing a target, so counting it as a false positive would measure
the schema rather than the detector.

Usage
-----
    python eval/box_gt/score_boxes.py
    python eval/box_gt/score_boxes.py --iou 0.5 --runs 05_symbol_ablation/runs_mark7
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

import paths  # noqa: E402
from schema_brand_person import class_of_prompt  # noqa: E402
from dedup import dedup_frame  # noqa: E402

CLASSES = ["brand", "person"]


def iou(a, b) -> float:
    x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    ub = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    denom = ua + ub - inter
    return inter / denom if denom > 0 else 0.0


def centre_in(box, region) -> bool:
    cx, cy = (box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2
    return region["x1"] <= cx <= region["x2"] and region["y1"] <= cy <= region["y2"]


def average_precision(records: List[Dict], n_positives: int) -> float:
    """101-point interpolated AP over the score-ordered detection list (COCO convention)."""
    if not n_positives:
        return float("nan")
    records = sorted(records, key=lambda r: -r["score"])
    tp = fp = 0
    points = []
    for record in records:
        if record["tp"]:
            tp += 1
        else:
            fp += 1
        points.append((tp / n_positives, tp / (tp + fp)))
    if not points:
        return 0.0
    total = 0.0
    for i in range(101):
        target = i / 100
        best = max((p for r, p in points if r >= target), default=0.0)
        total += best
    return total / 101


def read_detections(path, frame_ids, nms: float = 0.0):
    """Return (per-class detections, ALL detections per frame regardless of class).

    With `nms` above zero, class-agnostic suppression across prompt terms runs per frame first
    (see tools/dedup.py). Six near-synonymous brand terms return the same mark several times,
    and the matcher counts every repeat after the first as a false positive.
    """
    raw: Dict[str, List[Dict]] = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)["data"]
            except Exception:
                continue
            frame_id = os.path.splitext(os.path.basename(data["source_media"]))[0]
            if frame_id not in frame_ids:
                continue
            raw[frame_id].append({"box": data["frame_info"]["box"], "tag": data["tag"],
                                  "score": float(data["additional_info"].get("score", 1.0))})

    out: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    every: Dict[str, List[Dict]] = defaultdict(list)
    for frame_id, rows in raw.items():
        if nms > 0:
            rows = dedup_frame(rows, iou_threshold=nms)
        for det in rows:
            row = {**det["box"], "score": det["score"]}
            every[frame_id].append(row)
            cls = class_of_prompt(det["tag"])
            if cls in CLASSES:
                out[cls][frame_id].append(row)
    return out, every


def coverage_recall(all_dets, gt, cls, thr) -> float:
    """Fraction of ground-truth boxes of `cls` hit by ANY detection, ignoring its label."""
    hit = total = 0
    for frame_id, truth in gt.get(cls, {}).items():
        rows = all_dets.get(frame_id, [])
        for box in truth:
            total += 1
            if any(iou(det, box) >= thr for det in rows):
                hit += 1
    return hit / total if total else float("nan")


def score_run(dets, gt, ignores, cls, thr) -> Dict:
    records: List[Dict] = []
    n_positives = sum(len(v) for v in gt.get(cls, {}).values())
    matched_ious: List[float] = []

    for frame_id, rows in dets.get(cls, {}).items():
        truth = list(gt.get(cls, {}).get(frame_id, []))
        claimed = [False] * len(truth)
        regions = ignores.get(frame_id, [])
        for det in sorted(rows, key=lambda r: -r["score"]):
            best, best_iou = -1, 0.0
            for i, box in enumerate(truth):
                if claimed[i]:
                    continue
                value = iou(det, box)
                if value > best_iou:
                    best, best_iou = i, value
            if best >= 0 and best_iou >= thr:
                claimed[best] = True
                matched_ious.append(best_iou)
                records.append({"score": det["score"], "tp": True})
            else:
                # Unmatched inside an ignore region is neither right nor wrong.
                if any(centre_in(det, region) for region in regions):
                    continue
                records.append({"score": det["score"], "tp": False})

    ap = average_precision(records, n_positives)
    best = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 0.0}
    ordered = sorted(records, key=lambda r: -r["score"])
    tp = fp = 0
    for record in ordered:
        tp += record["tp"]
        fp += not record["tp"]
        recall = tp / n_positives if n_positives else 0.0
        precision = tp / (tp + fp)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if f1 > best["f1"]:
            best = {"f1": f1, "precision": precision, "recall": recall,
                    "threshold": record["score"]}
    return {"ap": ap, "positives": n_positives, "detections": len(records),
            "mean_iou": (sum(matched_ious) / len(matched_ious)) if matched_ious else float("nan"),
            **best}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default=paths.BOX_LABELS)
    parser.add_argument("--runs", default="04_brand_person_mark")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--nms", type=float, default=0.6,
                        help="class-agnostic IoU threshold for cross-prompt de-duplication "
                             "(tools/dedup.py); 0 disables it. Defaults ON because it is a "
                             "pipeline post-process, so scoring without it measures a "
                             "configuration nothing would ship")
    parser.add_argument("--json-out", default=os.path.join(HERE, "scores_boxes.json"))
    args = parser.parse_args()

    if not os.path.exists(args.labels):
        print(f"no box labels at {args.labels}.\n"
              f"Build the task with eval/box_gt/make_box_task.py, label in a browser with\n"
              f"eval/box_gt/label_boxes.html, then export box_labels.json into eval/box_gt/.",
              file=sys.stderr)
        return 1

    with open(args.labels) as handle:
        raw = json.load(handle)["frames"]

    gt: Dict[str, Dict[str, List[Dict]]] = {c: defaultdict(list) for c in CLASSES}
    ignores: Dict[str, List[Dict]] = defaultdict(list)
    frame_ids = set()
    for frame_id, entry in raw.items():
        if not entry.get("done"):
            # Only completed frames may be scored: a partially labelled frame's unlabelled
            # real objects would be counted as false positives against every detector.
            continue
        frame_ids.add(frame_id)
        for box in entry["boxes"]:
            if box["cls"] == "ignore":
                ignores[frame_id].append(box)
            elif box["cls"] in CLASSES:
                gt[box["cls"]][frame_id].append(box)

    counts = {c: sum(len(v) for v in gt[c].values()) for c in CLASSES}
    print(f"{len(frame_ids)} completed frames | " +
          " | ".join(f"{c}: {counts[c]} boxes" for c in CLASSES) +
          f" | ignore regions: {sum(len(v) for v in ignores.values())}")
    if not frame_ids:
        print("nothing marked done yet", file=sys.stderr)
        return 1

    results = {}
    for path in sorted(glob.glob(paths.runs_glob(args.runs))):
        name = os.path.basename(os.path.dirname(path))
        dets, every = read_detections(path, frame_ids, nms=args.nms)
        results[name] = {}
        for c in CLASSES:
            row = score_run(dets, gt, ignores, c, args.iou)
            row["coverage"] = coverage_recall(every, gt, c, args.iou)
            results[name][c] = row

    header = (f"{'run':<26}{'brand AP':>9}{'P':>7}{'R':>7}{'cov':>7}{'IoU':>7}   "
              f"{'person AP':>10}{'P':>7}{'R':>7}{'cov':>7}{'IoU':>7}")
    print(f"\nIoU {args.iou}\n{header}\n{'-' * len(header)}")
    for name in sorted(results, key=lambda n: -(results[n]["brand"]["ap"]
                                                + results[n]["person"]["ap"]) / 2):
        b, p = results[name]["brand"], results[name]["person"]
        print(f"{name:<26}{b['ap']:>9.3f}{b['precision']:>7.2f}{b['recall']:>7.2f}"
              f"{b['coverage']:>7.2f}{b['mean_iou']:>7.2f}   "
              f"{p['ap']:>10.3f}{p['precision']:>7.2f}{p['recall']:>7.2f}"
              f"{p['coverage']:>7.2f}{p['mean_iou']:>7.2f}")
    print("\ncov = class-agnostic coverage: ground-truth boxes hit by ANY detection, whatever "
          "it was labelled.\n      The tag map is not consulted for it, which is why it is the "
          "fair number for the\n      prompt-free backends -- and why it can replace the map "
          "entirely.")

    with open(args.json_out, "w") as handle:
        json.dump({"iou": args.iou, "frames": len(frame_ids), "gt_counts": counts,
                   "results": results}, handle, indent=1)
    print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
