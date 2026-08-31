#!/usr/bin/env python3
"""Cost, breadth and (optionally) frame-presence for the brand / person runs.

What this is for now
--------------------
It used to be the ranking instrument. It is not any more -- box_gt/score_boxes.py ranks, and
box_gt/operating_point.py picks the threshold. What survives here needs no ground truth at all
and is still wanted: throughput, latency, detection counts, and how many distinct tags a backend
emits outside the prompt list.

The frame-presence table is behind --presence and should be read as a diagnostic, for two
reasons. Its brand labels on 75 of 100 frames are derived from a superseded 8-class pass, which
asked whether a frame had a PROMINENT logo -- the box pass caught small marks it had missed, and
disagreed on 4 of the 25 frames where both exist, always in that direction. And the metric
saturates regardless: person is present in 97 frames of 100, so it has three negatives.

Frame-level presence, when you do ask for it
--------------------------------------------
For each class, did the detector fire on the frames where a human says the class is present,
and stay quiet on the frames where it is not.

    recall    = frames correctly fired on / frames where the class is present
    fp_rate   = frames wrongly fired on   / frames where the class is absent
    precision = frames correctly fired on / frames fired on
    F1        = harmonic mean of precision and recall

All dimensionless, so they compare across models whose raw scores do not. Detector scores are
not comparable across models, so every run is swept over its own threshold grid and reported
at its own best operating point.

One scoring mode: the detector's own label
------------------------------------------
A detection counts for a class when its tag is one of the schema's seven prompt terms. That is
total over what a promptable detector can emit -- it can only answer with terms it was given --
so no mapping is involved and none is needed.

The prompt-free and closed backends still appear, and legitimately: `person` and `logo` are
terms their own vocabularies happen to contain, so those detections are counted directly. What
is NOT done is crediting a synonym -- yolo11's `sports ball` earns nothing, and its brand score
of 0.00 is a real measurement (COCO-80 has no logo class, so it cannot find marks) rather than
an artefact. A backend that emitted no prompt term at all would be reported as unscored instead
of shown as a row of zeros.

Bridging synonyms used to be done with a hand-built map from those vocabularies onto
brand/person. It
was retired (see tools/deprecated/README.md): built from keyword families with veto lists, it
could not survive compounds -- an activity veto swallowed "fishing boat" and "wedding cake",
and a person rule captured "record player" -- and a single review pass found around ninety such
errors across 4,607 terms. Each correction moved the errors rather than removing them.

It turned out to be unnecessary as well as unreliable. Under the mark schema there are no
synonyms left to bridge: `logo` and `person` are the terms in question and both vocabularies
contain them verbatim.

box_gt/score_boxes.py goes further, with class-agnostic IoU coverage: of the ground-truth boxes
of a class, how many were hit by ANY detection, whatever it was labelled. That is the fair
measure for a prompt-free backend that boxes the right object under a different name, and it is
closer to what the pipeline does with a label, which is nothing.

What this does NOT measure
--------------------------
Box quality, and whether a firing was on the right object. A detector can score perfectly here
while boxing the wrong thing. Read the contact sheets alongside these tables.

It also cannot say much about `person` on this frame set: person is present in 97 of 100
frames, so there are only 3 negatives and precision is near 1.0 for anything that fires at
all. That ceiling is a property of the footage, not of the detectors, and the report says so
rather than pretending the resulting F1 ranks them.

Usage
-----
    python eval/tools/score_bp.py
    python eval/tools/score_bp.py --runs 03_prompt_ablation/runs_mark
    python eval/tools/score_bp.py --visual
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paths  # noqa: E402
from schema_brand_person import class_of_prompt, derive  # noqa: E402

CLASSES = ["brand", "person"]


def box_gt_presence() -> Dict[str, Set[str]]:
    """Exact presence for the frames that have box ground truth.

    Presence is a strictly weaker statement than a box: if a frame carries a `brand` box then
    brand is present, and if a fully-labelled frame carries none then it is absent. So wherever
    box ground truth exists it is not merely better evidence than the derived label, it is the
    same question already answered exactly, and it overrides.

    It matters in one direction here. The 8-class pass asked whether a frame had a prominent
    logo; the box pass caught every mark including small ones, and found brand in four frames
    the derived label calls empty (and none the other way). Those four were scored as false
    positives against every detector that correctly found the mark.
    """
    if not os.path.exists(paths.BOX_LABELS):
        return {}
    with open(paths.BOX_LABELS) as handle:
        frames = json.load(handle)["frames"]
    out: Dict[str, Set[str]] = {}
    for frame_id, entry in frames.items():
        if not entry.get("done"):
            # A partially labelled frame proves presence but never absence.
            continue
        out[frame_id] = {b["cls"] for b in entry["boxes"] if b["cls"] in CLASSES}
    return out


def load_labels(path: str, derived_out: str = None) -> Tuple[Dict[str, Set[str]], bool, int]:
    """Return (frame_id -> present classes, is_provisional, frames overridden by box GT)."""
    explicit = os.path.join(paths.FRAMESET, "brand_person_labels.json")
    if os.path.exists(explicit):
        with open(explicit) as handle:
            data = json.load(handle)
        return ({k: set(v["present"]) for k, v in data["labels"].items()}, False, 0)

    # Fall back to deriving from the 8-class labels so numbers exist before the review lands.
    with open(path) as handle:
        old = json.load(handle)
    labels = {k: set(derive(v["present"])) for k, v in old["labels"].items()}

    corrected = 0
    for frame_id, exact in box_gt_presence().items():
        if frame_id in labels:
            corrected += labels[frame_id] != exact
            labels[frame_id] = exact

    if derived_out:
        with open(derived_out, "w") as handle:
            json.dump({"schema": CLASSES, "provisional": True,
                       "derived_from": os.path.basename(path),
                       "labels": {k: {"present": sorted(v), "unsure": False}
                                  for k, v in labels.items()}}, handle, indent=1)
    return labels, True, corrected


def read_run(path: str, valid: Set[str]) -> Tuple[Dict[str, List[Tuple[str, float]]], int]:
    """frame_id -> [(tag, score)], and the total detection count."""
    fired: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    total = 0
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
            if frame_id not in valid:
                continue
            fired[frame_id].append((data["tag"],
                                    float(data["additional_info"].get("score", 1.0))))
            total += 1
    return fired, total


def score_at(fired, labels, cls, threshold) -> Dict:
    tp = fp = fn = tn = 0
    for frame_id, present in labels.items():
        hit = any(class_of_prompt(tag) == cls
                  for tag, score in fired.get(frame_id, []) if score >= threshold)
        if cls in present:
            tp += hit
            fn += not hit
        else:
            fp += hit
            tn += not hit
    positives, negatives = tp + fn, fp + tn
    recall = tp / positives if positives else float("nan")
    fp_rate = fp / negatives if negatives else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    # A class with no positives cannot be scored; a class with positives but no true positives
    # scores 0, NOT nan. Treating that as "not applicable" would reward a detector for never
    # attempting the class -- which is exactly the bug that put yolo11 first in the 8-class
    # sweep before it was fixed.
    if not positives:
        f1 = float("nan")
    elif tp == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": recall, "fp_rate": fp_rate, "precision": precision, "f1": f1,
            "positives": positives, "negatives": negatives}


def sweep(fired, labels, cls, grid, target_recall=0.9) -> Dict:
    """Best-F1 operating point, plus the cost of hitting a fixed recall.

    Best-F1 turns out to be nearly useless on this schema. Both targets are near-ubiquitous
    in this footage -- brand in 58 frames of 100, person in 97 -- so a detector that fires
    indiscriminately already scores ~0.86 on brand and ~0.99 on person, and the whole field
    lands inside a 0.06 band. F1 is not measuring detector quality there, it is measuring the
    base rate.

    `fp_at_recall` is the discriminative number: sweep down until recall reaches
    target_recall, and report the false-positive rate paid to get there. Detectors that only
    reach high recall by firing on everything are separated from ones that do it selectively,
    which is exactly the distinction best-F1 hides.
    """
    best = None
    at_target = None
    for threshold in grid:
        row = score_at(fired, labels, cls, threshold)
        if best is None or (row["f1"] == row["f1"] and row["f1"] > best["f1"]):
            best = row
        if row["recall"] == row["recall"] and row["recall"] >= target_recall:
            # Higher thresholds come later in the grid, so the last qualifying row is the
            # most selective one that still meets the recall target.
            if at_target is None or row["fp_rate"] <= at_target["fp_rate"]:
                at_target = row
    best = dict(best)
    best["target_recall"] = target_recall
    best["fp_at_recall"] = at_target["fp_rate"] if at_target else float("nan")
    best["threshold_at_recall"] = at_target["threshold"] if at_target else float("nan")
    return best


def breadth(fired) -> Dict:
    """How many distinct tags the run emitted, and how many are outside the prompt list.

    No taxonomy is applied. A promptable backend cannot emit anything but its prompts, so
    `off_prompt` is zero for those by construction; for a prompt-free backend it counts the
    breadth of its native vocabulary on this footage. Whether those tags name objects or scene
    words is deliberately not judged here -- that judgement was the tag map, and it is gone.
    """
    distinct: Set[str] = set()
    off_prompt: Set[str] = set()
    for rows in fired.values():
        for tag, _ in rows:
            key = tag.lower()
            distinct.add(key)
            if class_of_prompt(key) == "OTHER":
                off_prompt.add(key)
    return {"distinct_tags": len(distinct), "off_prompt_tags": len(off_prompt)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="04_brand_person_mark")
    parser.add_argument("--visual", action="store_true",
                        help="score the per-exemplar visual-prompt runs instead (one class per run)")
    parser.add_argument("--labels", default=paths.PRESENCE_LABELS)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--presence", action="store_true",
                        help="also print the frame-presence table. Off by default: 75 of its "
                             "100 brand labels are derived from a superseded 8-class pass, and "
                             "the metric saturates regardless. Box ground truth replaced it.")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if args.visual:
        args.runs = "02_brand_person_101/runs_visual"

    with open(paths.FRAMES_JSON) as handle:
        frames = json.load(handle)["frames"]
    valid = {f["id"] for f in frames}

    labels, provisional, corrected = load_labels(
        args.labels, derived_out=os.path.join(paths.FRAMESET, "brand_person_labels_derived.json"))
    labels = {k: v for k, v in labels.items() if k in valid}

    counts = {c: sum(1 for v in labels.values() if c in v) for c in CLASSES}
    print(f"{len(labels)} labelled frames | " +
          " | ".join(f"{c}: {counts[c]} present, {len(labels)-counts[c]} absent"
                     for c in CLASSES))
    if provisional:
        from schema_brand_person import DERIVE_BRAND_FROM
        print("LABELS ARE PROVISIONAL: derived from the 8-class labels (brand = "
              f"{' | '.join(DERIVE_BRAND_FROM)}). Box-level ground truth supersedes them.")
        if corrected:
            print(f"  {corrected} of those frames were CORRECTED from box ground truth, which is "
                  f"exact where it exists.\n"
                  f"  The remaining frames keep the derived label, so the two halves of this "
                  f"table are not labelled to the same standard --\n"
                  f"  which is the reason box AP, not presence F1, is the headline metric.")
    for cls in CLASSES:
        absent = len(labels) - counts[cls]
        if absent < 10:
            print(f"  NOTE: only {absent} negative frames for `{cls}` -- precision and F1 are "
                  f"near-ceiling for anything that fires, so rank on recall and fp_rate.")

    grid = [i / args.steps for i in range(args.steps)]
    run_paths = sorted(glob.glob(paths.runs_glob(args.runs)))
    if not run_paths:
        print(f"no runs under {args.runs}", file=sys.stderr)
        return 1

    latency: Dict[str, Dict] = {}
    for path in glob.glob(os.path.join(paths.experiment(args.runs), "**", "summary*.json"),
                          recursive=True):
        with open(path) as handle:
            for entry in json.load(handle):
                key = entry.get("backend") or entry.get("run")
                if key:
                    latency[key] = entry

    results: Dict[str, Dict] = {}
    unscored: List[str] = []

    for path in run_paths:
        name = os.path.basename(os.path.dirname(path))
        fired, total = read_run(path, valid)
        entry = {"detections": total, "breadth": breadth(fired)}
        meta = latency.get(name, {})
        for key in ("throughput_ms_per_frame", "throughput_fps",
                    "latency_ms_median", "latency_ms_p90"):
            if key in meta:
                entry[key] = meta[key]
        entry["label"] = {c: sweep(fired, labels, c, grid) for c in CLASSES}
        # A backend that never emits a prompt term is answering from its own vocabulary, which
        # this scorer cannot evaluate. Say so rather than printing a row of zeros.
        if entry["breadth"]["distinct_tags"] and not (
                entry["breadth"]["distinct_tags"] - entry["breadth"]["off_prompt_tags"]):
            unscored.append(name)
            continue
        results[name] = entry

    mode = "label"

    # ---- the label-free table, which is the default because it needs no ground truth ----
    print(f"\n{'=' * 92}\nCOST AND BREADTH  (no labels involved)\n{'=' * 92}")
    cost = (f"{'run':<26} {'ms/frame':>9} {'fps':>7} {'latency':>9} {'p90':>7} "
            f"{'detections':>11} {'tags':>6} {'off-prompt':>11}")
    print(cost)
    print("-" * len(cost))
    for name in sorted(results, key=lambda n: results[n].get("throughput_ms_per_frame", 9e9)):
        e = results[name]
        print(f"{name:<26} {e.get('throughput_ms_per_frame', float('nan')):>9.0f} "
              f"{e.get('throughput_fps', float('nan')):>7.1f} "
              f"{e.get('latency_ms_median', float('nan')):>9.0f} "
              f"{e.get('latency_ms_p90', float('nan')):>7.0f} "
              f"{e['detections']:>11,} {e['breadth']['distinct_tags']:>6} "
              f"{e['breadth']['off_prompt_tags']:>11}")

    if not args.presence:
        print("\nFrame-presence scoring is OFF by default -- pass --presence to print it.")
        print("Its brand labels are derived from a superseded 8-class pass on 75 of 100 frames")
        print("and it saturates besides (person present in 97 of 100). Rank detectors with")
        print("box_gt/score_boxes.py, and choose a threshold with box_gt/operating_point.py.")
        out = args.json_out or os.path.join(paths.experiment(args.runs), "scores.json")
        with open(out, "w") as handle:
            json.dump({"presence_scored": False, "frames": len(labels),
                       "results": results, "not_scored": unscored}, handle, indent=1)
        print(f"\nwrote {out}")
        return 0

    print(f"\n{'=' * 100}\nLABEL MODE  (the detector's own tag must be one of the "
          f"schema's prompt terms)\n{'=' * 100}")
    header = (f"{'run':<26} {'brand F1':>9} {'rec':>6} {'fp@.9':>7} "
              f"{'person F1':>10} {'rec':>6} {'fp@.9':>7} {'ms/fr':>7} {'lat':>6} "
              f"{'tags':>6} {'off':>5}")
    print(header)
    print("-" * len(header))

    def rank_key(name):
        # Primary key is fp@.9 -- the false-positive rate at the most selective threshold that
        # still reaches 90% brand recall -- because F1 saturates on this frame set and fp@.9
        # keeps some separation. Detectors that never reach 90% recall have no such threshold
        # and report nan; they sort last as a group, and within it by brand F1 descending, so
        # the tail is ordered by something rather than by glob order.
        brand = results[name][mode]["brand"]
        value = brand["fp_at_recall"]
        reached = value == value
        return (0 if reached else 1, value if reached else 0.0, -brand["f1"])

    for name in sorted(results, key=rank_key):
        entry = results[name]
        b, p = entry[mode]["brand"], entry[mode]["person"]
        print(f"{name:<26} {b['f1']:>9.2f} {b['recall']:>6.2f} {b['fp_at_recall']:>7.2f} "
              f"{p['f1']:>10.2f} {p['recall']:>6.2f} {p['fp_at_recall']:>7.2f} "
              f"{entry.get('throughput_ms_per_frame', float('nan')):>7.0f} "
              f"{entry.get('latency_ms_median', float('nan')):>6.0f} "
              f"{entry['breadth']['distinct_tags']:>6} "
              f"{entry['breadth']['off_prompt_tags']:>5}")

    if unscored:
        print(f"\nNOT SCORED ({len(unscored)}): {', '.join(sorted(unscored))}")
        print("  These answer from their own vocabulary, not the prompt list, so against this")
        print("  schema they would score zero by construction. Score them with")
        print("  box_gt/score_boxes.py, whose class-agnostic coverage needs no tag mapping.")

    out = args.json_out or os.path.join(paths.experiment(args.runs), "scores.json")
    with open(out, "w") as handle:
        json.dump({"provisional_labels": provisional, "label_counts": counts,
                   "frames": len(labels), "results": results, "not_scored": unscored},
                  handle, indent=1)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
