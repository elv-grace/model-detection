#!/usr/bin/env python3
"""Suggest `class_conf` thresholds from a low-threshold calibration run.

Why this exists
---------------
YOLOE scores a region by similarity between its region embedding and the prompt's text
embedding. Generic prompts ("standalone object", "prop", "person") sit high and flat against
nearly every region, while specific ones ("billboard") are sharply peaked. Under a single
global `conf` gate the generic classes consume the whole `max_detections` budget and bury the
classes you actually care about. `class_conf` fixes that, but the shipped default
(`{"object": 0.5}`) is a placeholder, not a measurement.

Step 1 — produce ungated scores. The gates you are trying to set would truncate the very
distribution you need to see, so run with them effectively off:

    ./buildscripts/testers/test-model.sh --params '{
      "fps": 0.2, "conf": 0.01, "class_conf": {}, "max_detections": 300,
      "min_crop_pixels": 1, "cross_class_nms_iou": 1.0
    }'

(Low fps keeps the run cheap; you want frame *diversity*, not frame count. Dedupe stages 1
and 2 stay on, so what you measure is the post-dedupe distribution the real run will see.)

Step 2 — analyze:

    python scripts/calibrate_class_conf.py test-output/out.jsonl --max-per-frame 3

What the criterion is, and is not
---------------------------------
This is a **detection-budget** criterion: it caps how many detections per frame each class is
allowed, so no class can crowd out the others, and reports the score quantile that achieves
it. It is label-free, which is why it is runnable today.

It is **not** a precision criterion. It cannot tell you whether a kept detection is correct.
For that you need annotations on a sample; then pick the per-class threshold that maximizes
F1 instead of the one that hits a count target. Treat this output as a defensible starting
point that fixes the crowding-out problem, not as a substitute for labelled evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# Percentiles reported per class, to show the shape of the distribution rather than one number.
_REPORT_QUANTILES = (0.50, 0.75, 0.90, 0.95, 0.99)


def _quantile(sorted_values: List[float], q: float) -> float:
    """Nearest-rank quantile. Avoids a numpy dependency for a script."""
    if not sorted_values:
        return float("nan")
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def read_scores(paths: List[str]) -> Tuple[Dict[str, List[float]], int]:
    """Return ({parent_term: [scores]}, number of distinct frames seen).

    Frames are counted as distinct (source_media, frame_idx) pairs so the per-frame rates are
    right even when several files are tagged in one run.
    """
    scores: Dict[str, List[float]] = defaultdict(list)
    frames = set()
    skipped_no_score = 0

    for path in paths:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("type") != "tag":
                    continue

                data = row.get("data", {})
                info = data.get("additional_info") or {}
                # The optional whole-frame vector is not a detection and has no score.
                if info.get("kind") == "frame":
                    continue

                frame_info = data.get("frame_info") or {}
                frames.add((data.get("source_media"), frame_info.get("frame_idx")))

                score = info.get("score")
                if score is None:
                    skipped_no_score += 1
                    continue
                scores[data.get("tag") or "<untagged>"].append(float(score))

    if skipped_no_score:
        print(
            f"warning: {skipped_no_score} tag(s) had no additional_info.score and were "
            f"ignored — was this output produced by this model?",
            file=sys.stderr,
        )
    return scores, len(frames)


def suggest(
    scores: Dict[str, List[float]], frames: int, max_per_frame: float, floor: float
) -> Tuple[Dict[str, float], List[str]]:
    """Lowest threshold (from observed scores) that caps each class at max_per_frame.

    Returns (overrides, governed_by_global). A class lands in the second list when the
    threshold it needs is at or below `floor` — the global `conf` already caps it, so an
    explicit entry would be redundant. Emitting one anyway is not harmless: `class_conf`
    replaces rather than merges, so redundant entries make the dict harder to keep correct.
    """
    overrides: Dict[str, float] = {}
    governed_by_global: List[str] = []
    budget = max(1, int(round(max_per_frame * frames)))

    for label, values in scores.items():
        if len(values) <= budget:
            continue  # already under budget at any threshold
        # Keep the top `budget` scores: the threshold is the lowest of those.
        descending = sorted(values, reverse=True)
        # Nudge above the boundary score so the cap is not overshot by ties.
        threshold = round(min(0.99, descending[budget - 1] + 0.005), 3)
        if threshold <= floor:
            governed_by_global.append(label)
        else:
            overrides[label] = threshold
    return overrides, governed_by_global


def find_silenced(
    scores: Dict[str, List[float]], conf: float, class_conf: Dict[str, float]
) -> List[Tuple[str, float, float]]:
    """Classes whose best observed score cannot clear the gate that would apply to them.

    This is the failure mode a percentile table hides: a gate set above a class's entire
    score range does not trim that class, it removes it, and the output looks like "this
    content has no logos" rather than "the threshold is wrong". Open-vocabulary prompts make
    it easy to hit — YOLOE scores concrete COCO-vocabulary nouns ("person") far higher than
    abstract or compositional ones ("writing surface", "standalone object"), so one global
    threshold tuned for the former silences the latter.
    """
    silenced = []
    for label, values in scores.items():
        gate = class_conf.get(label, conf)
        best = max(values)
        if best < gate:
            silenced.append((label, best, gate))
    return sorted(silenced, key=lambda row: row[1])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("jsonl", nargs="+", help="tagger output(s) from a low-threshold run")
    parser.add_argument(
        "--max-per-frame",
        type=float,
        default=3.0,
        help="detections per frame each class is allowed (default 3)",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=0.25,
        help="the global conf in use; thresholds at or below it are left to it (default 0.25)",
    )
    parser.add_argument(
        "--class-conf",
        type=str,
        default="{}",
        help="class_conf currently configured, as JSON, for the silenced-class check",
    )
    args = parser.parse_args()
    try:
        current_class_conf = json.loads(args.class_conf)
    except json.JSONDecodeError as exc:
        print(f"--class-conf is not valid JSON: {exc}", file=sys.stderr)
        return 2

    scores, frames = read_scores(args.jsonl)
    if not scores:
        print("no scored detections found", file=sys.stderr)
        return 1
    if frames == 0:
        print("no frames identified (missing frame_info?)", file=sys.stderr)
        return 1

    total = sum(len(v) for v in scores.values())
    print(f"\n{frames} frames, {total} detections ({total / frames:.1f} per frame)\n")

    header = f"{'class':<10} {'count':>7} {'/frame':>7} {'share':>6} " + " ".join(
        f"{'p' + str(int(q * 100)):>6}" for q in _REPORT_QUANTILES
    )
    print(header)
    print("-" * len(header))
    for label, values in sorted(scores.items(), key=lambda kv: -len(kv[1])):
        ordered = sorted(values)
        quantiles = " ".join(
            f"{_quantile(ordered, q):>6.3f}" for q in _REPORT_QUANTILES
        )
        print(
            f"{label:<10} {len(values):>7} {len(values) / frames:>7.1f} "
            f"{len(values) / total:>5.0%} {quantiles}"
        )

    # Surface silenced classes FIRST: a gate above a class's entire score range removes the
    # class rather than trimming it, and the output then looks like "this content has no
    # logos" instead of "the threshold is wrong". No amount of budget tuning reveals that.
    silenced = find_silenced(scores, args.floor, current_class_conf)
    if silenced:
        print("!!! SILENCED CLASSES — gated above every score they achieved !!!")
        for label, best, gate in silenced:
            source = "class_conf" if label in current_class_conf else "global conf"
            print(
                f"  {label:<10} best score {best:.3f} < gate {gate:.3f} ({source})"
                f"  -> 0 detections survive"
            )
        print(
            "  These gates delete the class instead of trimming it. Lower them, or fix the\n"
            "  prompt phrasing in prompts.py — a class scoring this low usually means the\n"
            "  prompt does not describe the content, not that the content is absent.\n"
        )

    overrides, governed = suggest(scores, frames, args.max_per_frame, args.floor)

    print(f"--- suggested class_conf (cap {args.max_per_frame}/frame, global conf {args.floor}) ---")
    if not overrides:
        print("no overrides needed: every class is under budget at the global conf")
    else:
        print(json.dumps(overrides, indent=2, sort_keys=True))
        # class_conf REPLACES the default dict rather than merging into it, so a partial
        # paste silently ungates whatever it omits. This dict is complete as printed:
        # classes absent from it are under budget and correctly governed by the global conf.
        print(
            "\n(complete as printed — class_conf REPLACES the default rather than merging,\n"
            " so paste it whole. Classes omitted are under budget at the global conf.)"
        )
        print("\neffect:")
        for label, threshold in sorted(overrides.items()):
            kept = sum(1 for v in scores[label] if v >= threshold)
            before = len(scores[label])
            print(
                f"  {label:<10} {before:>6} -> {kept:>6} detections "
                f"({before / frames:>5.1f} -> {kept / frames:>4.1f} per frame)"
            )
    if governed:
        print(f"\nalready capped by the global conf, no entry needed: {', '.join(sorted(governed))}")

    print(
        "\nReminder: this is a detection-budget criterion, not a precision one. "
        "It stops generic classes crowding out specific ones; it cannot tell you whether "
        "a kept detection is correct. Use labelled samples for that."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
