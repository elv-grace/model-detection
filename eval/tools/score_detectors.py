#!/usr/bin/env python3
"""Rank detector runs against per-frame presence labels.

Why frame-level presence rather than box mAP
--------------------------------------------
Detection scores are not comparable across models: YOLOE's text-similarity scores, the
prompt-free head's scores, and YOLO11's sigmoid class scores are on different scales with
different calibration, so "detections per frame" and score percentiles cannot rank them. Box
mAP would be comparable but needs box annotation, which is ~10x the labelling effort.

Frame-level presence is the cheap middle: for each class, did the detector fire on the frames
where a human says the class is present, and did it stay quiet on the frames where it is not.
Both numbers are dimensionless and directly comparable across models.

    recall  = frames correctly fired on / frames where the class is present
    fp_rate = frames wrongly fired on   / frames where the class is absent
    F1      = harmonic mean of precision and recall, over frames

What this does NOT measure: box quality, and whether a firing was on the *right object*. A
detector can score perfectly here while boxing the wrong thing — exactly what happened with
"writing surface" matching desks instead of the chalkboard. Always read the contact sheets
(eval/contact_sheet.py) alongside these tables.

Usage
-----
    python eval/score_detectors.py \
        --labels eval/presence_labels.json \
        yoloe11-text=eval/runs/yoloe11-text/out.jsonl \
        yoloe11-pf=eval/runs/yoloe11-pf/out.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))


def load_manifest() -> Tuple[List[Dict], List[Dict]]:
    path = os.path.join(HERE, "frames.json")
    with open(path) as handle:
        data = json.load(handle)
    return data["frames"], data["classes"]


def build_tag_map(classes: List[Dict], aliases: Dict[str, List[str]]) -> Dict[str, str]:
    """tag string -> schema class key.

    Text-prompted runs emit the parent term directly, so the key maps to itself. Prompt-free
    runs emit raw vocabulary terms, which is why every `prompts` entry also maps in: those are
    exact vocabulary terms chosen for that reason. `--aliases` extends this without editing
    the schema.
    """
    mapping: Dict[str, str] = {}
    for spec in classes:
        key = spec["key"]
        mapping[key] = key
        for term in spec.get("prompts", []):
            mapping[term] = key
    for key, terms in aliases.items():
        for term in terms:
            mapping[term] = key
    return mapping


def read_run(
    path: str, valid_ids: Set[str], min_score: float = 0.0
) -> Tuple[Dict[str, Set[str]], Counter, Counter, int]:
    """Return (frame_id -> set of raw tags fired, tag counts, unmapped-tag counts, n detections).

    `source_media` is the frame file the detector was given, so its basename without extension
    is the manifest frame id. `min_score` gates detections, which is how the threshold sweep
    evaluates one run at many operating points.
    """
    fired: Dict[str, Set[str]] = defaultdict(set)
    tag_counts: Counter = Counter()
    total = 0
    unknown_frames: Counter = Counter()

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
            if info.get("kind") == "frame":
                continue

            frame_id = os.path.splitext(os.path.basename(data.get("source_media", "")))[0]
            if frame_id not in valid_ids:
                unknown_frames[frame_id] += 1
                continue

            if float(info.get("score") or 0.0) < min_score:
                continue

            tag = data.get("tag") or ""
            # Prompt-free runs put the vocabulary term in `tag`; text runs put the parent term
            # in `tag` and the phrasing in additional_info.prompt. Prefer `tag`, fall back.
            if not tag:
                tag = info.get("prompt") or ""
            fired[frame_id].add(tag)
            tag_counts[tag] += 1
            total += 1

    return fired, tag_counts, unknown_frames, total


def score(
    fired: Dict[str, Set[str]],
    labels: Dict[str, Dict],
    classes: List[Dict],
    tag_map: Dict[str, str],
    frame_ids: List[str],
) -> Dict[str, Dict]:
    """Per-class frame-level recall / fp_rate / precision / F1."""
    # Frames marked unsure are excluded entirely: a disputed ground truth would penalise or
    # reward detectors arbitrarily.
    usable = [
        fid for fid in frame_ids
        if fid in labels and not labels[fid].get("unsure") and not labels[fid].get("skipped")
    ]

    results: Dict[str, Dict] = {}
    for spec in classes:
        key = spec["key"]
        positives = [f for f in usable if key in labels[f].get("present", [])]
        negatives = [f for f in usable if key not in labels[f].get("present", [])]

        def hit(fid: str) -> bool:
            return any(tag_map.get(t) == key for t in fired.get(fid, ()))

        tp = sum(1 for f in positives if hit(f))
        fp = sum(1 for f in negatives if hit(f))
        recall = tp / len(positives) if positives else float("nan")
        fp_rate = fp / len(negatives) if negatives else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")

        # F1 is only genuinely undefined when the class has no positives — there is then no
        # ground truth to score against. Everything else must resolve to a number.
        #
        # In particular, a detector that emits NOTHING for a class has precision 0/0 = nan,
        # but its recall is 0, so its F1 is 0 — not undefined. Treating that as n/a and
        # dropping it from the macro average rewards a model for failing to try: YOLO11
        # scored n/a on logo/sign_or_text/screen/ball, and averaging over the 2 classes it
        # did attempt ranked it first over models that attempted all 8.
        if not positives:
            f1 = float("nan")
        elif tp == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        results[key] = {
            "n_pos": len(positives), "n_neg": len(negatives),
            "tp": tp, "fp": fp,
            "recall": recall, "fp_rate": fp_rate, "precision": precision, "f1": f1,
        }
    results["_usable_frames"] = {"n": len(usable)}
    return results


def fmt(value: float) -> str:
    return "  n/a" if value != value else f"{value:5.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", nargs="+", metavar="NAME=PATH",
                        help="detector runs to score, e.g. yoloe11-pf=eval/runs/pf/out.jsonl")
    parser.add_argument("--labels", default=os.path.join(HERE, "presence_labels.json"),
                        help="presence labels exported from label_presence.html")
    parser.add_argument("--aliases", default="{}",
                        help='JSON {class_key: [extra tag strings]} to extend the tag mapping')
    parser.add_argument("--show-unmapped", type=int, default=12,
                        help="how many unmapped tags to list per run (0 disables)")
    parser.add_argument("--sweep", action="store_true",
                        help="pick each detector's own best threshold by macro F1 (recommended: "
                             "raw scores are not comparable across models)")
    parser.add_argument("--min-support", type=int, default=5,
                        help="classes with fewer positives or negatives than this are marked "
                             "low-support and excluded from macro F1 (default 5)")
    args = parser.parse_args()

    frames, classes = load_manifest()
    frame_ids = [f["id"] for f in frames]
    valid_ids = set(frame_ids)

    if not os.path.exists(args.labels):
        print(f"no labels at {args.labels}\n"
              f"export them from eval/label_presence.html first", file=sys.stderr)
        return 1
    with open(args.labels) as handle:
        labels = json.load(handle)["labels"]

    tag_map = build_tag_map(classes, json.loads(args.aliases))

    parsed: List[Tuple[str, str]] = []
    for spec in args.runs:
        if "=" not in spec:
            print(f"expected NAME=PATH, got {spec!r}", file=sys.stderr)
            return 2
        name, _, path = spec.partition("=")
        if not os.path.exists(path):
            print(f"missing run output: {path}", file=sys.stderr)
            return 2
        parsed.append((name, path))

    all_results: Dict[str, Dict] = {}
    summaries: List[Tuple[str, float]] = []

    def macro_at(path: str, threshold: float) -> float:
        fired_at, _, _, _ = read_run(path, valid_ids, threshold)
        res = score(fired_at, labels, classes, tag_map, frame_ids)
        values = [
            res[s["key"]]["f1"] for s in classes
            if min(res[s["key"]]["n_pos"], res[s["key"]]["n_neg"]) >= args.min_support
            and res[s["key"]]["f1"] == res[s["key"]]["f1"]
        ]
        return sum(values) / len(values) if values else float("nan")

    for name, path in parsed:
        threshold = 0.0
        if args.sweep:
            # Detector scores are not comparable across models, so a single shared threshold
            # would rank them by score inflation rather than by quality. Give each its own
            # best operating point instead, and report which one it needed.
            grid = [i / 100 for i in range(0, 96, 2)]
            scored = [(macro_at(path, t), t) for t in grid]
            best = max((v, t) for v, t in scored if v == v)
            threshold = best[1]

        fired, tag_counts, unknown, total = read_run(path, valid_ids, threshold)
        results = score(fired, labels, classes, tag_map, frame_ids)
        all_results[name] = results

        usable = results["_usable_frames"]["n"]
        gate = f", best threshold {threshold:.2f}" if args.sweep else ""
        print(f"\n{'=' * 72}\n{name}   ({total} detections over {usable} usable frames{gate})"
              f"\n{'=' * 72}")
        print(f"{'class':<15} {'n+':>4} {'n-':>4} {'TP':>4} {'FP':>4} "
              f"{'recall':>6} {'fp_rt':>6} {'prec':>6} {'F1':>6}")
        print("-" * 72)
        macro: List[float] = []
        low_support: List[str] = []
        for spec in classes:
            key = spec["key"]
            r = results[key]
            # A class with a handful of positives (or negatives) has metric granularity
            # coarser than the differences we are trying to detect: at n_pos=2, recall moves
            # in steps of 0.5. Report it, but keep it out of the number that ranks detectors.
            thin = min(r["n_pos"], r["n_neg"]) < args.min_support
            mark = " *" if thin else ""
            print(f"{key:<15} {r['n_pos']:>4} {r['n_neg']:>4} {r['tp']:>4} {r['fp']:>4} "
                  f"{fmt(r['recall'])} {fmt(r['fp_rate'])} {fmt(r['precision'])} "
                  f"{fmt(r['f1'])}{mark}")
            if thin:
                low_support.append(key)
            elif r["f1"] == r["f1"]:
                macro.append(r["f1"])
        macro_f1 = sum(macro) / len(macro) if macro else float("nan")
        print("-" * 72)
        print(f"{'macro F1':<15} {fmt(macro_f1)}   (mean over {len(macro)} well-supported classes)")
        if low_support:
            print(f"  * low support (<{args.min_support} pos or neg), excluded from macro F1: "
                  f"{', '.join(low_support)}")
        summaries.append((name, macro_f1))

        if unknown:
            print(f"\n  warning: {sum(unknown.values())} detections on {len(unknown)} frame ids "
                  f"not in the manifest — was this run made over eval/frames/?")

        if args.show_unmapped:
            unmapped = Counter({t: c for t, c in tag_counts.items() if t not in tag_map})
            if unmapped:
                # For prompt-free runs this list IS the generic-object signal: how many
                # distinct vocabulary terms fire, and whether they look plausible.
                print(f"\n  {len(unmapped)} distinct tags outside the schema "
                      f"({sum(unmapped.values())} detections) — the generic-object bucket:")
                for tag, count in unmapped.most_common(args.show_unmapped):
                    print(f"    {count:>6}  {tag}")

    if len(summaries) > 1:
        print(f"\n{'=' * 72}\nranking by macro F1\n{'=' * 72}")
        for rank, (name, value) in enumerate(
            sorted(summaries, key=lambda kv: (-(kv[1] if kv[1] == kv[1] else -1))), 1
        ):
            print(f"  {rank}. {name:<28} {fmt(value)}")

    print("\nFrame-level presence only: this cannot see box quality or whether a firing was on "
          "the right object.\nRead eval/contact_sheet.py output alongside these numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
