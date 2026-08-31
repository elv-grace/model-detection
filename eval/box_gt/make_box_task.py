#!/usr/bin/env python3
"""Build the box-level labelling task: pick a subset, seed proposals, emit a self-contained tool.

Why box-level ground truth is now the blocking item
---------------------------------------------------
Frame presence is exhausted on this schema. brand is present in 75 frames of 100 and person in
97, so every detector scores 0.84-0.90 and 0.98-1.00 and the field sits inside a 0.06 band --
the metric is reporting the base rate, not detector skill. It also cannot see *what* was boxed,
which is the thing method 2 depends on: a crop is only useful downstream if it actually
contains the object.

Boxes fix both. Recall and precision at IoU 0.5 separate detectors that find objects from
detectors that merely fire on the right frames, and box tightness is directly the crop quality
the embedder will inherit.

What a `brand` box means
------------------------
The MARK ITSELF -- the logo, wordmark, emblem or badge -- and not the object carrying it. Box
the GAP wordmark, not the hoodie; the NFL shield, not the helmet. This is the reframing the
prompt ablation forced: asked for `sportswear` a detector returns the hoodie, asked for `logo`
it returns the wordmark, and the wordmark is the crop that retrieves against a logo pool.

A `person` box is the whole body, since the downstream pose and posture distinction lives there
rather than in a face crop.

Why 25 frames is enough
-----------------------
Frames are not the sample -- instances are. At roughly 10 people and 8 brand-bearing objects
per frame on this footage, 25 frames is on the order of 450 labelled boxes, which pins recall
to a few points. Doubling the frames would buy less than fixing the label definition would.

Why proposals are seeded rather than drawn from scratch
-------------------------------------------------------
Drawing ~450 boxes by hand is hours of work; accepting, rejecting and nudging pre-drawn ones is
tens of minutes. Proposals are the union of three deliberately unlike detectors, so the seed
is not biased toward any one of them:

    gdino         strongest on mark prompts (191 mark detections, 100% mark-like)
    owlv2         independent architecture, Apache-2.0, different failure modes
    yoloe11-text  CLIP-style text encoder, finds marks the transformers miss
    yolo11        COCO-80, independent training data -- contributes person boxes

Union, then NMS. Seeding from a single detector would bake that detector's blind spots into the
ground truth and quietly guarantee it wins.

The seed is a starting point and nothing more: anything the union missed must still be drawn by
hand, which is the part that keeps recall honest. The tool marks proposal-derived boxes so that
if the union turns out to have missed a lot, that is visible rather than hidden.

Usage
-----
    python eval/box_gt/make_box_task.py --frames 25
    python eval/box_gt/make_box_task.py --frames 25 --iou 0.55 --max-per-frame 40
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))

from PIL import Image  # noqa: E402

import paths  # noqa: E402
from schema_brand_person import class_of_prompt  # noqa: E402

# Seeds come only from runs prompted with the schema's own terms, so a `brand` proposal is a
# MARK by construction. Seeding brand from a prompt-free run would propose shoes and hoodies --
# brand-BEARING objects under the old definition, but not what a brand box means now.
SEED_RUNS = ["gdino", "owlv2", "yoloe11-text", "yolo11"]


def iou(a, b) -> float:
    x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    ub = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (ua + ub - inter) if (ua + ub - inter) > 0 else 0.0


def choose_frames(frames: List[Dict], labels: Dict, want: int) -> List[Dict]:
    """Stratify by source clip, round-robin, so no single video dominates the subset."""
    by_source: Dict[str, List[Dict]] = defaultdict(list)
    for frame in frames:
        by_source[frame["id"].rsplit("__", 1)[0]].append(frame)
    for rows in by_source.values():
        rows.sort(key=lambda f: f["id"])

    picked: List[Dict] = []
    index = 0
    sources = sorted(by_source, key=lambda s: -len(by_source[s]))
    while len(picked) < want and any(index < len(by_source[s]) for s in sources):
        for source in sources:
            if len(picked) >= want:
                break
            if index < len(by_source[source]):
                picked.append(by_source[source][index])
        index += 1
    return picked[:want]


def proposals_for(frame_ids, runs_dir, nms, cap, min_score) -> Dict[str, List[Dict]]:
    raw: Dict[str, List[Dict]] = defaultdict(list)
    for run in SEED_RUNS:
        path = os.path.join(runs_dir, run, "out.jsonl")
        if not os.path.exists(path):
            print(f"  (skipping {run}: no run found)")
            continue
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
                score = float(data["additional_info"].get("score", 1.0))
                if score < min_score:
                    continue
                cls = class_of_prompt(data["tag"])
                if cls not in ("brand", "person"):
                    continue
                raw[frame_id].append({"cls": cls, "score": score, "src": run,
                                      "tag": data["tag"], **data["frame_info"]["box"]})

    out: Dict[str, List[Dict]] = {}
    for frame_id, rows in raw.items():
        rows.sort(key=lambda r: -r["score"])
        kept: List[Dict] = []
        for row in rows:
            box = {k: row[k] for k in ("x1", "y1", "x2", "y2")}
            if (box["x2"] - box["x1"]) * (box["y2"] - box["y1"]) > 0.85:
                # Near-full-frame proposals are never useful ground truth and are exactly what
                # the degenerate visual-prompt runs produce.
                continue
            if any(k["cls"] == row["cls"] and iou(box, k) > nms for k in kept):
                continue
            kept.append({**box, "cls": row["cls"], "seed": row["src"],
                         "tag": row["tag"], "score": round(row["score"], 3)})
            if len(kept) >= cap:
                break
        out[frame_id] = kept
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--runs", default=os.path.join(paths.CURRENT, "runs"))
    parser.add_argument("--iou", type=float, default=0.55, help="NMS over seed proposals")
    parser.add_argument("--max-per-frame", type=int, default=35)
    parser.add_argument("--min-score", type=float, default=0.20)
    parser.add_argument("--out", default=os.path.join(HERE, "label_boxes.html"))
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--quality", type=int, default=82)
    parser.add_argument("--task-json", default=os.path.join(HERE, "box_task.json"))
    args = parser.parse_args()

    with open(paths.FRAMES_JSON) as handle:
        frames = json.load(handle)["frames"]
    labels = {}
    path = paths.PRESENCE_LABELS
    if os.path.exists(path):
        with open(path) as handle:
            labels = json.load(handle)["labels"]

    picked = choose_frames(frames, labels, args.frames)
    ids = {f["id"] for f in picked}
    print(f"{len(picked)} frames from {len({f['id'].rsplit('__',1)[0] for f in picked})} clips")

    seeds = proposals_for(ids, args.runs, args.iou, args.max_per_frame, args.min_score)
    total = sum(len(v) for v in seeds.values())
    per_cls = defaultdict(int)
    for rows in seeds.values():
        for row in rows:
            per_cls[row["cls"]] += 1
    print(f"{total} seed proposals ({dict(per_cls)}), "
          f"{total/max(1,len(picked)):.1f} per frame")

    payload = []
    for frame in picked:
        image_path = os.path.join(paths.FRAMESET, frame["frame"])
        # Re-encoded rather than embedded verbatim: the frozen frames are PNG, which makes a
        # 25-frame self-contained page ~28 MB and slow to open over a remote link. JPEG at
        # 1600px long side is ~6x smaller and loses nothing that matters for drawing a box.
        image = Image.open(image_path).convert("RGB")
        if max(image.size) > args.max_side:
            scale = args.max_side / max(image.size)
            image = image.resize((max(1, int(image.width * scale)),
                                  max(1, int(image.height * scale))), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=args.quality, optimize=True)
        blob = base64.b64encode(buffer.getvalue()).decode("ascii")
        payload.append({"id": frame["id"], "frame_idx": frame.get("frame_idx", 0),
                        "data": f"data:image/jpeg;base64,{blob}",
                        "boxes": seeds.get(frame["id"], [])})

    with open(args.task_json, "w") as handle:
        json.dump({"frames": [{"id": p["id"], "boxes": p["boxes"]} for p in payload]},
                  handle, indent=1)

    with open(os.path.join(HERE, "label_boxes_template.html")) as handle:
        template = handle.read()
    html = template.replace("/*__TASK__*/null", json.dumps(payload))
    with open(args.out, "w") as handle:
        handle.write(html)
    size = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out} ({size:.1f} MB, self-contained) and {args.task_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
