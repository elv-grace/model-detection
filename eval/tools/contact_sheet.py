#!/usr/bin/env python3
"""Render top-scoring detections per class into contact sheets for visual review.

Why this exists
---------------
Frame-level presence metrics (eval/score_detectors.py) cannot see *what* was boxed. A detector
scores perfectly on `board` presence while boxing desks instead of the chalkboard — which is
exactly what happened with the prompt "writing surface", and the per-class score table implied a
mislabelling that box overlap later disproved. Aggregates mislead; pixels do not.

Each cell shows the detection box (drawn in red) inside a slightly wider crop, so both *what*
was found and *how well it was framed* are visible. Cells are ordered by score, so reading the
sheet left-to-right is reading precision@K.

Usage
-----
    python eval/contact_sheet.py eval/runs/yoloe11-pf/out.jsonl --name yoloe11-pf
    python eval/contact_sheet.py <run.jsonl> --name X --top 24 --context 0.25

Writes eval/sheets/<name>__<class>.jpg, plus <name>__OTHER.jpg holding the highest-scoring
tags that fall outside the labelling schema — for a prompt-free run that sheet *is* the
generic-object bucket, and the only way to judge whether it is finding anything useful.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))

CELL = 220          # px per cell (square)
LABEL_H = 30        # px of caption strip under each cell
COLS = 6
PAD = 6
BG = (24, 26, 30)
FG = (231, 233, 238)
BOX = (255, 72, 72)


def load_manifest() -> Tuple[Dict[str, Dict], List[Dict]]:
    with open(os.path.join(HERE, "frames.json")) as handle:
        data = json.load(handle)
    return {f["id"]: f for f in data["frames"]}, data["classes"]


def build_tag_map(classes: List[Dict], aliases: Dict[str, List[str]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for spec in classes:
        mapping[spec["key"]] = spec["key"]
        for term in spec.get("prompts", []):
            mapping[term] = spec["key"]
    for key, terms in aliases.items():
        for term in terms:
            mapping[term] = key
    return mapping


def read_detections(path: str, frames: Dict[str, Dict]) -> List[Dict]:
    out: List[Dict] = []
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
            if frame_id not in frames:
                continue
            box = (data.get("frame_info") or {}).get("box")
            if not box:
                continue
            out.append({
                "frame_id": frame_id,
                "tag": data.get("tag") or info.get("prompt") or "",
                "prompt": info.get("prompt") or "",
                "score": float(info.get("score") or 0.0),
                "box": box,
            })
    return out


def crop_cell(
    image: Image.Image, box: Dict[str, float], context: float
) -> Image.Image:
    """Crop a padded view around `box` and draw the box itself inside it."""
    width, height = image.size
    x1, y1 = box["x1"] * width, box["y1"] * height
    x2, y2 = box["x2"] * width, box["y2"] * height
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)

    # Expand to a square-ish view so cells are visually comparable regardless of box aspect.
    side = max(bw, bh) * (1 + 2 * context)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    vx1, vy1 = cx - side / 2, cy - side / 2
    vx2, vy2 = cx + side / 2, cy + side / 2

    # Clamp into the frame, then pad with background so a clamped view is not stretched.
    ix1, iy1 = int(max(0, vx1)), int(max(0, vy1))
    ix2, iy2 = int(min(width, vx2)), int(min(height, vy2))
    if ix2 <= ix1 or iy2 <= iy1:
        return Image.new("RGB", (CELL, CELL), BG)

    view = image.crop((ix1, iy1, ix2, iy2)).convert("RGB")
    canvas = Image.new("RGB", (int(side), int(side)), BG)
    canvas.paste(view, (int(ix1 - vx1), int(iy1 - vy1)))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        [x1 - vx1, y1 - vy1, x2 - vx1, y2 - vy1],
        outline=BOX,
        width=max(1, int(side / 110)),
    )
    return canvas.resize((CELL, CELL), Image.LANCZOS)


def build_sheet(
    title: str,
    detections: List[Dict],
    frames: Dict[str, Dict],
    context: float,
    show_tag: bool,
) -> Optional[Image.Image]:
    if not detections:
        return None

    rows = (len(detections) + COLS - 1) // COLS
    header = 34
    sheet = Image.new(
        "RGB",
        (COLS * (CELL + PAD) + PAD, header + rows * (CELL + LABEL_H + PAD) + PAD),
        BG,
    )
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(13)
        small = ImageFont.load_default(11)
    except TypeError:  # Pillow < 10.1 has no size arg on load_default
        font = small = ImageFont.load_default()

    draw.text((PAD, 10), title, fill=FG, font=font)

    # Cache open frames: many detections share a frame, and these PNGs are up to 1920px.
    cache: Dict[str, Image.Image] = {}

    for i, det in enumerate(detections):
        col, row = i % COLS, i // COLS
        x = PAD + col * (CELL + PAD)
        y = header + row * (CELL + LABEL_H + PAD)

        frame_id = det["frame_id"]
        if frame_id not in cache:
            cache[frame_id] = Image.open(os.path.join(HERE, frames[frame_id]["frame"]))
        sheet.paste(crop_cell(cache[frame_id], det["box"], context), (x, y))

        label = f"{det['score']:.3f}"
        if show_tag and det["tag"]:
            label += f"  {det['tag'][:22]}"
        draw.text((x + 2, y + CELL + 3), label, fill=FG, font=small)
        draw.text((x + 2, y + CELL + 16), frame_id[:30], fill=(150, 156, 168), font=small)

    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", help="detector output jsonl")
    parser.add_argument("--name", required=True, help="short run name, used in filenames")
    parser.add_argument("--top", type=int, default=24, help="cells per sheet (default 24)")
    parser.add_argument("--context", type=float, default=0.2,
                        help="fraction of box size shown around it (default 0.2)")
    parser.add_argument("--aliases", default="{}",
                        help='JSON {class_key: [extra tag strings]} to extend the tag mapping')
    parser.add_argument("--out", default=os.path.join(HERE, "sheets"))
    args = parser.parse_args()

    frames, classes = load_manifest()
    tag_map = build_tag_map(classes, json.loads(args.aliases))
    detections = read_detections(args.run, frames)
    if not detections:
        print("no usable detections found (were the frames in eval/frames/?)", file=sys.stderr)
        return 1

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for det in detections:
        grouped[tag_map.get(det["tag"], "OTHER")].append(det)

    os.makedirs(args.out, exist_ok=True)
    order = [spec["key"] for spec in classes] + ["OTHER"]

    print(f"{len(detections)} detections -> {args.out}")
    for key in order:
        items = sorted(grouped.get(key, []), key=lambda d: -d["score"])[: args.top]
        if not items:
            print(f"  {key:<15} (none)")
            continue
        total = len(grouped[key])
        title = (f"{args.name}  ·  {key}  ·  top {len(items)} of {total}"
                 f"  ·  scores {items[0]['score']:.3f}-{items[-1]['score']:.3f}")
        # OTHER holds many distinct tags, so show each cell's tag; per-class sheets do not
        # need it (every cell is that class) unless the run is prompt-free.
        sheet = build_sheet(title, items, frames, args.context, show_tag=True)
        if sheet is None:
            continue
        path = os.path.join(args.out, f"{args.name}__{key}.jpg")
        sheet.save(path, quality=88)
        print(f"  {key:<15} {total:>5} detections -> {os.path.basename(path)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
