#!/usr/bin/env python3
"""Contact sheets for the brand/person runs.

Why these are worth more than usual here
----------------------------------------
The frame-presence metric saturates on this schema: brand is present in 62 frames of 100 and
person in 97, so precision is near-ceiling for anything that fires and the table cannot rank
the detectors on its own. The sheets are unaffected by that, because they do not consult the
labels at all -- they show what was actually boxed.

Box ground truth (box_gt/) now measures localisation properly, and it is the headline. The
sheets remain worth reading beside it for the thing no scalar reports: WHAT was boxed. Box AP
says owlv2 leads on brand; the sheets say whether it is finding wordmarks or crops of jersey
fabric, and only one of those retrieves against a logo pool.

Each cell draws the detection box inside a slightly wider crop, so both *what* was found and
*how well it was framed* are visible. Cells run in score order, so reading left to right is
reading precision@K.

Near-duplicate suppression
--------------------------
Necessary here in a way it was not for the 8-class sweep. These runs are ungated, so Grounding
DINO alone emits over a hundred detections per frame and a naive top-24 would be two dozen
near-identical boxes on one jersey. Cells are therefore suppressed against those
already chosen in the same frame above an IoU threshold, so a sheet shows 24 distinct findings
rather than one finding 24 times.

Usage
-----
    python eval/tools/sheet_bp.py --runs 04_brand_person_mark --backends gdino owlv2
    python eval/tools/sheet_bp.py --runs 02_brand_person_101/runs_visual --classes brand person
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paths  # noqa: E402
from schema_brand_person import class_of_prompt  # noqa: E402

CELL = 200
LABEL_H = 28
COLS = 6
PAD = 5
BG = (16, 19, 22)
PANEL = (23, 27, 32)
FG = (231, 234, 238)
DIM = (140, 150, 160)
BOX_COLOUR = {"brand": (70, 200, 175), "person": (240, 170, 60), "OTHER": (150, 160, 172)}


def font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/opt/conda/fonts/DejaVuSans.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def iou(a: Dict, b: Dict) -> float:
    x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def read(path: str, frames: Dict[str, Dict]) -> Dict[str, List[Dict]]:
    by_class: Dict[str, List[Dict]] = {"brand": [], "person": [], "OTHER": []}
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
            if frame_id not in frames:
                continue
            tag = data["tag"]
            # Bucketed by the prompt list alone. Anything a prompt-free backend emits from its
            # own vocabulary lands on the OTHER sheet, which is the honest place for it: this
            # tool shows what was boxed, and guessing whether "sports ball" ought to count as
            # brand is exactly the mapping that was retired.
            bucket = class_of_prompt(tag)
            by_class.setdefault(bucket, []).append({
                "frame": frame_id, "tag": tag,
                "score": float(data["additional_info"].get("score", 1.0)),
                "box": data["frame_info"]["box"],
            })
    for rows in by_class.values():
        rows.sort(key=lambda r: -r["score"])
    return by_class


def pick(rows: List[Dict], top: int, nms: float) -> List[Dict]:
    kept: List[Dict] = []
    per_frame: Dict[str, List[Dict]] = {}
    for row in rows:
        seen = per_frame.setdefault(row["frame"], [])
        if any(iou(row["box"], other) > nms for other in seen):
            continue
        seen.append(row["box"])
        kept.append(row)
        if len(kept) >= top:
            break
    return kept


def render(rows: List[Dict], frames: Dict[str, Dict], title: str, bucket: str,
           context: float, out_path: str) -> None:
    cols = min(COLS, max(1, len(rows)))
    rowc = (len(rows) + cols - 1) // cols
    head = 34
    width = cols * (CELL + PAD) + PAD
    height = head + rowc * (CELL + LABEL_H + PAD) + PAD
    sheet = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(sheet)
    draw.text((PAD + 2, 9), title, fill=FG, font=font(15))

    colour = BOX_COLOUR.get(bucket, BOX_COLOUR["OTHER"])
    for index, row in enumerate(rows):
        cx, cy = index % cols, index // cols
        ox = PAD + cx * (CELL + PAD)
        oy = head + cy * (CELL + LABEL_H + PAD)
        draw.rectangle([ox, oy, ox + CELL, oy + CELL + LABEL_H], fill=PANEL)
        try:
            image = Image.open(os.path.join(paths.FRAMESET, frames[row["frame"]]["frame"])).convert("RGB")
        except Exception:
            continue
        iw, ih = image.size
        box = row["box"]
        x1, y1 = box["x1"] * iw, box["y1"] * ih
        x2, y2 = box["x2"] * iw, box["y2"] * ih
        bw, bh = max(2.0, x2 - x1), max(2.0, y2 - y1)
        # Widen to a square window around the box so the crop shows its surroundings; that is
        # what makes a mislocated box visible rather than merely a tight one.
        side = max(bw, bh) * (1.0 + 2 * context)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        left, top_ = mx - side / 2, my - side / 2
        left = min(max(0.0, left), max(0.0, iw - side))
        top_ = min(max(0.0, top_), max(0.0, ih - side))
        right, bottom = min(iw, left + side), min(ih, top_ + side)
        crop = image.crop((int(left), int(top_), int(right), int(bottom)))
        if crop.width < 2 or crop.height < 2:
            continue
        scale = CELL / max(crop.width, crop.height)
        crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
        px = ox + (CELL - crop.width) // 2
        py = oy + (CELL - crop.height) // 2
        sheet.paste(crop, (px, py))
        draw.rectangle(
            [px + (x1 - left) * scale, py + (y1 - top_) * scale,
             px + (x2 - left) * scale, py + (y2 - top_) * scale],
            outline=colour, width=2)
        tag = row["tag"][:22]
        draw.text((ox + 4, oy + CELL + 3), tag, fill=FG, font=font(11))
        draw.text((ox + 4, oy + CELL + 15), f"{row['score']:.3f}  {row['frame'][:16]}",
                  fill=DIM, font=font(9))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sheet.save(out_path, quality=76, optimize=True)
    print(f"  {out_path}  ({len(rows)} cells, {os.path.getsize(out_path)//1024} KB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="04_brand_person_mark")
    parser.add_argument("--backends", nargs="*", default=None)
    parser.add_argument("--classes", nargs="*", default=["brand", "person", "OTHER"])
    parser.add_argument("--top", type=int, default=24)
    parser.add_argument("--nms", type=float, default=0.35,
                        help="suppress a cell against ones already chosen in the same frame")
    parser.add_argument("--context", type=float, default=0.35)
    parser.add_argument("--out", default=None,
                        help="default: <experiment>/sheets")
    args = parser.parse_args()
    out_dir = args.out or os.path.join(paths.experiment(args.runs), "sheets")

    with open(paths.FRAMES_JSON) as handle:
        frames = {f["id"]: f for f in json.load(handle)["frames"]}

    # NOT `paths` -- that is the module imported above, and rebinding it here shadowed it
    # for the whole function.
    run_paths = sorted(glob.glob(paths.runs_glob(args.runs)))
    if args.backends:
        wanted = set(args.backends)
        run_paths = [p for p in run_paths if os.path.basename(os.path.dirname(p)) in wanted]
    if not run_paths:
        print("no matching runs", file=sys.stderr)
        return 1

    for path in run_paths:
        name = os.path.basename(os.path.dirname(path))
        by_class = read(path, frames)
        for bucket in args.classes:
            rows = pick(by_class.get(bucket, []), args.top, args.nms)
            if not rows:
                continue
            render(rows, frames, f"{name}  ·  {bucket}  ·  top {len(rows)} by score",
                   bucket, args.context, os.path.join(out_dir, f"{name}__{bucket}.jpg"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
