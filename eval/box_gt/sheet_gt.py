#!/usr/bin/env python3
"""Render the box ground truth back onto its frames, for reviewing the labelling itself.

The ground truth is the instrument every other number is measured against, so it is the one
artefact with nothing downstream to catch its mistakes. A mislabelled frame does not announce
itself in the scores -- it just quietly moves every detector's AP in the same direction. This
draws the boxes back onto the frames so they can be checked by eye.

Two views:

    --mode frames   one panel per frame, all boxes drawn in place. Answers "is anything
                    missed, is anything boxed that should not be".
    --mode crops    one panel per box, cropped to the box with context around it. Answers
                    "is this box actually tight on the thing it claims to be" -- which the
                    frame view hides for small marks, and small marks are most of `brand`.

    python eval/box_gt/sheet_gt.py --mode crops --cls brand
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))

import paths  # noqa: E402

COLOURS = {"brand": (255, 92, 60), "person": (60, 190, 255), "ignore": (150, 150, 150)}


def load_font(size: int):
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def frame_paths():
    with open(paths.FRAMES_JSON) as handle:
        meta = json.load(handle)
    return {f["id"]: os.path.join(paths.FRAMESET, f["frame"]) for f in meta["frames"]}


def draw_frames(labels, sources, out, cell, cols):
    font = load_font(max(11, cell // 26))
    panels = []
    for frame_id in sorted(labels):
        entry = labels[frame_id]
        image = Image.open(sources[frame_id]).convert("RGB")
        W, H = image.size
        image.thumbnail((cell, cell), Image.LANCZOS)
        w, h = image.size
        draw = ImageDraw.Draw(image)
        counts = {"brand": 0, "person": 0, "ignore": 0}
        for box in entry["boxes"]:
            counts[box["cls"]] = counts.get(box["cls"], 0) + 1
            xy = [box["x1"] * w, box["y1"] * h, box["x2"] * w, box["y2"] * h]
            colour = COLOURS.get(box["cls"], (255, 255, 255))
            # Ignore regions are hatched by a dashed-looking double outline so they read as
            # "excluded" rather than as another positive.
            draw.rectangle(xy, outline=colour, width=1 if box["cls"] == "ignore" else 2)
            if box["cls"] == "ignore":
                draw.rectangle([xy[0] + 3, xy[1] + 3, xy[2] - 3, xy[3] - 3],
                               outline=colour, width=1)
        caption = (f"{frame_id}  {W}x{H}  "
                   f"b{counts['brand']} p{counts['person']} i{counts['ignore']}")
        bar = Image.new("RGB", (w, 20), (18, 18, 20))
        ImageDraw.Draw(bar).text((4, 4), caption, font=font, fill=(235, 235, 235))
        panel = Image.new("RGB", (w, h + 20), (18, 18, 20))
        panel.paste(image, (0, 0))
        panel.paste(bar, (0, h))
        panels.append(panel)
    grid(panels, out, cols)


def draw_crops(labels, sources, out, cell, cols, wanted, context):
    """One panel per box: the box drawn inside a wider window, so tightness is visible."""
    font = load_font(11)
    panels = []
    for frame_id in sorted(labels):
        image = None
        for box in labels[frame_id]["boxes"]:
            if box["cls"] != wanted:
                continue
            if image is None:
                image = Image.open(sources[frame_id]).convert("RGB")
            W, H = image.size
            x1, y1, x2, y2 = box["x1"] * W, box["y1"] * H, box["x2"] * W, box["y2"] * H
            pw, ph = (x2 - x1) * context, (y2 - y1) * context
            win = (max(0, x1 - pw), max(0, y1 - ph), min(W, x2 + pw), min(H, y2 + ph))
            crop = image.crop([int(v) for v in win])
            if crop.width < 4 or crop.height < 4:
                continue
            scale = cell / max(crop.width, crop.height)
            crop = crop.resize((max(1, int(crop.width * scale)),
                               max(1, int(crop.height * scale))), Image.LANCZOS)
            draw = ImageDraw.Draw(crop)
            draw.rectangle([(x1 - win[0]) * scale, (y1 - win[1]) * scale,
                            (x2 - win[0]) * scale, (y2 - win[1]) * scale],
                           outline=COLOURS[wanted], width=2)
            caption = f"{frame_id}  {int(x2 - x1)}x{int(y2 - y1)}px"
            bar = Image.new("RGB", (crop.width, 16), (18, 18, 20))
            ImageDraw.Draw(bar).text((3, 2), caption, font=font, fill=(210, 210, 210))
            panel = Image.new("RGB", (crop.width, crop.height + 16), (18, 18, 20))
            panel.paste(crop, (0, 0))
            panel.paste(bar, (0, crop.height))
            panels.append(panel)
    grid(panels, out, cols)


def grid(panels, out, cols):
    if not panels:
        print("nothing to draw", file=sys.stderr)
        return
    rows = (len(panels) + cols - 1) // cols
    cw = max(p.width for p in panels) + 6
    ch = max(p.height for p in panels) + 6
    sheet = Image.new("RGB", (cols * cw, rows * ch), (10, 10, 12))
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % cols) * cw + 3, (i // cols) * ch + 3))
    sheet.save(out, quality=88)
    print(f"{out}  {len(panels)} panels  {sheet.size[0]}x{sheet.size[1]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default=paths.BOX_LABELS)
    parser.add_argument("--mode", choices=["frames", "crops"], default="frames")
    parser.add_argument("--cls", choices=["brand", "person"], default="brand")
    parser.add_argument("--cell", type=int, default=420)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--context", type=float, default=0.6)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.labels) as handle:
        labels = {k: v for k, v in json.load(handle)["frames"].items() if v.get("done")}
    sources = frame_paths()
    out = args.out or os.path.join(
        HERE, f"gt_{args.mode}{'_' + args.cls if args.mode == 'crops' else ''}.jpg")

    if args.mode == "frames":
        draw_frames(labels, sources, out, args.cell, args.cols)
    else:
        draw_crops(labels, sources, out, args.cell, args.cols, args.cls, args.context)
    return 0


if __name__ == "__main__":
    sys.exit(main())
