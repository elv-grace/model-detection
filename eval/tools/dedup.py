#!/usr/bin/env python3
"""Class-agnostic non-maximum suppression across prompt terms.

Why this is needed at all
-------------------------
The brand schema asks for six near-synonymous terms -- logo, letter logo, car logo, emblem,
brand, label -- because no single one of them finds every mark. The cost is that a mark which
answers to several of them is returned several times. On the ground-truth frames Grounding DINO
emits 768 pairs of brand boxes overlapping at IoU 0.5 or more from only 1,245 detections, 399 of
them between *different* prompt terms.

That matters in two places, and they want the same fix:

    scoring     the box scorer matches each ground-truth box once, so the second detection of a
                mark is a false positive. Part of the low brand precision (0.25 for gdino) is
                this rather than genuine mistakes.

    production  the pipeline embeds every crop. Six crops of one wordmark cost six SigLIP
                forward passes and put six near-identical vectors in the index, which is wasted
                compute at write time and a cluster of duplicates at query time.

Why class-agnostic
------------------
Suppression has to run ACROSS terms, not within each one. Standard per-class NMS would keep the
`logo` box and the `letter logo` box of the same wordmark, since they belong to different
classes -- and those cross-term pairs are the majority of the problem (399 of 768 for gdino).
The label is metadata here; the crop is the product. Two boxes on the same pixels are one
finding whatever they were called.

Suppression is confined to a single class bucket, though. A `person` box overlapping a `logo`
box is two findings -- the mark on the jersey and the player wearing it -- and both are wanted.

Keeping the right survivor
--------------------------
The kept detection is the highest-scoring of the group, and it carries the suppressed terms with
it in `also_matched` rather than discarding them. That preserves the information for free: a
crop that answered to `logo` and `letter logo` and `brand` is more confidently a mark than one
that answered to `brand` alone, and downstream may want that.

    from dedup import dedup_frame
    kept = dedup_frame(detections, iou_threshold=0.6)

Choosing the threshold
----------------------
0.6, chosen against box ground truth by a constraint rather than by maximising a score. The test
is that class-agnostic coverage must not fall: if suppression reduces the number of ground-truth
marks hit by any detection, it is merging marks that are really distinct -- adjacent logos on a
hoarding, a wordmark inside an emblem -- and that is a true loss the AP does not show. Coverage
survives intact at 0.6 and starts falling below about 0.5. Maximising AP alone would have picked
0.45, which eats real marks to buy a better-looking number.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schema_brand_person import class_of_prompt  # noqa: E402


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


def dedup_frame(detections: List[Dict], iou_threshold: float = 0.6,
                bucket=class_of_prompt) -> List[Dict]:
    """Suppress overlapping detections within each class bucket, across prompt terms.

    `detections` are dicts carrying at least `box`, `tag` and `score`. Returns the survivors in
    descending score order, each with an `also_matched` list of the terms it absorbed.
    """
    groups: Dict[str, List[Dict]] = {}
    for det in detections:
        groups.setdefault(bucket(det["tag"]), []).append(det)

    kept: List[Dict] = []
    for rows in groups.values():
        rows = sorted(rows, key=lambda r: -r["score"])
        survivors: List[Dict] = []
        for det in rows:
            absorbed = None
            for other in survivors:
                if iou(det["box"], other["box"]) >= iou_threshold:
                    absorbed = other
                    break
            if absorbed is None:
                survivors.append({**det, "also_matched": []})
            elif det["tag"] != absorbed["tag"] and det["tag"] not in absorbed["also_matched"]:
                absorbed["also_matched"].append(det["tag"])
        kept.extend(survivors)
    return sorted(kept, key=lambda r: -r["score"])
