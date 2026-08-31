# 06_resolution — does input size fix the small-mark problem?

Two thirds of ground-truth brand marks are under 32 px on the short side, and coverage falls off
sharply below 24 px. Resolution is the obvious lever, and `imgsz` is already a knob in the
tagger's RuntimeConfig. This measures whether pulling it helps, per model.

## Result: it is a lever for the ultralytics models only

| model | low | mid | high |
|---|---|---|---|
| yoloe26-text | 640: 0.062 / 0.15 / 61ms | 960: 0.096 / 0.23 / 74ms | **1280: 0.133 / 0.25 / 98ms** |
| yoloe11-text | 640: 0.023 / 0.10 | 960: 0.053 / 0.18 | 1280: 0.062 / 0.21 |
| world-text | 640: 0.033 / 0.11 | 960: 0.031 / 0.09 | 1280: 0.023 / 0.09 |
| gdino | **800: 0.205 / 0.61 / 293ms** | 1100: 0.057 / 0.17 | 1400: 0.001 / 0.01 |
| owlv2 | 800: 0.274 / 0.48 / 157ms | 1100: 0.185 / 0.35 | 1400: 0.268 / 0.48 |

(brand AP / class-agnostic coverage / ms per frame, box GT at IoU 0.5, de-dup on)

**yoloe26-text more than doubles brand AP for 1.6x compute** — and the gain lands exactly where
the theory says it should. Coverage by mark size, 640 -> 1280:

| band | <16px | 16-24 | 24-32 | 32-48 | 48+ |
|---|---|---|---|---|---|
| marks | 37 | 69 | 20 | 28 | 38 |
| yoloe26 @640 | 0.05 | 0.04 | 0.10 | 0.21 | 0.39 |
| yoloe26 @1280 | 0.08 | 0.16 | 0.30 | 0.46 | 0.39 |

The mid bands move, the 48+ band is flat because it was never resolution-limited, and <16 px
stays near zero — still below the floor even at 1280. That is the signature of a genuine
small-object resolution response rather than a general accuracy shift.

World-v2 gets *worse* with resolution. Benefit is a per-model property, not a free lever.

## The transformer detectors cannot be scaled this way

Grounding DINO collapses: brand AP 0.205 -> 0.001, and person collapses with it (0.627 -> 0.021).
Person collapsing is the tell — resolution does not destroy person detection, so this was checked
as a plumbing bug before being reported.

It is not one. A direct probe (`--hf-imgsz 800` against the stock default) returns byte-identical
output, so the override is faithful; and batch size 1 against 4 gives identical detections and
identical box geometry, so it is not batch padding interacting with `target_sizes`. What actually
happens is visible in the boxes: median normalised box width goes 0.038 -> 0.487 -> 0.623 while
the detection count falls 2589 -> 365 -> 240. The model stops localising and emits a few large
diffuse boxes.

That is DETR-family behaviour. The decoder's learned reference points and object queries are tuned
to the training resolution (800 shortest edge / 1333 longest), and pushing the input past it moves
the queries off the distribution they were trained on. Grounding DINO is effectively locked to its
native resolution. OWLv2 is the same story with a softer curve — it needs
`interpolate_pos_encoding=True` to accept a non-960 input at all, and does not improve when it
does.

## What this changes

Brand candidates, each at its own best resolution:

| | brand AP | coverage | ms/frame batched | ms/frame batch-1 |
|---|---|---|---|---|
| owlv2 @960 native | 0.308 | 0.54 | 193 | 203 |
| gdino @800 native | 0.205 | 0.61 | 293 | 297 |
| yoloe26-text @1280 | 0.133 | 0.25 | 98 | 158 |

The gap between yoloe26 and gdino narrows from 3.3x to 1.5x on AP, for roughly a third of the
cost. Coverage is still 2.4x apart, so gdino continues to find far more distinct marks — but
yoloe26 at 1280 is a materially better fast path than yoloe26 at 640 was.

Person wants the opposite: yolo11 is best at 640 (AP 0.752) and degrades to 0.702 at 1280. The
resolution knob has to be per-detector, not global.

## Reproducing

```bash
bash eval/experiments/06_resolution/run_sweep.sh   # ultralytics rungs
bash eval/experiments/06_resolution/run_hf.sh      # HF rungs, batch scales down with size
```
