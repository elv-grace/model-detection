# 07_gdino_tiny — is the smaller Swin backbone a better `coverage` backend?

`coverage` mode exists because Grounding DINO finds 2.4x as many distinct marks as the fast
path, and it costs ~2x end to end for that. The obvious lever is the smaller checkpoint:
`grounding-dino-tiny` is Swin-T where base is Swin-B.

(This is a *smaller* model, not a distilled one. Distillation moves capability into a smaller
student; it does not make a teacher more accurate. Distilling Grounding DINO into a
brand-specific student is a training project, not a config change, and is out of scope here.)

## Result: no. Keep base.

| | brand AP | brand cov | brand IoU | person AP | person cov | ms/frame |
|---|---|---|---|---|---|---|
| grounding-dino-base | **0.205** | **0.61** | 0.82 | 0.628 | 0.95 | 293 |
| grounding-dino-tiny | 0.162 | 0.56 | 0.81 | **0.783** | **0.98** | 232 |
| yoloe26-text @1280 | 0.133 | 0.25 | 0.79 | 0.653 | 0.92 | 98 |
| yolo11 @640 | 0.000 | 0.04 | — | 0.752 | 0.95 | 53 |

Tiny keeps **92% of base's coverage** — the metric `coverage` mode exists for — and 79% of its
AP. But it is only **21% faster**, and that is the reason to decline it: the saving is roughly
proportional to the loss, so there is no free lunch on this axis.

21% is smaller than a Swin-T-for-Swin-B swap suggests, and the explanation is where the time
actually goes. The BERT text encoder and the DETR decoder are the same size in both checkpoints,
so shrinking the vision backbone only shrinks part of the model. Anyone reaching for a cheaper
transformer detector should expect this ceiling.

Held out, tiny is also slightly less stable than base on brand (drop 0.031 against 0.023,
threshold range 0.130-0.192 against 0.142-0.162).

## The one genuinely interesting number

**gdino-tiny beats YOLO11 on person: AP 0.783 against 0.752, coverage 0.98 against 0.95.** It is
the best person detector measured.

It still should not serve person, because it costs 232 ms against YOLO11's 53 ms — **4.4x the
compute for 4% more AP**. But it is worth recording that the person choice is a cost decision
rather than a quality one, in case a future deployment is latency-insensitive.

## Where tiny would make sense

Nowhere in the current design. It sits between the two shipped options without dominating
either: for throughput yoloe26 is 2.4x faster still, and for coverage base is 9% better. A middle
rung only helps if something needs exactly that trade, and nothing here does.

## Reproducing

```bash
bash eval/experiments/07_gdino_tiny/run.sh
python3 eval/box_gt/score_boxes.py --runs 07_gdino_tiny
python3 eval/box_gt/operating_point.py --runs 07_gdino_tiny --backends gdino-tiny
```
