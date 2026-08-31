# eval/

Detector evaluation for the brand / person tagger.

```
frameset/      the 100 frozen frames, the manifest, the presence labels
tools/         every script, plus the vocabulary dump
tools/deprecated/  the retired tag map, kept for its measurements
experiments/   NN_name/ per experiment, holding only that run's outputs
box_gt/        box-level ground truth: task builder, labelling tool, scorer
report/        report source, build script, built page
```

Code and data are separated rather than grouped strictly by experiment, because the scripts are
shared and the runs are not — putting `run_bp.py` inside the experiment that first used it would
force experiments 2, 3 and 4 to copy it or reach sideways into a sibling.

## The pipeline being evaluated

```
source media -> detector (default prompts, or a caller-supplied text target)
             -> crops -> SigLIP 2 embeddings -> vector index
```

Identification is downstream. "Which brand is this" and "is this the same person" are answered
by cosine similarity against a pool, the way model-logo and model-celeb already work. The
detector owes only crops that are tight around the entity itself.

Three modes: no target (default prompts), a text target (used verbatim), or general (no prompts
at all — a prompt-free backend catalogues whatever is there and its native labels pass through
as metadata). **Image prompts are deliberately unsupported**; see `tools/schema_brand_person.py`
for the 42 runs behind that decision.

## The schema

Seven prompts, in `tools/schema_brand_person.py`:

```
brand   logo, letter logo, car logo, emblem, brand, label   (the MARK, never the object)
person  person
```

`brand` is the mark itself — the GAP wordmark, not the hoodie. Asked for `sportswear` a detector
returns the garment; asked for `logo` it returns the wordmark, and the wordmark is the crop that
retrieves against a logo pool.

## Experiments, in the order they were run

Each exists because the previous one produced a result that changed the question.

| dir | what it tested | what it found |
|---|---|---|
| `01_general_8class` | 8 generic classes, 29 prompts | prompts mined from the model's own vocabulary took logo F1 from ~0 to 0.80 |
| `02_brand_person_101` | first reframing: 5 tiers, 101 prompts, plus 42 image-prompt runs | the metric saturated; image prompting is degenerate for brand and redundant for person |
| `03_prompt_ablation` | bare words, 6 mark terms, 6 marks + 5 surfaces | object nouns overshadow marks (1% vs 100% mark-like); surfaces overshadow them one level up |
| `04_brand_person_mark` | **current** — 7 prompts, brand = the mark | owlv2 and gdino are the only backends that find marks (box AP 0.310 / 0.166 vs 0.062 next) |
| `05_symbol_ablation` | `symbol` as a 7th brand term, and alone | rejected: costs the leader (0.310 → 0.284), gains nothing |
| `06_resolution` | input size 640/960/1280 and 800/1100/1400 | a lever for the ultralytics models only (yoloe26 brand AP 0.062 → 0.133); the DETR-family detectors collapse off their native resolution |
| `07_gdino_tiny` | Swin-T instead of Swin-B for the `coverage` backend | declined: 92% of the coverage but only 21% faster — the shared BERT encoder and DETR decoder cap the saving |
| `08_embedders` | **Phase B** — siglip2-base-naflex (768-d) vs siglip2-large-384 (1024-d) | quality is a tie (r@1 0.926 vs 0.929); base-naflex is 6.1x cheaper per crop, so it wins on cost |
| `09_min_crop` | retrieval vs crop pixel size, against the mark-size distribution | `min_crop_pixels` 32 → **16**: the trade turns over there, and 32 was discarding two thirds of all marks to buy precision already in reach |

## Running things

```bash
# from repo root, inside the container
P="podman run --rm --entrypoint /opt/conda/envs/mlpod/bin/python \
   --volume=$(pwd)/.cache:/root/.cache --volume=$(pwd)/eval:/elv/eval \
   --device nvidia.com/gpu=0 --network host -e HF_HOME=/root/.cache \
   -e ELV_WEIGHTS_DIR=/root/.cache/detection general_detection"

$P /elv/eval/tools/run_bp.py                      # all 8 backends, default prompts
$P /elv/eval/tools/run_bp.py --prompts logo person --suffix __x --out /elv/eval/experiments/...
$P /elv/eval/tools/sheet_bp.py                    # contact sheets

python3 eval/tools/score_bp.py                    # cost + breadth, no labels or GPU needed
python3 eval/box_gt/score_boxes.py                # THE ranking: box AP, coverage, mean IoU
python3 eval/box_gt/operating_point.py            # held-out threshold, leave-one-clip-out
python3 eval/box_gt/sheet_gt.py --mode crops --cls brand   # review the ground truth itself
```

## Box-level ground truth  —  the headline metric

Frame presence is close to exhausted here: person appears in 97 frames of 100, so it has three
negatives and cannot be scored. Boxes are the instrument that replaces it, and they also measure
the localisation quality the crop-and-embed pipeline depends on.

25 frames from 11 clips are labelled: **192 brand marks, 254 people, 15 ignore regions**. Of the
524 seeded proposals, 225 were rejected and 158 boxes drawn fresh -- for brand, 72% of the final
boxes were drawn by hand, so the ground truth is not a consensus of the models it grades.

Where box ground truth exists it also supersedes frame presence, and `score_bp.py` folds it in:
it answers the same question exactly, and it corrected four frames (brand positives 58 -> 62,
all in the direction of small marks the 8-class pass had missed).

```bash
$P /elv/eval/box_gt/make_box_task.py --frames 25 --min-score 0.10   # builds label_boxes.html
# label in a browser, export box_labels.json into box_gt/
python3 eval/box_gt/score_boxes.py
```

Two post-processes are on by default in scoring because they are part of the pipeline, not of
the measurement: **cross-prompt de-duplication** at IoU 0.6 (`tools/dedup.py` -- six brand terms
return one mark several times; suppression lifts gdino brand AP 0.166 -> 0.205 and cuts crops by
a third), and the class-agnostic coverage constraint that chose that 0.6. Fitting AP would have
picked 0.45, which silently merges genuinely distinct marks.

`operating_point.py` answers the separate question of what a *config file* delivers: thresholds
picked on ten clips and evaluated on the eleventh. The split is by clip, not frame, because
frames from one clip share the same physical logos. It found that the YOLOE text heads do not
generalise their threshold -- yoloe26-text person F1 0.656 in-sample, 0.483 held out, with
chosen thresholds spanning a four-fold range against yolo11's 0.101-0.119.

`box_gt/label_boxes.html` is a single self-contained file with all 25 frames embedded — copy it
anywhere, no server or assets needed. Proposals are pre-drawn from a balanced union of gdino,
owlv2, yoloe11-text and yolo11, so labelling is accept-and-nudge rather than draw-from-scratch.
Label every instance before marking a frame done; only completed frames are scored.

The dominant finding is that brand is a **small-object problem**: ground-truth marks have a
median short side of 22 px and two thirds are under 32 px, and every detector's coverage falls
off sharply below 24 px. That sets the real constraint on `min_crop_pixels` in Phase B.

`score_boxes.py` reports class-aware AP/P/R **and** class-agnostic coverage — of the ground-truth
boxes of a class, how many were hit by any detection whatever it was labelled. Coverage is the
fair number for a prompt-free backend that boxes the right object under a different name, and it
is what allowed the tag map to be retired.

## Provisional labels

`frameset/presence_labels.json` was labelled under the old 8-class schema. `brand` is derived
from it as `logo` alone, then corrected on the 25 frames that have box ground truth, giving 62
positives of 100. The other 75 frames still carry the derived label, so the two halves are not
labelled to the same standard -- which is why box AP is the headline and presence is kept mainly
as the vehicle for the speed measurements.
