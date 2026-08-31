# Deprecated: the detector-vocabulary tag map

Retired once box-level ground truth arrived. Kept because the measurements in `mine_vocab.py`
are the evidence behind the current schema, not because anything should import this again.

## What it was for

Exactly one thing: scoring the prompt-free backends. `yoloe11-pf` and `yoloe26-pf` answer from
a fixed 4585-term vocabulary and `yolo11` from COCO-80, so against a brand/person schema they
score zero by construction — "sports ball" matches none of the seven prompts. The map existed
to translate. It was never used by the tagger: in the default and text-target modes a detector
can only answer with terms it was given, and in the general-entities mode the native label is
metadata, since a downstream query is matched against the crop embedding rather than the label
string.

## Why it was retired

It could not be made reliable. The map is built from keyword families with veto lists, and
whole-word matching still fires inside compounds, so every correction created new errors one
compound down:

- an activity veto for `easter`, `fishing`, `shopping`, `wedding`, `yoga` swallowed
  `easter bunny`, `fishing boat`, `shopping cart`, `wedding cake` and `yoga mat`, which are
  ordinary objects
- `PERSON_FORCE` containing `face` and `player` captured `face powder`, `face towel` and
  `record player`
- the same veto sent `street artist`, `theatre actor`, `wedding couple` and
  `wedding photographer` to NONOBJECT, when all four are people
- roughly sixty verbs and attributes (`eat`, `fly`, `laugh`, `sit`, `sunny`, `windy`, `white`,
  `love`, `play basketball`) sat in OTHER, inflating the breadth statistic with words no
  detector can localise

A hand review found about ninety such errors in a single pass over 4,607 terms. Each round of
patching moved the errors rather than removing them, which is what a wrong abstraction looks
like.

## What replaced it

`box_gt/score_boxes.py` computes **class-agnostic IoU coverage**: of the ground-truth boxes of
a class, how many are hit by ANY detection, whatever it was labelled. No mapping, and a closer
match to what the pipeline does with a label (nothing). It also answers the question the map
was approximating — "did a crop land on the object" — directly rather than through a proxy.

## What is still worth reading here

`mine_vocab.py` produced the finding that shaped the schema: ranking all 4,585 vocabulary terms
against averaged prototypes in SigLIP 2 text space, `brand` had a median decision margin of
0.0093 against `person`'s 0.0270 — level with the non-object noise floor. That is why `brand`
is defined as the mark itself rather than as a category of object.
