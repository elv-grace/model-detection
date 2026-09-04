# model-detection — general entity detection vector tagger

Open-vocabulary "crop and embed" tagger. YOLO detects persons and YOLOE/Grounding-DINO detects non-person entities in each sampled frame;
SigLIP 2 embeds each crop; and every detection becomes one vector `Tag` carrying its normalized bounding box.

Implements `common_ml`'s `FrameModel`, so frame extraction (ffmpeg/PyAV), image-vs-video
dispatch, and tag serialization are handled by the tagger runtime.  
Model defined in `general_detection/`. `eval/` contains experiment results for testing different detection and embedding models.

- [Original YOLOE paper](https://arxiv.org/html/2503.07465v1)
- [Ultralytics YOLOE docs](https://docs.ultralytics.com/models/yoloe)
- [Comparing models for detection](https://www.robolabs.ai/resources/blog/open-vocabulary-foundation-models-vision-language) + Google Gemini search
    - **YOLOE-26** (open vocabulary, text + image prompting, prompt-free option)
    - YOLOE-11 (open vocabulary, text + image prompting, prompt-free option)
    - YOLO-World-v2 (open vocabulary, text prompting)
    - **YOLO11** (closed vocabulary)
    - **Grounding DINO** (open vocabulary, text prompting)
    - OWLv2 (open vocabulary, text prompting)
- SIGlip 2 models: **SIGlip2-base-naflex** (768-d) vs. SIGlip2-large-384 (1024-d)

## Pipeline

```
frame (H,W,3 uint8 RGB)
  ├─ person targets  ─→ YOLO11        (closed COCO-80, @640)
  └─ everything else ─→ YOLOE-26 @1280   ("fast", default)
                     or Grounding DINO @800 ("coverage")
       └─ dedupe (3 stages, below)
            └─ gate: class_conf, min_box_size, min_crop_pixels
                 └─ crop (+ crop_padding context) and cap at max_detections
                      └─ SigLIP 2 NaFlex vision tower, ONE batch across both detectors
                           └─ L2-normalize → FrameTag(tag, vector, box, additional_info)
                                          (+ FrameTag(tag, box) if output_tags)
```

The two detectors run sequentially and the order does not matter: on one GPU they compete for
the same SMs, so running them concurrently costs the same wall-clock plus scheduling overhead.
A detector with nothing routed to it is never loaded, so a person-only request never pays for
the brand model.

### Why two detectors

Measured against box ground truth, the classes want different models and nothing forces one
model to serve both:

| | brand AP | brand coverage | person AP | person mean IoU | ms/frame |
|---|---|---|---|---|---|
| YOLOE-26 @1280 | 0.133 | 0.25 | 0.653 | 0.86 | 98 |
| Grounding DINO @800 | 0.205 | 0.61 | 0.628 | 0.85 | 293 |
| **YOLO11 @640** | 0.000 | 0.04 | **0.752** | **0.89** | **53** |

YOLO11 beats every open-vocabulary model at the one class COCO was built around, while being
the cheapest model measured. It also *cannot* do brand — COCO-80 has no logo class, and its
0.04 coverage says it does not box marks incidentally either.

### Dedupe

Each parent term is expanded into several phrasings, and YOLOE assigns a distinct class id
per phrasing. That creates duplicates by construction, so three stages run in order:

1. **Ultralytics' internal NMS** (`iou`, default 0.7) — runs inside `predict`, **per class
   id**, i.e. per phrasing. Two "logo" boxes on the same sign collapse.
2. **Synonym-group NMS** (`nms_iou`, default 0.6) — ours, and the highest-value post-process
   measured. Stage 1 cannot do this: "logo" and "letter logo" are different class ids, so
   ultralytics never compares their boxes, and one jersey badge survives as one box per
   phrasing — N near-identical crops, N near-identical vectors, N stacked overlay rectangles.
   On the ground-truth frames Grounding DINO emitted **768 overlapping brand-box pairs from
   1,245 detections**, 399 of them between *different* terms. Suppressing them lifts its brand
   AP 0.166 → 0.205 with coverage unchanged, and cuts crops reaching the embedder by ~⅓.

   The 0.6 is set by a **constraint, not by maximising a score**: coverage must not fall. If
   suppression reduces the ground-truth marks hit by any detection, it is merging marks that
   are genuinely distinct — adjacent logos on one hoarding, a wordmark inside an emblem — and
   AP cannot see that, because both boxes were being counted anyway. Coverage is intact at 0.6
   and starts falling below ~0.5. Maximising AP alone would have chosen 0.45.
3. **Cross-parent NMS** (`cross_class_nms_iou`, default 1.0 = **disabled**, and it should stay
   disabled) — the only cross-parent overlap available under this schema is brand against
   person, and a mark on a jersey overlapping the player wearing it is **two real findings**,
   not a duplicate. Enabling this would delete one of them.

## Output

Each detection becomes a `Tag` with:

- `tag` — the **parent term** (`logo`, `text`, `product`, `screen`, `board`, `person`, `object`)
- `vector` — L2-normalized SigLIP 2 crop embedding (cosine == dot product)
- `box` — normalized `{x1, y1, x2, y2}`, the **un-padded** detection box
- `additional_info` — provenance, see below

`common_ml` then attaches `start_time`/`end_time` (ms), `source_media`, and `frame_info.{frame_idx, box}`. Tags stay per-frame with their boxes intact.

The box is carried **twice**: in `box` and again in `additional_info.box`. `box` becomes
`frame_info.box`, which is what EVIE draws — but the vectorstore's embedding schema has no box
column, so a search row comes back with `additional_info` and no geometry. Stamping it there
means the box travels with the vector it belongs to, rather than having to be joined back from
a tag track a vector-only run never wrote.

### `output_tags` — a plain tag beside each vector tag

`output_tags` parameter (default `false`) will emit a **second** `Tag` right after each detection's vector tag if `true`: same label, same box, **no `vector`**, and no embedder provenance 
(`embedder`, `dim`, `max_num_patches` describe a vector this tag does not carry). Everything else in
`additional_info` is identical.

It is a visualization aid. EVIE draws the box from `frame_info` either way, but a vector-less
tag also lands as a tag track.

The vector-less tag is also **run-length merged**, which the vector tags are not:
`AVModel._combine_adjacent` skips any tag that has a vector, so on video these twins are
additionally emitted as merged spans per label (with `additional_info` and `frame_info` dropped
by the merge) on top of the per-frame copies. That merged span is what reads as a track rather
than a dotted line of one-frame tags — but it means the tag count grows by more than 2×.

```bash
--params '{"output_tags": true}'
```

### `additional_info`

| field | why it is there |
|---|---|
| `kind` | `crop`, or `frame` for the optional whole-frame vector |
| `box` | the same normalized box as `Tag.frame_info.box`, repeated here because this is the only field a vectorstore search row returns |
| `prompt` | which phrasing actually fired — makes per-phrasing recall measurable |
| `score` | detector confidence |
| `detector` | which backend found it (`yoloe-26l-seg`, `grounding-dino-base`, `yolo11l`). **Per tag, not per run** — two detectors are in play |
| `embedder`, `dim` | so the index can validate what it is storing. Absent on `output_tags` twins, which carry no vector |
| `max_num_patches` | resolution budget the vector was produced at. Absent on `output_tags` twins |
| `crop_padding` | **changes the vector** — measurably. Mixing paddings in one index costs retrievals silently. See [below](#crop_padding-changes-the-vector-measured) |
| `upscale` | the NaFlex scale actually applied to this crop, so the heavily-interpolated tail can be filtered downstream without re-tagging |

## Seeing the boxes in the video editor

Set [`output_tags`](#output_tags--a-plain-tag-beside-each-vector-tag) to true to get a vector-less
copy of each detection, which shows up in the editor's tag tracks alongside the overlaid bounding boxes.

## Prompting modes — text only, and why

Three modes were evaluated. **Text prompting is the only one that survived**, which simplifies
the design rather than constraining it: the target parameter is text-only, image similarity
becomes a query-side operation on the vector index, and the detector choice is freed from
YOLOE's AGPL-3.0 licence.

| mode | verdict |
|---|---|
| **Text** | **implemented.** The target parameter names terms; `Tag.tag` is the parent term, so the label space stays yours and filterable. |
| **Visual** (image exemplars) | **rejected.** 42 single-exemplar runs. |
| **Prompt-free** (4585-term vocabulary) | **rejected for brand.** |

### Visual prompts are degenerate for brand and redundant for person

Prompted with a Nike or UPS logo crop, YOLOE returns **near-full-frame boxes**. It scores 0.44
frame-presence recall, which looks like a result until the contact sheet shows there is no
localisation happening at all — a box covering the frame contains the logo trivially. The metric
counts it; the crop is worthless to an embedder.

Person exemplars *do* give tight, correct boxes — and give nothing the word `person` did not
already give. Identification (*which* person, which pose) is a downstream query against the
embeddings either way.

### Prompt-free cannot find marks

The prompt-free checkpoints catalogue whatever is present from a 4585-term vocabulary that does
contain `logo`, `car logo`, `letter logo`, `person`. Measured against box ground truth they
reach brand AP **0.019** (yoloe11-pf) and **0.010** (yoloe26-pf) against 0.133–0.205 for the
text-prompted backends, with class-agnostic coverage of 0.11 and 0.07. Having the word in the
vocabulary is not the same as finding the thing.

### Naming the target: marks, never the object carrying them

The single largest effect measured. With 101 concrete object nouns only **1%** of brand
detections carried a mark-like label — 63% were `shoe`. With the six mark terms, **100%** do.

It is not only the label that changes, it is the crop: asked for `sportswear` you get the hoodie,
asked for `logo` you get the wordmark on it, and for retrieval against a logo pool the wordmark
is the crop you need. Mark-*carrying* surfaces (`sign`, `banner`, `billboard`) reproduce the same
failure one level up and were rejected too — a banner is a surface a logo sits on, so the box
lands on the banner. `symbol` was tested as a seventh brand term and rejected: it costs the
leading model AP and wins the argmax on boxes `logo` already had.

Vectors stay comparable regardless of mode: the prompt never enters the embedding, it only
decides which regions get embedded.

## Runtime parameters

Injected per request as a JSON `--params` object; see `general_detection/config.py`, which
documents the provenance of every default. Frame sampling rate (`fps`) is handled
generically by the tagger runtime.

### Detection

| param | default | meaning |
|---|---|---|
| `detect_target` | `null` → `["brand", "person"]` | What to look for. A known parent (`brand`, `person`) expands to its phrasings; anything else becomes its own parent with itself as the phrasing, so `["car"]` is valid. **Routed** to a detector — see below. |
| `brand_detector` | `"fast"` | `"fast"` = YOLOE-26 @1280; `"coverage"` = Grounding DINO @800. |
| `class_prompts` | brand + person | Explicit `{parent: [phrasings]}`. Normally left alone and driven by `detect_target`. |
| `class_conf` | `{}` | Per-parent confidence override. Empty because **each backend carries its own measured threshold** — see below. |
| `iou` | 0.7 | Ultralytics' internal per-class NMS. |
| `nms_iou` | 0.6 | Synonym-group NMS (stage 2). Set by the coverage constraint, not by fitting. |
| `cross_class_nms_iou` | 1.0 | Cross-parent NMS (stage 3), disabled — and should stay disabled. |
| `max_detections` | 30 | Cap per frame, highest score first. **The primary cost knob** — each survivor is one SigLIP 2 forward pass. For the `fast` backend it is also the *real* gate, since that backend's threshold is near zero. |
| `brand_imgsz` / `brand_conf` | `null` | Override the brand backend's measured defaults. |
| `person_imgsz` / `person_conf` | `null` | Override the person backend's measured defaults. |

#### Routing

`person` goes to YOLO11; every other target goes to the open-vocabulary backend. A detector
with nothing routed to it is never constructed.

```bash
--params '{}'                                  # brand + person, both detectors
--params '{"detect_target": ["person"]}'       # YOLO11 only; brand model never loads
--params '{"detect_target": ["car"]}'          # open-vocab only; person model never loads
--params '{"brand_detector": "coverage"}'      # Grounding DINO for brand, YOLO11 for person
```

Only `person` routes to the closed backend today. Its COCO-80 vocabulary holds 79 other nouns
and routing those there too is a plausible optimisation, but it is **unmeasured** — the
comparison was never run for any class but person — so it is not done.

#### Thresholds are per backend, not shared

There is no global `conf`. Detector scores are not comparable across backends — YOLOE's
text-similarity scores, YOLO11's sigmoid class scores and Grounding DINO's query scores are on
different scales — so one number could only ever be right for one of them. Each backend carries
the threshold that leave-one-clip-out selection chose for it:

| backend | imgsz | conf | threshold range across 11 folds |
|---|---|---|---|
| YOLOE-26 (`fast`) | 1280 | 0.007 | 0.007–0.007 — the most stable in the study |
| Grounding DINO (`coverage`) | 800 | 0.15 | 0.142–0.162 |
| YOLO11 (person) | 640 | 0.11 | 0.101–0.119 |

The previous global default of 0.25 was ultralytics' closed-vocabulary value and measured badly
here: it emitted 0.08 non-person detections per frame — a person detector wearing an
open-vocabulary prompt list.

#### `imgsz` is per detector because one value cannot be right for all three

| model | behaviour with resolution |
|---|---|
| YOLOE-26 | **gains** — brand AP 0.062 @640 → 0.133 @1280 |
| Grounding DINO | **cannot be scaled** — collapses to AP 0.001 above its native 800 |
| YOLO11 | **best at 640** — person AP 0.752 @640 → 0.702 @1280 |

Grounding DINO's collapse is DETR-family behaviour, not a bug: verified that imgsz 800
reproduces the stock default byte-identically and that batch size changes nothing, while
detections fall 2589 → 240.

**Raising `imgsz` does not make crops bigger.** Boxes are rescaled to source coordinates and
crops are taken from the original frame, so a 40 px mark is 40 px at any `imgsz`. Measured:
person crop median is flat at 53/51/52 px across 640/960/1280, while the brand median *falls*
55 → 41 px because higher resolution finds more small marks. Resolution buys recall on small
objects; it makes the average crop *smaller*.

### Crop selection

| param | default | meaning |
|---|---|---|
| `min_box_size` | 0.0 | Drop boxes below this normalized area. |
| `min_crop_pixels` | 16 | Drop crops whose short side is under this many source pixels. Measured, [below](#min_crop_pixels-is-16-and-that-is-a-measurement-it-used-to-be-32). |
| `crop_padding` | 0.06 | Context added each side before cropping; the reported box stays un-padded. |

### Embedding

| param | default | meaning |
|---|---|---|
| `max_num_patches` | 256 | NaFlex resolution budget (16×16 patches). |
| `max_upscale` | `null` | Optional cap on NaFlex upscaling. **Do not set to 1.0** — see below. |
| `normalize` | `true` | L2-normalize so cosine == dot product. |
| `embed_batch_size` | 32 | Crops per forward pass. |
| `embed_whole_frame` | `false` | Also emit one whole-frame vector per frame. |

### Output

| param | default | meaning |
|---|---|---|
| `output_tags` | `false` | Also emit a vector-less tag beside each detection's vector tag, so the detection shows up in EVIE's ordinary tag tracks. Doubles the tag count per frame; no extra inference. [Above](#output_tags--a-plain-tag-beside-each-vector-tag). |

## Notes on the parameters that matter

### Thresholds are calibrated, and the calibration is held out

Earlier versions of this file documented a 128-frame `class_conf` calibration under a
seven-class schema (`logo`/`text`/`product`/`screen`/`board`/`person`/`object`). That schema and
that calibration are both retired. Thresholds now come from box-level ground truth with
**leave-one-clip-out** selection: the threshold is chosen on ten clips and evaluated on the
eleventh, eleven times over.

The split is by **clip, not frame**, because three frames from one possession share camera,
lighting, jerseys and hoardings — and the same physical logos at the same scale. A frame-level
split would put near-duplicates on both sides and inherit most of the optimism it exists to
measure.

What that buys is an honest estimate of what a config file delivers, plus a stability signal:

| | in-sample F1 | held-out F1 | drop | threshold range |
|---|---|---|---|---|
| brand · Grounding DINO @800 | 0.309 | 0.286 | −0.023 | 0.142–0.162 |
| brand · YOLOE-26 @1280 | 0.271 | **0.267** | −0.004 | 0.007–0.007 |
| person · YOLO11 @640 | 0.734 | **0.719** | −0.015 | 0.101–0.119 |
| person · YOLOE-26 | 0.656 | 0.483 | **−0.173** | **0.043–0.171** |

The last row is why YOLO11 serves person. YOLOE-26 looks competitive in-sample and collapses
held out, with chosen thresholds spanning a four-fold range — meaning no single config value
works across content types. That is a deployment property no aggregate metric surfaces.

If a wrong tag costs more than a missing one, select for precision instead of F1:
`python eval/box_gt/operating_point.py --mode precision --target-precision 0.6`. At a 0.6 target
Grounding DINO holds 0.120 held-out F1 with a −0.018 drop.

### What the defaults will not fix

Brand detection is a **small-object problem**, and it is the binding constraint. Ground-truth
marks have a median short side of **22 px**; two thirds are under 32 px. Coverage falls off a
cliff there — Grounding DINO covers 0.92 of marks 80 px and larger but only 0.41 below 16 px,
and it is the best of the field at both ends.

No prompt or threshold moves this, and it is why `min_crop_pixels` is the parameter it is. The
old default of 32 discarded roughly two thirds of the marks the detector is asked to find; it is
now **16**, measured against where a crop actually stops retrieving from a pool —
[see below](#min_crop_pixels-is-16-and-that-is-a-measurement-it-used-to-be-32). That recovers
most of the marks but does not repair the underlying problem: the detector still has to *find* a
22 px mark before any of this applies, and coverage below 16 px is 0.41 even for the best model
in the field.

### `crop_padding` changes the vector (measured)

`crop_padding` adds a margin around the detection box before cropping. At the default 0.06 the
object fills 1/1.12 = **89%** of the crop's linear extent rather than 100%, and the extra pixels
are real image content. SigLIP encodes the whole crop, so two crops of the *same* object at
different padding differ in both content and composition, and their vectors move apart.

That is the mechanism. The size of the effect was measured on the 446 ground-truth boxes, where
object identity is known exactly, by embedding every object at four paddings and asking whether
a query built at one padding still retrieves the same object from an index built at another:

| padding of index | self-similarity vs 0.06 | margin over distractors | **recall@1** |
|---|---|---|---|
| 0.00 | 0.965 | +0.27 | 0.922 |
| **0.06** (shipped) | 1.000 | +0.31 | **1.000** ← control |
| 0.12 | 0.966 | +0.27 | 0.922 |
| 0.25 | 0.932 | +0.24 | 0.755 |

*(brand marks; person behaves the same with a shallower curve — 0.957 at ±0.06, 0.807 at 0.25.
Distractor floor — similarity between* different *objects at matched padding — is 0.69 for brand,
0.73 for person.)*

**Read it as degradation, not corruption.** Same-object vectors stay ~0.24–0.31 above the
distractor floor at every padding, so a mixed index still mostly works. But a mismatch of ±0.06
loses **8% of top-1 retrievals** for brand (4% for person), and a 0.06-against-0.25 mismatch
loses **24%**. Nothing surfaces those as errors — the query returns a confident wrong neighbour.
Note also that 0.00 and 0.12 score identically: what matters is the *magnitude* of the mismatch,
not its direction.

#### Fixing it

1. **Match at query time — free and exact.** Build the query crop at the padding the index was
   written with. Every tag carries `additional_info.crop_padding`, which exists precisely so a
   query pipeline can read it rather than assume it.
2. **Re-embed rather than re-tag.** If padding must change, the boxes are already stored and
   boxes plus source media regenerate crops without re-running detection. This is the cheap
   direction: embedding is ~5.5 ms/crop against 98–293 ms/frame for detection, so re-embedding an
   existing index costs a small fraction of what producing it did.
3. **If an index is already mixed**, partition it by `crop_padding` and query each partition with
   a matched query, then merge the result lists. Exact, and it needs no re-embedding — the
   provenance field is what makes it possible.
4. **Do not try to correct with a learned transform.** Unmeasured here, and the margins above say
   the headroom does not justify it.

Reproduce with `python eval/tools/padding_sensitivity.py --cls brand`.

### `min_crop_pixels` is 16, and that is a measurement (it used to be 32)

SigLIP 2's NaFlex processor binary-searches a scale to *fill* `max_num_patches`, with **no
cap at 1.0** (`scale_max = 100.0` in transformers' `get_image_size_for_max_num_patches`). It
never leaves a small crop small. At `max_num_patches=256`:

| source crop | target | upscale |
|---|---|---|
| 16×16 | 256×256 | 16× |
| 32×32 | 256×256 | 8× |
| 64×64 | 256×256 | 4× |
| 256×256 | 256×256 | 1× |
| 1080×1920 frame | 192×336 | 0.18× |

Whole-frame embedding always *downsamples*; crop embedding almost always *upsamples*, and the
vision tower encodes interpolation blur as texture. **That mechanism is real and it is
confirmed** on the ground-truth crops: a crop's nearest neighbour among other crops is far
closer to it in *size* than chance (median |log₂ size ratio| **0.31** against **0.83** for a
random pair), and crops under 16 px sit **+0.164** above the distractor floor in similarity to
*each other*, against **−0.084** for crops over 48 px. Size genuinely becomes an axis in the
embedding.

What the mechanism could not say is *where the cliff is*, and `32` was a guess at it. The
measurement: downscale a pool query to a target short side, retrieve against native-resolution
references (n=800 brands, gallery 2,382, binomial SE 0.018).

| query short side | 8 px | 12 px | 16 px | 20 px | 24 px | 32 px | 48 px | native |
|---|---|---|---|---|---|---|---|---|
| **recall@1** | 0.124 | 0.354 | **0.598** | 0.748 | 0.821 | 0.911 | 0.934 | 0.936 |
| NaFlex upscale | 22× | 14.8× | 11.1× | 8.9× | 7.4× | 5.6× | 3.7× | 1× |
| **hit/miss cosine gap** | **−0.011** | 0.025 | 0.045 | 0.069 | 0.094 | 0.114 | 0.132 | 0.148 |

The last row is why a floor is needed at all. It is the gap between the top-1 cosine of a
*correct* retrieval and of a *wrong* one. **At 8 px it is negative** — wrong answers come back
more confidently than right ones — so no downstream similarity gate can filter them. Sub-8px
crops are not weak evidence, they are noise asserting itself, and they would poison a
similarity-thresholded index.

#### Why 16 rather than 32

Composing the retrieval curve with the real mark-size distribution (brand median short side
22 px) gives what a threshold actually delivers on this footage:

| `min_crop_pixels` | marks kept | identification recall | identification precision |
|---|---|---|---|
| 0 | 1.000 | 0.741 | 0.741 |
| 12 | 0.896 | 0.715 | 0.799 |
| **16** | **0.807** | **0.671** | **0.832** |
| 20 | 0.615 | 0.542 | 0.882 |
| 24 | 0.448 | 0.411 | 0.918 |
| 32 (old default) | 0.344 | 0.320 | 0.932 |

Marginal precision bought per unit of recall given up: **1.9** from 8→12, **0.75** from 12→16,
then **0.39** from 16→20 and **0.15** from 24→32. The trade turns over at 16. Above it the gate
costs more than it buys, and at 32 it was discarding two thirds of the marks to buy a precision
point already within reach.

**Throughput cost is ~8–11%, not the 2.3× the mark distribution implies** — because the
detectors do not find most sub-32px marks in the first place, so few crops are added:

| detector | crops/frame @32 | crops/frame @16 |
|---|---|---|
| yoloe-26 @1280 | 17.4 | 18.9 |
| grounding-dino @800 | 14.5 | 15.6 |
| yolo11 @640 | 10.9 | 12.1 |

**And a low floor is the recoverable error.** Every tag carries `additional_info.upscale`,
which at a fixed patch budget is a monotone function of crop area, so a consumer can raise the
effective floor by filtering. A floor set too *high* is not recoverable — the crop was never
embedded, and getting it back means re-decoding the video and re-running detection.

One caveat, stated in the direction it cuts: the curve was measured on clean reference art
downscaled cleanly. A real broadcast crop at the same pixel size carries motion blur and
compression that this does not, so every recall figure above is **optimistic**, and 16 is a
floor rather than a comfortable operating point.

Reproduce with `python eval/tools/size_sensitivity.py`.

### Do not set `max_upscale` to 1.0

It does not remove a distribution shift, it swaps it for a more extreme one. At native
resolution a 32×32 crop is encoded from **4 patch tokens** against a documented budget of
256, with position embeddings interpolated onto a 2×2 grid.

It is also actively bad for search. At a fixed budget the sequence length is near-constant
(247–256) across a 24×24 crop and a 200×300 one, so **crop size cannot become an axis in the
embedding**. Under `max_upscale=1.0` it swings 4→247 in lockstep with crop size, making size
a dominant nuisance dimension — you get a "small things" cluster and a "big things" cluster
instead of a logos cluster and a billboards cluster. A moderate cap (~4.0) is the sane
version if this is worth experimenting with; it touches only the smallest crops.

When set, crops are bucketed by budget before batching (the processor takes one
`max_num_patches` per call). Padding is masked out, so a crop's vector depends only on its
own budget, never on which crops it was batched with.

## Model choices

**`google/siglip2-base-patch16-naflex`, 768-d, no projection.**

- **NaFlex, and the reason is cost — not the aspect-ratio argument.** Measured against
  `siglip2-large-patch16-384` on brand-logo retrieval over a 5,953-image gallery, quality is a
  **tie**: r@1 0.926 vs 0.929, well inside the ±0.026 noise band at n=1500. What is not a tie is
  price — **5.95 ms/crop against 36.4 ms**, GPU-only on already-decoded crops, so at ~17
  crops/frame the embedder costs 101 ms/frame instead of 619 ms.

  The aspect argument is a mechanism, not a measured advantage. Fixed-resolution variants do
  resize every input to a square, so a 20×160 banner becomes 96×672 under NaFlex but 384×384
  under `-384`. That squash is real, and an earlier version of this section asserted it decides
  the choice. **It does not, at retrieval level.** Restricting to wide queries (aspect ≥ 2.5 —
  half the pool; median 2.59, p90 5.57, max 10.4) NaFlex led at one sample size and `-384` led at
  another, so the effect is noise. Keep NaFlex for the 6× cost advantage; do not cite the squash
  as settled.
- **base rather than so400m**. so400m is ~4–5× the compute per crop, and unlike a
  whole-frame embedder that cost is paid once per *detection*, up to `max_detections` times
  per frame. It is also **ruled out by the vectorstore**: so400m emits 1152-d and the store caps
  at 1024.
- **Escalation order** if crop quality proves short: `max_num_patches` 256 → 576, then
  `google/siglip2-large-patch16-384` (1024-d, the largest that fits the cap) — accepting 6× the
  per-crop cost for a difference that did not register at n=1500. Raising `imgsz` is *not* on
  this list: it finds more small marks, it does not make crops bigger.

### What retrieval actually delivers

Same-domain retrieval (a clean reference image against other references) reaches r@1 **0.93**.
That is the optimistic number, and it is not what the tagger does. Querying a *detected crop*
against the pool — small, motion-blurred, off-angle art against clean references — reaches
**0.60**. Expect the shipped pipeline nearer 0.6 than 0.9.

Failure is **bimodal**: r@5 equals r@1 exactly, so when the right brand is not the top hit it is
not in the top five either. A similarity threshold should therefore separate hits from misses
cleanly, rather than needing to reason about a long tail.

**Pool freshness bounds all of this.** KIA scored 0.00 for both checkpoints — because the pool's
KIA references are the old oval logo while the detections are the 2021 rebrand wordmark. Those
are visually different marks, and no encoder choice repairs a stale reference. A brand that
rebrands is invisible to this index until its pool entry is refreshed. Details in
[eval/experiments/08_embedders/](eval/experiments/08_embedders/).

The emitted dimension is read from the checkpoint (`config.hidden_size`), never hardcoded.
The 768-d vectors **will be right-zero-padded to 1024-d in the vectorstore**; trailing zeros
leave the dot product and both norms unchanged, so cosine similarity is preserved exactly.
This is lossless, unlike a learned projection down to 1024.

## Query side

Text→crop and crop→crop are **different score regimes and need separate thresholds.**

SigLIP/CLIP image and text embeddings do not share a cone on the unit sphere — they sit in
two roughly parallel regions separated by a near-constant offset (the "modality gap").
Typical cos(image, image) for related content lands ~0.5–0.9; cos(image, text) for a perfect
match lands ~0.05–0.3. One similarity threshold cannot serve both: tuned for image→image,
text queries return nothing; tuned for text, image queries return everything.

Crops compound this. The text tower was aligned to whole captioned images, so a tight 60×40
crop of a wordmark is off-distribution for that alignment — workable for prominent branded
objects, weak for the generic classes ("standalone object" as a *query* is close to
meaningless).

SigLIP 2 does give a fix that CLIP does not: its sigmoid loss trains a per-pair calibrated
probability, with no softmax over a batch, so scores are comparable across queries. Threshold
text→crop on that probability rather than on raw cosine:

```python
# Siglip2Model.forward, transformers/models/siglip2/modeling_siglip2.py:888-889
prob = torch.sigmoid(cos * model.logit_scale.exp() + model.logit_bias)
```

Note `logit_scale.exp()`: the stored parameter is the **log** of the scale and must be
exponentiated before use. Both scalars are `nn.Parameter`s on `Siglip2Model` (not on the
`Siglip2VisionModel` loaded here).

### Nothing needs to change at tagging time

This calibration cannot be pre-applied when tagging, and does not need to be:

- **It is a property of a (text, image) *pair*,** not of an image vector. `cos` only exists
  once a query arrives, so there is nothing to fold into a stored vector.
- **The query side already has the constants.** It must load the full `Siglip2Model` anyway
  to embed query text, and `logit_scale`/`logit_bias` come with it. The only thing the tags
  must carry is *which checkpoint* produced them, so the query side loads the same one — and
  `additional_info.embedder` already does that.
- **Trying to close the gap at tag time would break it.** Any transform applied to stored
  image vectors would have to be applied identically to text queries. Subtracting the image
  mean is the classic attempt, and it is exactly the mean-centering failure mode: every text
  query ends up dominated by the modality-gap offset and they all collapse toward one
  direction. Leave the geometry alone.

### The query encoder needs the checkpoint, not the crop settings

`Siglip2Model` out of the box is correct for embedding query **text**. The text tower takes
no image-side argument — `get_text_features(input_ids)` is a function of the text alone — so
`crop_padding`, `max_num_patches` and `imgsz` cannot and do not enter the query embedding.

What they affect is where the *stored* vectors land, and therefore the cosine a given pair
produces. So the requirement is **consistency within the index, not knowledge at the query
encoder**:

- Keep `crop_padding` and `max_num_patches` fixed for everything in one space. That is why
  both are stamped into `additional_info` — as a homogeneity check and filter key, not as a
  query-side input.
- Changing either shifts the score distribution, so a threshold calibrated before the change
  is wrong after it. Re-embed, or partition the index and threshold each partition.
- **Query-by-example is the exception.** When the query is an image rather than text, it goes
  through the *vision* tower, so it should be preprocessed the way the indexed crops were —
  same `max_num_patches`, and comparable framing if the query is itself a crop.

One tagging-time knob does genuinely trade off here: **`crop_padding`**. The text tower was
aligned to whole captioned images, so more surrounding context moves a crop closer to that
distribution. If text→crop search matters more than crop→crop, that argues for padding above
the shipped 0.06 — but pick a value and hold it for the life of the index.

The highest-precision path needs no cross-modal calibration at all: crop→crop retrieval
against labelled exemplars, which is what `model-logo` does with its feature pool.

## Build

```bash
chmod +x build.sh
make build          # or: ./build.sh   (no weights needed at build time)
```

### Rebuilding after a `common-ml` change (stale layer in podman)

```bash
git submodule update --init --recursive
buildscripts/build_container.bash -t general_detection:latest . -f Containerfile --no-cache
```

## Deployment — requires a persistent `/root/.cache` mount

**No weights are baked into this image.** Two separate downloads happen on first load:

- SigLIP 2 from the HuggingFace hub → `HF_HOME`
- the YOLOE checkpoint **and** the MobileCLIP text encoder that `get_text_pe()` needs →
  `storage.cache_path`. Ultralytics resolves both relative to the CWD, which
  `general_detection/detector.py` handles by chdir-ing into the cache during load.

Both live under `/root/.cache`, so **one** mounted volume there covers both. Without it they
land in the container's ephemeral writable layer and are re-fetched every run. See `test.sh`
for a working invocation (`--volume=detection_cache:/root/.cache`).

## Tests

```bash
pip install -e .[test]
pytest tests/
```

The unit tests stub the detector and embedder, so they need no weights and no GPU. Set
`ELV_DETECTION_INTEGRATION=1` to additionally run the end-to-end test against real weights
(requires `test-files/1.mp4` and a GPU).

For a container smoke test, drop media into `test-files/` and run `./test.sh` (optionally
`./test.sh 5` for fps=5).

```bash
make test # default detector and prompts (default path)

IMAGE_NAME=general_detection ./buildscripts/testers/test-model.sh --params '{"output_tags": true}' # each detection also emits a vector-less tag, for the EVIE tag tracks

IMAGE_NAME=general_detection ./buildscripts/testers/test-model.sh --params '{"brand_detector": "coverage", "detect_target": ["brand"]}' 2>&1 | grep -iE "targets:|loading|grounding text|TEST PASSED|TEST FAILED|out.jsonl:|✔|✘|Traceback|Error|error code" | head -30 # alternative detector (higher coverage) and brand only prompt
```

## License

**AGPL-3.0** — see [LICENSE](LICENSE). This tagger links `ultralytics` (YOLOE), which is
AGPL-3.0, so the combined work is AGPL-3.0.

The detector is isolated behind `general_detection/detector.py` specifically so it can be
replaced. Swapping in an Apache-2.0 open-vocabulary detector (OWLv2, Grounding DINO — the
former would keep the whole stack inside HF transformers) means editing that one module;
nothing downstream depends on YOLOE.
