# 09 — `min_crop_pixels`

**Result: 32 → 16.** The gate's mechanism is real, but 32 was set by that mechanism rather than
by data, and it was discarding two thirds of the marks the detector exists to find in order to
buy a precision point that 16 already gets most of the way to.

Reproduce: `python eval/tools/size_sensitivity.py` (writes `scores.json`).

## The question

`min_crop_pixels` drops a detection whose **un-padded** box has a shorter side below N source
pixels. The original 32 came from an argument about upscale factor, not a measurement:

> SigLIP 2's NaFlex processor binary-searches a scale to *fill* `max_num_patches`, with no cap
> at 1.0 (`scale_max = 100.0` in transformers' `get_image_size_for_max_num_patches`). A 16 px
> crop is upscaled ~11× at budget 256. The vision tower encodes interpolation blur as texture,
> so heavily upscaled crops cluster with *each other* rather than by content.

Two things had to be established separately: whether that happens, and where the cliff is. The
first is a mechanism question, the second is an operating-point question, and the second does
not follow from the first.

## (1) Benefit — retrieval against query crop size

Pool-to-pool retrieval with the **query downscaled** to a target short side and the gallery left
at native resolution. That is the production query shape — a small crop against clean reference
art — and unlike the ground truth it carries real brand identity, so recall is measured rather
than inferred. 800 queries, 2,382 gallery images, 800 brands; binomial SE 0.018.

| query short side | r@1 | r@5 | MRR | NaFlex upscale | vs native | **hit/miss cosine gap** |
|---|---|---|---|---|---|---|
| 8 px | 0.124 | 0.159 | 0.145 | 22.0× | −0.812 | **−0.011** |
| 12 px | 0.354 | 0.404 | 0.384 | 14.8× | −0.583 | 0.025 |
| **16 px** | **0.598** | 0.660 | 0.630 | 11.1× | −0.339 | 0.045 |
| 20 px | 0.748 | 0.797 | 0.773 | 8.9× | −0.189 | 0.069 |
| 24 px | 0.821 | 0.871 | 0.844 | 7.4× | −0.115 | 0.094 |
| 32 px | 0.911 | 0.936 | 0.923 | 5.6× | −0.025 | 0.114 |
| 48 px | 0.934 | 0.958 | 0.944 | 3.7× | −0.003 | 0.132 |
| 64 px | 0.939 | 0.960 | 0.948 | 2.8× | +0.002 | 0.140 |
| 96 px | 0.941 | 0.963 | 0.950 | 1.9× | +0.005 | 0.143 |
| native | 0.936 | 0.964 | 0.949 | 1.0× | — | 0.148 |

**The last column is the one that justifies having a floor at all.** It is the gap between the
mean top-1 cosine of a *correct* retrieval and of a *wrong* one. At 8 px it is **negative**: the
wrong answers come back more confidently than the right ones. A crop that small does not fail
quietly and it cannot be rescued by a downstream similarity threshold — it is noise asserting
itself, and it would poison a similarity-gated index. The gap recovers monotonically with size
and is essentially back to native by 32 px.

Note also that 32 px is already **statistically tied with native** (−0.025 against a 0.035 tie
band). Everything above 32 is free of retrieval cost and pure yield cost.

**Direction of the caveat:** clean reference art downscaled cleanly is easier than a real
broadcast crop of the same pixel size, which carries motion blur and compression this does not.
Every number above is therefore optimistic, which makes the chosen value a *floor* rather than a
comfortable operating point.

## (2) Mechanism — is size the dominant axis on real crops?

192 ground-truth `brand` crops at the shipped `crop_padding` of 0.06.

If heavy upscaling makes size an axis, a crop's nearest neighbour should be closer to it in size
than chance. It is, decisively:

| | median \|log₂ size ratio\| |
|---|---|
| crop → its nearest neighbour | **0.31** |
| crop → a random crop | 0.83 |

And by size bin, mean cosine *within* the bin against the all-pairs floor of 0.699:

| size bin | n | mean sim within | vs floor |
|---|---|---|---|
| 0–16 px | 37 | 0.863 | **+0.164** |
| 16–24 px | 69 | 0.782 | +0.083 |
| 24–32 px | 20 | 0.720 | +0.022 |
| 32–48 px | 28 | 0.641 | −0.058 |
| 48 px+ | 38 | 0.615 | −0.084 |

Monotone, and it crosses the floor between 24 and 32 px. **The mechanism the old default was
built on is confirmed.** It is confounded — small marks may genuinely resemble one another more
than large ones do — but the confound and the mechanism are the same phenomenon seen from two
sides, since "all blurry things look alike" is what encoding interpolation as texture means.

## (3) Cost — what each threshold discards

Fraction of ground-truth objects surviving, from the box GT (192 brand, 254 person):

| `min_crop_pixels` | brand | person |
|---|---|---|
| 0 | 1.000 | 1.000 |
| 8 | 0.979 | 1.000 |
| 12 | 0.896 | 0.988 |
| **16** | **0.807** | **0.969** |
| 20 | 0.615 | 0.949 |
| 24 | 0.448 | 0.933 |
| 32 | 0.344 | 0.866 |
| 48 | 0.198 | 0.646 |
| 64 | 0.109 | 0.437 |

Brand median short side 22 px (p10 12, p90 67); person 59 px (p10 28, p90 156). Person is barely
affected anywhere in this range — the parameter is a brand parameter in practice.

## (4) Decision — the curves composed

Interpolating (1) at each ground-truth mark's own size and applying the gate of (3):

- **identification recall** — of all marks, the fraction that survive the gate *and* retrieve
  the right brand. Gating can only lower it.
- **identification precision** — of the marks that survive, the fraction that retrieve the right
  brand. Gating raises it.

| `min_crop_pixels` | kept | ident recall | ident precision | hit/miss gap |
|---|---|---|---|---|
| 0 | 1.000 | 0.741 | 0.741 | 0.085 |
| 8 | 0.979 | 0.739 | 0.754 | 0.087 |
| 12 | 0.896 | 0.715 | 0.799 | 0.094 |
| **16** | **0.807** | **0.671** | **0.832** | **0.100** |
| 20 | 0.615 | 0.542 | 0.882 | 0.114 |
| 24 | 0.448 | 0.411 | 0.918 | 0.126 |
| 32 | 0.344 | 0.320 | 0.932 | 0.132 |
| 48 | 0.198 | 0.186 | 0.939 | 0.140 |

Marginal precision bought per unit of recall given up:

| step | Δprecision | Δrecall | ratio |
|---|---|---|---|
| 8 → 12 | +0.045 | −0.024 | **1.9** |
| 12 → 16 | +0.033 | −0.044 | **0.75** |
| 16 → 20 | +0.050 | −0.129 | 0.39 |
| 20 → 24 | +0.036 | −0.131 | 0.27 |
| 24 → 32 | +0.014 | −0.091 | 0.15 |

The trade turns over at 16. Below it the gate is buying more than it costs; above it, less.

## Why 16 and not something more conservative

Three arguments, in ascending order of weight.

1. **Yield.** 16 keeps 81% of marks against 34% at 32, and end-to-end identification recall more
   than doubles (0.671 against 0.320) for 0.10 of precision.

2. **The throughput cost is small.** The mark distribution suggests 2.3× more crops; the real
   figure is **8–11%**, because the detectors do not find most sub-32px marks in the first
   place. Measured on the ground-truth frames at each backend's shipping threshold, after
   `max_detections=30`:

   | detector | crops/frame @32 | crops/frame @16 | Δ |
   |---|---|---|---|
   | yoloe-26 @1280 | 17.4 | 18.9 | +9% |
   | grounding-dino @800 | 14.5 | 15.6 | +8% |
   | yolo11 @640 | 10.9 | 12.1 | +11% |

3. **The errors are asymmetric, and this is the argument that settles it.** Setting the floor
   too low is *recoverable*: every tag carries `additional_info.upscale`, which at a fixed patch
   budget is a monotone function of crop area, so a consumer that wants a stricter floor filters
   for it. Setting the floor too high is *not* recoverable — the crop was never embedded, and
   getting it back means re-decoding the video and re-running detection, which is 98–293 ms per
   frame against ~6 ms per crop to embed. Given a symmetric-looking trade, take the side whose
   mistake can be undone downstream.

## What this does not settle

- The floor stays a **brand** decision. Person is at 0.969 yield at 16 and would tolerate a much
  higher floor; nothing here argues for splitting the parameter per class, but it could be.
- 8 px is established as *harmful* rather than merely useless, and 16 is the first size where
  the hit/miss gap is comfortably positive. Whether the right answer under real broadcast
  degradation is 16 or 20 is inside the caveat on (1) and is not resolved here.
