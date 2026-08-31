#!/usr/bin/env python3
"""The brand / person schema: what the tagger prompts for.

The pipeline this serves
------------------------
    source media  ->  detector (default prompts, or a caller-supplied text target)
                  ->  crops    ->  SigLIP 2 embeddings  ->  vector index

Identification is not a detection job. "Which brand is this" and "is this the same person"
are answered downstream by cosine similarity against a pool, which is what model-logo and
model-celeb already do. The detector's only job is to produce crops that are TIGHT AROUND THE
THING ITSELF, so the embedding encodes the entity and not its surroundings.

Three modes, one prompt list
----------------------------
    no target      the default prompts below: brand marks and people
    text target    the caller's terms, used verbatim
    general        no prompts at all -- a prompt-free detector catalogues whatever is there,
                   and its native labels pass through as metadata

Note what is NOT here: no tag map, no class taxonomy over detector vocabularies. In the first
two modes a detector can only answer with terms it was given, so a map would be a no-op. In the
general mode the native label is metadata rather than the retrieval key -- a downstream query
for "basketball" is matched against the crop EMBEDDING, not against the label string -- so
normalising those labels into brand/person/other buckets would be work that nothing consumes.
(Such a map used to exist, purely so the prompt-free runs could be scored. It has been retired
to tools/deprecated/ -- class-agnostic IoU coverage in box_gt/score_boxes.py measures the same
thing by observation instead of by hand-maintained keyword rules.)

Two prompt lists, nothing else
------------------------------
An earlier version of this file had five prompt tiers, a strict/loose split, and 101 terms.
Measurement collapsed it to this:

    person   One word. "person" alone reaches recall 1.00 on every promptable backend --
             identical to the full 101-term list including 18 role words (player, referee,
             commentator, fashion model...). The role words bought nothing.

    brand    Six MARK terms, and deliberately no object terms. This is the largest single
             effect found in the whole sweep. With 101 concrete object nouns, only 1% of
             Grounding DINO's brand detections carried a mark-like label -- 63% were `shoe`,
             12% `sportswear`. `logo` and `brand` were both in that list; they simply lose the
             argmax to whatever garment the mark is printed on. With the six mark terms alone,
             100% are marks.

             It is not only the label that changes, it is the crop. Asked for `sportswear` you
             get the hoodie; asked for `logo` you get the GAP wordmark on it. For retrieval
             against a logo pool the wordmark is the crop you need. The geometry follows:
             mark-only crops are 1% under 32px (median 84px) against 51% under 32px
             (median 31px) for the 101-term list.

Why not a seventh term
----------------------
`symbol` was the strongest remaining candidate and was tested head-to-head against box ground
truth, both added to the six and on its own. Added, it *costs* the leader -- owlv2 brand AP
0.310 -> 0.284 -- and moves nothing else beyond noise. On its own it reaches coverage 0.56 on
gdino but only 0.17 on owlv2, so it is a weaker standalone term as well.

It is not a bad term; that is the point. 8 of its 14 detections on ground-truth frames land on
real marks. It simply wins the argmax on boxes `logo` already had and re-ranks them lower, which
is what adding a near-synonym to a single-label attribution scheme does. The list is at the point
where more terms cost more than they return.

Why no mark-CARRYING surfaces
-----------------------------
`sign`, `banner`, `billboard` and `advertisement` were tested and rejected. They reproduce the
same overshadowing failure one level up -- a banner is a surface a logo sits on, so the box
lands on the banner. Adding them dropped mark share from 100% to 64% (gdino), 33% (owlv2) and
4% (world-text, whose apparent recall gain was 20 banners and a single car logo).

Why concrete mark nouns rather than the bare word "brand"
---------------------------------------------------------
Both were measured. The bare word works only on Grounding DINO, whose BERT cross-attention can
ground an abstract query; the CLIP-style encoders compare one pooled text vector against region
vectors and have no useful embedding for "brand" -- world-text managed a single detection in
100 frames at score 0.008, yoloe26-text topped out at 0.068 against 0.920 for its own concrete
nouns. The six concrete mark nouns rescue them (world-text 0.01 -> 0.59 recall, yoloe11-text
0.19 -> 0.71) and cost Grounding DINO nothing: 191 mark detections against 69 for the bare word.

Image prompts are deliberately not supported
--------------------------------------------
Only YOLOE accepts them, and 42 single-exemplar runs showed they add nothing this pipeline
needs. For brand they were degenerate -- a Nike crop produced near-full-frame boxes on
arbitrary scenes at 0.02-0.26, not localisation at all. For person they worked but generalised
to the category: a basketball-referee exemplar found every person in the footage, which is what
the word "person" already gives at recall 1.00.

That is expected behaviour rather than a bug. A visual prompt builds a *category* prototype in
YOLOE's own 512-d space; finding a specific logo or a specific person is an *instance* question,
and instance matching belongs in SigLIP space over real crops -- downstream, where the pool
lives. So the optional target parameter is text-only, and image search is a query-side operation
on the embeddings.

Dropping the image-prompt requirement also frees the detector choice: it was the one thing that
forced YOLOE, and with it AGPL-3.0. Grounding DINO and OWLv2 are both Apache-2.0.
"""
from __future__ import annotations

from typing import Dict, List

CLASSES = ["brand", "person"]

# Six mark terms, all present in the yoloe-pf vocabulary (checked against vocab_dump.json).
# "wordmark", "trademark" and "watermark" are not in it and are omitted.
BRAND_PROMPTS: List[str] = [
    "logo", "letter logo", "car logo", "emblem", "brand", "label",
]

PERSON_PROMPTS: List[str] = ["person"]

DEFAULT_PROMPTS: List[str] = BRAND_PROMPTS + PERSON_PROMPTS

# tag -> class for the default prompts. A promptable detector cannot emit anything else, so
# this is total over its possible output rather than a best-effort mapping.
PROMPT_CLASS: Dict[str, str] = {
    **{term: "brand" for term in BRAND_PROMPTS},
    **{term: "person" for term in PERSON_PROMPTS},
}


def _norm(tag: str) -> str:
    return " ".join(str(tag).lower().split())


def class_of_prompt(term: str) -> str:
    """'brand', 'person', or 'OTHER' for a caller-supplied target term."""
    return PROMPT_CLASS.get(_norm(term), "OTHER")


# --------------------------------------------------------------------------------------
# Provisional labels derived from the 8-class presence labels
# --------------------------------------------------------------------------------------
# The 100 frozen frames were labelled under an older schema in which `brand` did not exist.
# This bootstraps the new labels so numbers exist before box-level ground truth lands. It is
# an APPROXIMATION and every table built on it is marked provisional.
# `logo` alone. The old schema's `car` and `bottle_or_cup` are brand-BEARING objects, which
# was the earlier definition of brand; under the mark definition they are not brand at all.
DERIVE_BRAND_FROM = ["logo"]
DERIVE_PERSON_FROM = ["person"]


def derive(old_present: List[str]) -> List[str]:
    present = set(old_present)
    out = []
    if present & set(DERIVE_BRAND_FROM):
        out.append("brand")
    if present & set(DERIVE_PERSON_FROM):
        out.append("person")
    return out


if __name__ == "__main__":
    print(f"brand  ({len(BRAND_PROMPTS)}): {', '.join(BRAND_PROMPTS)}")
    print(f"person ({len(PERSON_PROMPTS)}): {', '.join(PERSON_PROMPTS)}")
    print(f"default prompt list: {len(DEFAULT_PROMPTS)} terms")
