"""Detection targets for the entity tagger, and the routing from a target to a detector.

The schema
----------
Two classes, measured into their current form against box-level ground truth (see eval/):

    brand   six MARK terms, and deliberately no object terms. `brand` means the mark itself --
            the GAP wordmark, not the hoodie; the NFL shield, not the helmet. Asked for
            `sportswear` a detector returns the garment; asked for `logo` it returns the
            wordmark on it, and the wordmark is the crop that retrieves against a logo pool.
            With 101 concrete object nouns only 1% of brand detections carried a mark-like
            label (63% were `shoe`); with these six, 100% do.

    person  one word. `person` alone reaches 0.92-0.97 class-agnostic coverage on every
            promptable backend -- identical to a 101-term list including 18 role words
            (player, referee, commentator, fashion model). The role words bought nothing.

Mark-CARRYING surfaces (`sign`, `banner`, `billboard`) were tested and rejected: they reproduce
the overshadowing failure one level up, since a banner is a surface a logo sits on, so the box
lands on the banner. `symbol` was tested as a seventh brand term and rejected too -- it costs the
leading model AP and wins the argmax on boxes `logo` already had.

Routing
-------
The two classes want different detectors, and nothing forces one model to serve both. `person`
goes to a closed COCO detector, which beats every open-vocabulary model at the one class COCO was
built around while being the cheapest model in the study. Everything else goes to the
open-vocabulary detector.

A caller-supplied target is routed the same way, so `--params '{"detect_target": ["car"]}'` runs
only the open-vocabulary detector and `["person"]` runs only the closed one. Only `person` routes
to the closed detector today. Its COCO-80 vocabulary holds 79 other nouns and routing those there
too is a plausible optimisation, but it is unmeasured -- the comparison was never run for any
class but person -- so it is not done.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# The mark terms. Order is stable because it fixes class ids for a given config.
BRAND_PROMPTS: List[str] = ["logo", "letter logo", "car logo", "emblem", "brand", "label"]
PERSON_PROMPTS: List[str] = ["person"]

DEFAULT_CLASS_PROMPTS: Dict[str, List[str]] = {
    "brand": list(BRAND_PROMPTS),
    "person": list(PERSON_PROMPTS),
}

# Parents the closed detector serves. Everything else goes to the open-vocabulary one.
CLOSED_VOCAB_PARENTS = {"person"}

# The COCO-80 label the closed detector emits for each parent it serves. Kept explicit rather
# than assuming parent == COCO name, so a parent could be renamed without breaking the lookup.
CLOSED_VOCAB_LABEL: Dict[str, str] = {"person": "person"}


def expand_target(target: List[str]) -> Dict[str, List[str]]:
    """Turn a caller's target list into the {parent: [phrasings]} form the detectors take.

    A term naming a known parent expands to that parent's phrasings, so `brand` becomes the six
    mark terms rather than the literal word -- which matters, because the bare word `brand` is a
    far weaker prompt than the mark list and only Grounding DINO grounds it at all. Any other
    term becomes its own parent with itself as the single phrasing.
    """
    out: Dict[str, List[str]] = {}
    for term in target:
        key = term.strip()
        if not key:
            continue
        out[key] = list(DEFAULT_CLASS_PROMPTS.get(key, [key]))
    if not out:
        raise ValueError("detect_target resolved to no usable terms")
    return out


def split_by_detector(class_prompts: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]],
                                                                    Dict[str, List[str]]]:
    """Partition {parent: [phrasings]} into (open_vocab, closed_vocab).

    Either side may come back empty, and the caller is expected to skip that detector entirely
    rather than run it on nothing -- a person-only request should never load the open-vocabulary
    model, which is the largest available saving for the common case.
    """
    closed = {k: v for k, v in class_prompts.items() if k in CLOSED_VOCAB_PARENTS}
    openv = {k: v for k, v in class_prompts.items() if k not in CLOSED_VOCAB_PARENTS}
    return openv, closed


def flatten(class_prompts: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    """Return (prompts, parent_label_per_prompt), index-aligned.

    The detector's class ids index into `prompts`, so the two lists must stay parallel:
    class id -> prompt -> parent label. Ordering follows `class_prompts` insertion order,
    which keeps class ids stable for a given config.
    """
    prompts: List[str] = []
    labels: List[str] = []
    for label, phrasings in class_prompts.items():
        if not phrasings:
            raise ValueError(f"class {label!r} has no prompts")
        for phrasing in phrasings:
            if phrasing in prompts:
                # One phrasing cannot belong to two parents: the detector emits a single
                # class id for it and the parent choice would be arbitrary.
                raise ValueError(f"prompt {phrasing!r} is used by more than one class")
            prompts.append(phrasing)
            labels.append(label)
    if not prompts:
        raise ValueError("class_prompts is empty")
    return prompts, labels
