#!/usr/bin/env python3
"""Rank a detector vocabulary by how well each term maps to `brand`, `person` or neither.

Why not just write the list by hand
-----------------------------------
The first attempt at a hand-written tag map had 35 of its 60 terms never fire on this
footage, while missing ones that did -- "leather shoe", "denim jacket", "neckband", and the
vocabulary's own misspelling "dinning table". Regex mining is no better: matching /man/
against the 4585-term vocabulary returns "mango", "manhole", "manatee" and "pavement"
alongside "man" and "businessman".

Why SigLIP 2's text tower is the right judge
--------------------------------------------
Not an arbitrary choice of embedding model. Under the "general detection + downstream
filtering" pipeline (method 2), the class decision is *already* made by cosine similarity in
SigLIP 2 space -- a query embedding against crop embeddings. Scoring vocabulary terms in that
same space is therefore a preview of the mechanism that will do the real filtering, not a
proxy for it.

The model is loaded straight from HF transformers into model-detection's own cache. It shares
nothing with any other SigLIP 2 deployment.

What comes out
--------------
A ranked proposal, not an answer. Terms land in one of four buckets and the file is meant to
be read and corrected by hand before use:

    brand      the term names something whose identity is a brand
    person     the term names a person or people
    OTHER      a real object, but not a target -- counts toward breadth
    NONOBJECT  not an object at all. The prompt-free vocabularies are full of these
               ("darkness", "wedding reception", "brunette", "tournament", "hairstyle");
               they cannot be boxed, and counting them as OTHER would inflate a detector's
               apparent breadth with words it cannot localise.

Prototype sets are a variable, not a constant
---------------------------------------------
How `brand` is worded decides which terms it collects, so the wording is treated as something
to measure rather than assert. Three sets ship (--protos), and --diff reports exactly which
terms move between two of them, which is the evidence for choosing one:

    narrow      logos and cars. Faithful to the old `logo` class, and the reason it is here
                is as a baseline that visibly under-covers the reframing.
    categories  narrow, plus one SHORT concrete prototype per brand category (fashion,
                sports, automotive, electronics, beauty, retail, food and beverage,
                healthcare). Short and concrete because SigLIP is trained on image captions.
    enumerated  narrow, plus one long definitional sentence naming all eight categories at
                once. Kept to test the hypothesis that it is too diffuse: pooling a
                many-way enumeration into a single vector lands near the centroid of those
                categories rather than near any of them, and pulls toward the category
                *words* -- risking "boutique", "pharmacy", "industry" scoring as brand.

Usage
-----
    python eval/mine_vocab.py --source both --protos categories --show 40
    python eval/mine_vocab.py --diff eval/tag_map_narrow.json eval/tag_map_categories.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

_PERSON = [
    "a photo of a person",
    "a man standing",
    "a woman standing",
    "an athlete playing a sport",
    "a person's face",
    "a speaker addressing an audience",
    "people in a crowd",
]
_OTHER = [
    "a photo of an ordinary object",
    "a piece of furniture",
    "an animal",
    "a plant",
    "a tool",
    "a building",
    "a piece of food",
]
_NONOBJECT = [
    "an abstract concept",
    "a type of scene or location",
    "an activity or event",
    "a colour or texture",
    "a style or mood",
    "a hairstyle or physical attribute",
    "a season or time of day",
]

# The half of `brand` that survives from the old schema: the mark, and the car whose bodywork
# identifies it without a legible badge.
_BRAND_NARROW = [
    "a brand logo",
    "a company logo printed on a product",
    "a car of a recognisable make and model",
    "a product with a visible brand name",
]

# One short, concrete, caption-shaped prototype per brand category. Named products are used
# deliberately: SigLIP has seen these captions, and they anchor the category far more tightly
# than the category noun does.
_BRAND_CATEGORIES = _BRAND_NARROW + [
    "a designer handbag",                  # fashion
    "a team jersey with a sponsor logo",   # sports
    "a smartphone on a table",             # electronics
    "a bottle of perfume",                 # beauty
    "a shelf of packaged groceries",       # retail
    "a can of soft drink",                 # food and beverage
    "a box of medicine",                   # healthcare
]

_BRAND_ENUMERATED = _BRAND_NARROW + [
    "a design or symbol unique to a fashion, sports, automotive, electronics, beauty, "
    "retail, food and beverage, or healthcare item",
]

PROTOTYPE_SETS: Dict[str, Dict[str, List[str]]] = {
    "narrow": {"brand": _BRAND_NARROW, "person": _PERSON,
               "OTHER": _OTHER, "NONOBJECT": _NONOBJECT},
    "categories": {"brand": _BRAND_CATEGORIES, "person": _PERSON,
                   "OTHER": _OTHER, "NONOBJECT": _NONOBJECT},
    "enumerated": {"brand": _BRAND_ENUMERATED, "person": _PERSON,
                   "OTHER": _OTHER, "NONOBJECT": _NONOBJECT},
}

BUCKETS = ["brand", "person", "OTHER", "NONOBJECT"]


def emitted_tags(runs_glob: str) -> List[str]:
    seen: Dict[str, int] = {}
    for path in sorted(glob.glob(runs_glob)):
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)["data"]
                except Exception:
                    continue
                tag = " ".join(str(data.get("tag", "")).lower().split())
                if tag:
                    seen[tag] = seen.get(tag, 0) + 1
    return sorted(seen, key=lambda t: -seen[t])


def do_diff(path_a: str, path_b: str) -> int:
    with open(path_a) as handle:
        a = json.load(handle)
    with open(path_b) as handle:
        b = json.load(handle)
    map_a, map_b = a["map"], b["map"]
    moved: Dict[str, List[str]] = {}
    for term, bucket in map_a.items():
        other = map_b.get(term)
        if other and other != bucket:
            moved.setdefault(f"{bucket} -> {other}", []).append(term)
    name_a = os.path.basename(path_a)
    name_b = os.path.basename(path_b)
    print(f"{name_a}  ->  {name_b}")
    print(f"  {a['counts']}\n  {b['counts']}")
    total = sum(len(v) for v in moved.values())
    print(f"\n{total} of {len(map_a)} terms changed bucket\n")
    for transition, terms in sorted(moved.items(), key=lambda kv: -len(kv[1])):
        print(f"  {transition}  ({len(terms)})")
        for i in range(0, min(len(terms), 60), 5):
            print("      " + "  |  ".join(terms[i : i + 5]))
        if len(terms) > 60:
            print(f"      ... and {len(terms) - 60} more")
        print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diff", nargs=2, metavar=("A", "B"),
                        help="compare two proposals and list the terms that moved")
    parser.add_argument("--source", default="both", choices=["vocab", "emitted", "both"])
    parser.add_argument("--protos", default="categories", choices=list(PROTOTYPE_SETS))
    parser.add_argument("--vocab-dump", default=paths.VOCAB_DUMP)
    parser.add_argument("--runs", default=os.path.join(paths.EXPERIMENTS, "*", "runs*", "*", "out.jsonl"))
    parser.add_argument("--model", default="google/siglip2-base-patch16-naflex")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--out", default=None)
    parser.add_argument("--show", type=int, default=0, help="print the top N of each bucket")
    args = parser.parse_args()

    if args.diff:
        return do_diff(*args.diff)

    import torch
    from transformers import AutoModel, AutoTokenizer

    prototypes = PROTOTYPE_SETS[args.protos]
    out_path = args.out or os.path.join(paths.TOOLS, f"tag_map_{args.protos}.json")

    terms: List[str] = []
    if args.source in ("vocab", "both"):
        with open(args.vocab_dump) as handle:
            dump = json.load(handle)
        for key in ("yoloe_pf", "coco80"):
            for term in dump.get(key, []):
                norm = " ".join(term.lower().split())
                if norm not in terms:
                    terms.append(norm)
    if args.source in ("emitted", "both"):
        for term in emitted_tags(args.runs):
            if term not in terms:
                terms.append(term)

    print(f"{len(terms)} distinct terms, prototype set '{args.protos}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    def embed(strings: List[str]) -> "torch.Tensor":
        chunks = []
        for start in range(0, len(strings), args.batch):
            batch = strings[start : start + args.batch]
            # SigLIP 2 trains with a fixed 64-token padded text length; matching it keeps
            # these embeddings on the same footing as the query encoder at serving time.
            inputs = tokenizer(batch, padding="max_length", max_length=64, truncation=True,
                               return_tensors="pt").to(device)
            with torch.no_grad():
                feats = model.get_text_features(**inputs)
            # transformers 5.x returns BaseModelOutputWithPooling here rather than a tensor.
            if not isinstance(feats, torch.Tensor):
                feats = feats.pooler_output
            chunks.append(torch.nn.functional.normalize(feats, dim=-1).cpu())
            print(f"  embedded {min(start + args.batch, len(strings))}/{len(strings)}",
                  end="\r", flush=True)
        return torch.cat(chunks)

    proto = torch.stack([
        torch.nn.functional.normalize(embed(prototypes[b]).mean(0), dim=-1) for b in BUCKETS
    ])

    # "a photo of a X" matches the caption distribution SigLIP was trained on far better than
    # a bare noun, which otherwise reads as a word rather than as a depicted thing.
    vectors = embed([f"a photo of a {t}" for t in terms])
    print()

    sims = vectors @ proto.T
    best = sims.argmax(dim=1)
    ordered = sims.sort(dim=1, descending=True).values
    margin = (ordered[:, 0] - ordered[:, 1]).tolist()

    mapping: Dict[str, str] = {}
    detail: Dict[str, Dict] = {}
    for i, term in enumerate(terms):
        bucket = BUCKETS[int(best[i])]
        mapping[term] = bucket
        detail[term] = {"margin": round(float(margin[i]), 4),
                        "scores": {b: round(float(sims[i][j]), 4)
                                   for j, b in enumerate(BUCKETS)}}

    counts = {b: sum(1 for v in mapping.values() if v == b) for b in BUCKETS}
    print(f"proposal: {counts}")

    with open(out_path, "w") as handle:
        json.dump({
            "note": "PROPOSAL from eval/mine_vocab.py -- review by hand before use.",
            "prototype_set": args.protos,
            "prototypes": prototypes,
            "counts": counts,
            "map": mapping,
            "detail": detail,
        }, handle, indent=1)
    print(f"wrote {out_path}")

    if args.show:
        for bucket in BUCKETS:
            rows = sorted(((detail[t]["margin"], t) for t in terms
                           if mapping[t] == bucket), reverse=True)
            print(f"\n=== {bucket} ({len(rows)}) — top {args.show} by margin ===")
            for value, term in rows[: args.show]:
                print(f"   {value:.3f}  {term}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
