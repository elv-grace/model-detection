#!/usr/bin/env python3
"""Build eval/tag_map.json: how a raw detector tag is interpreted when scoring.

Different construction per class, because the evidence differs
--------------------------------------------------------------
`person` is taken from the SigLIP 2 mining proposal (eval/mine_vocab.py). That is defensible:
person had a median decision margin of 0.0270 against the next-best bucket, roughly three
times brand's, so the ranking carries real signal. Short hand-written force and veto lists fix
what the margin cannot -- body parts and attributes that are not people ("forehead",
"manicure", "hairstyle"), and collective nouns ("crowd", "team", "squad") which are excluded
on purpose: they name one box around many people and destroy the per-instance crops the
downstream pose distinction needs.

`brand` is NOT taken from the mining, because the mining showed it cannot be: brand's median
margin (0.0093) sits at the NONOBJECT noise floor, and the two prototype wordings that bracket
it assign 60 and 1876 terms respectively with nothing usable in between. So brand is built
from explicit category families instead -- automotive, fashion, footwear, bags, eyewear,
jewellery, electronics, food and beverage, beauty and health, and sports equipment, plus the
marks and surfaces brands appear on. The families are written out rather than inferred so
they can be argued with.

What `brand` means in this map
------------------------------
BRAND-BEARING, not brand-identified. The map answers "is this tag naming an object whose
identity could be a brand", which is the question method 2 needs: such an object is worth
cropping and embedding, and whether it IS a recognisable brand is settled downstream by cosine
against the pool. It deliberately does not try to answer "is this a brand", which the mining
demonstrated is not answerable from a label at all.

The test for a family member is "could you name the brand from its appearance alone" -- which
is why trophies and medals are in (a Vince Lombardi trophy is identifiable by shape, the same
argument as a Porsche's bodywork) and why events are out.

One consequence to keep in view when reading the tables: on sports footage this makes brand
coverage high almost by construction -- jerseys, sneakers, balls and signage are everywhere.
That is faithful to the definition rather than a flaw in it, but it does mean brand coverage
discriminates between detectors far less than person recall does.

Objects, venues and events
--------------------------
A vetoed term is not simply "not brand". Venues and events -- "auto showroom", "basketball
court", "car show" -- are not boxable objects at all, so they are routed to NONOBJECT rather
than OTHER. Leaving them in OTHER would inflate a detector's apparent breadth with words it
cannot localise, which is the failure the NONOBJECT bucket exists to prevent.

This is also why "olympics" would not belong under sports_equipment even though the Olympic
rings are one of the world's most recognisable marks: the rings are a mark, but the word
"olympics" names the event, and a box drawn around an event is the whole frame. (It is moot
here in any case -- no olympic term exists in either vocabulary.)

Every needle is checked against the vocabularies
------------------------------------------------
The lists below were audited against eval/vocab_dump.json (yoloe-pf's 4585 terms plus
COCO-80) and the schema's own prompt terms, and 49 dead needles that could never match
anything were removed. The test is not "is the needle a vocabulary entry" but "does it occur
as a word in some entry", since "shoe" legitimately matches "leather shoe".

Three were instructive. "sneaker" and "jersey" are in neither vocabulary -- what actually
fires is "leather shoe" and "sportswear". And "t-shirt" could never have matched at all,
because _matches splits hyphens before comparing, so a hyphenated needle is unreachable by
construction; "shirt" covers it.

Matching is whole-word
----------------------
Needles match whole words (or whole phrases when they contain a space), so "ball" does not
match "basketball" and "car" does not match "carpet". This matters: an earlier draft vetoed
"basketball" and "football" to stop a substring match that whole-word matching already
prevents, and in doing so suppressed real brand-bearing sports equipment. The same draft
vetoed "race", which killed "race car".

Usage
-----
    python eval/build_tag_map.py                    # writes eval/tag_map.json
    python eval/build_tag_map.py --review brand     # print a bucket for eyeballing
    python eval/build_tag_map.py --audit            # re-check every needle is reachable
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# "mark"    -> brand is the logo/wordmark/emblem itself (current schema)
# "bearing" -> brand is any object whose identity could be a brand (superseded)
BRAND_DEFINITION = os.environ.get("ELV_BRAND_DEFINITION", "mark")

import paths  # noqa: E402
from schema_brand_person import class_of_prompt  # noqa: E402

TAG_MAP_PATH = paths.TAG_MAP
_TAG_MAP_CACHE: Dict[str, Dict[str, str]] = {}


def load_tag_map(path: str = TAG_MAP_PATH) -> Dict[str, str]:
    """tag -> 'brand' | 'person' | 'OTHER' | 'NONOBJECT'.

    EVALUATION SCAFFOLDING, not part of the tagger. It exists only so a prompt-free run,
    which answers from its own 4585-term vocabulary, can be scored against brand/person
    ground truth at all. The production pipeline never needs it: in the default and
    text-target modes the detector can only answer with terms it was given, and in the
    general mode the native label is metadata rather than the retrieval key -- a downstream
    query is matched against the crop embedding, not the label string.

    Box-level scoring supersedes this with class-agnostic IoU coverage, which is both simpler
    and more faithful to what the pipeline actually does with a label (nothing).
    """
    if path not in _TAG_MAP_CACHE:
        if not os.path.exists(path):
            return {}
        with open(path) as handle:
            _TAG_MAP_CACHE[path] = {_norm(k): v for k, v in json.load(handle)["map"].items()}
    return _TAG_MAP_CACHE[path]


def classify(tag: str, mode: str = "label", tag_map: Dict[str, str] = None) -> str:
    """mode='label' uses the prompt list; mode='coverage' uses the tag map."""
    key = _norm(tag)
    if mode == "label":
        return class_of_prompt(key)
    return (tag_map if tag_map is not None else load_tag_map()).get(key, "OTHER")


# --------------------------------------------------------------------------------------
# brand: explicit category families
# --------------------------------------------------------------------------------------
# Matched as whole words against the tag, so "leather shoe", "cowboy boot" and "sports car"
# are all caught by their head noun without listing every compound the 4585-term vocabulary
# happens to contain.
BRAND_FAMILIES: Dict[str, List[str]] = {
    # The mark itself, and the surfaces marks are carried on. Flags, pennants, trophies and
    # medals belong here: they are identifiable by shape and livery, which is the point of
    # the reframing.
    # The mark ITSELF: a graphic identity of an organisation or product, and the only family
    # that counts as brand under the mark definition. Deliberately narrow -- a thing you could
    # crop and match against a logo pool.
    #
    # `label` is here because a bottle label is coextensive with its branding, and because it
    # is one of the schema's six prompts. `vector icon` and `app icon` are here because most
    # modern logos ARE flat vector symbols; the terms also cover generic UI glyphs, but on
    # this footage a detected vector symbol is far more likely a team or sponsor mark.
    # (Immaterial either way: neither term fires once in these runs.)
    "mark": [
        "logo", "emblem", "badge", "brand", "crest", "mascot", "label",
        "app icon", "vector icon", "watermark",
    ],
    # Surfaces that CARRY marks. Their own family, and NOT brand: a banner is a thing a logo
    # sits on, so a box on it is a box on the carrier -- the same overshadowing failure that
    # made the 101-term object list unusable, one level up. These were tested as prompts and
    # rejected for exactly that reason (mark share fell to 4% on world-text, whose apparent
    # recall gain was 20 banners and a single car logo).
    #
    # `sticker` and `business card` are carriers too: a design is printed ON them, and plenty
    # of stickers carry no brand at all.
    "mark_surface": [
        "sign", "signage", "billboard", "banner", "poster", "advertisement",
        "flag", "pennant", "trophy", "medal", "podium", "scoreboard",
        "sticker", "business card",
    ],
    "automotive": [
        "car", "automobile", "sedan", "suv", "truck", "van", "bus", "taxi", "jeep",
        "motorcycle", "scooter", "moped", "bicycle", "bike", "tractor", "vehicle",
        "tire", "brake light", "headlight", "windshield",
    ],
    "fashion": [
        "shirt", "blouse", "jacket", "coat", "hoodie", "sweater",
        "dress", "skirt", "pants", "jeans", "shorts", "suit", "uniform",
        "sportswear", "scarf", "glove", "belt", "tie", "hat", "cap",
        "sock", "garment", "clothing", "vest", "robe",
    ],
    "footwear": [
        "shoe", "boot", "sandal", "heel", "footwear", "trainer", "loafer",
    ],
    "bags": [
        "handbag", "backpack", "bag", "luggage", "suitcase", "briefcase",
        "duffel", "tote",
    ],
    "eyewear_jewellery": [
        "sunglasses", "glasses", "goggles", "necklace", "bracelet",
        "ring", "earring", "jewellery", "jewelry", "pendant", "brooch", "watch",
        "armband", "neckband", "sweatband",
    ],
    "electronics": [
        "laptop", "computer", "smartphone", "phone", "tablet", "camera", "television",
        "monitor", "speaker", "earphone", "keyboard",
        "controller", "drone", "printer", "microphone",
    ],
    "food_beverage": [
        "bottle", "can", "cup", "mug", "glass", "champagne flute", "carton", "package", "box",
        "tin", "jar", "snack", "cake", "pizza", "sandwich",
        "chocolate", "candy", "cereal", "soda", "beer", "wine", "coffee",
    ],
    "beauty_health": [
        "lipstick", "perfume", "cosmetic", "makeup", "shampoo", "cream",
        "medicine", "toothbrush", "toothpaste", "soap", "razor",
    ],
    # Sports equipment is brand-bearing in exactly the way the reframing means: a Spalding
    # basketball or an NFL football is identifiable from its appearance alone, and this
    # footage is NBA and NFL. Whole-word matching is what makes listing the specific balls
    # safe -- "basketball" is its own token and never collides with the generic "ball".
    "sports_equipment": [
        "ball", "basketball", "football", "baseball", "volleyball", "handball",
        "rugby ball", "racket", "bat", "club", "skateboard", "surfboard", "ski",
        "snowboard", "skate", "dumbbell", "helmet",
    ],
    # Instruments are brand-bearing in the strongest form of the argument: a Les Paul is
    # identifiable by its silhouette alone, exactly like a Porsche's bodywork. "organ" and
    # "bass" are given as phrases where the bare word would collide ("organization",
    # "seabass"); the -ist and -er players are people and are forced to person above.
    "musical_instruments": [
        "guitar", "piano", "drum", "violin", "cello", "flute", "clarinet",
        "trombone", "trumpet", "ukulele", "banjo", "accordion", "tambourine",
        "guzheng", "harpsichord", "bass guitar",
        "mouth organ", "pipe organ",
    ],
    # An Eames chair or an Aeron is recognised by shape without reading a label.
    "furniture": [
        "chair", "armchair", "couch", "table", "desk", "lamp", "shelf",
        "bookcase", "bookshelf", "cabinet", "stool", "bench", "mattress", "mirror",
        "clock", "carpet", "curtain", "vase",
    ],
    # "fan" is deliberately only ever a phrase: bare "fan" is a supporter, not an appliance.
    "appliances": [
        "refrigerator", "fridge", "oven", "microwave", "toaster", "blender",
        "washing machine", "vacuum", "heater", "air conditioner",
        "stove", "coffee machine", "ceiling fan", "floor fan", "mechanical fan",
    ],
    "tools": [
        "drill", "hammer", "wrench", "saw", "screwdriver", "toolbox", "ladder",
        "chain saw",
    ],
    # Art was named explicitly in the reframing, alongside logos and car models.
    "art": [
        "painting", "sculpture", "mural", "print", "picture frame",
    ],
    "childcare_pet": [
        "baby carriage", "car seat", "dog collar", "leash",
    ],
}

# Venues and premises. Not brand-bearing, and not boxable objects either, so they land in
# NONOBJECT rather than OTHER.
VENUE_VETO: List[str] = [
    "dealership", "showroom", "shop", "store", "market", "mall", "boutique",
    "factory", "workshop", "garage", "parking", "building", "room", "hall",
    "court", "field", "stadium", "arena", "rink", "track", "pitch", "gym",
    "street", "road", "highway", "lot", "museum", "gallery",
    "theater", "theatre", "nightclub", "cinema", "restaurant", "cafe", "bar",
]

# Events and abstractions. Same treatment: not objects, so NONOBJECT.
EVENT_VETO: List[str] = [
    "auto show", "car show", "fashion show",
    "game", "match", "tournament", "season", "championship",
    "fair", "exhibition", "party", "festival", "parade", "ceremony", "reception",
    "concert", "performance", "conference", "meeting", "wedding", "playing",
]

# Activities, sports and occasions. Not objects: you cannot draw a box around "yoga" or
# "triathlon", and the mining files several of them under `person` because people are what the
# word evokes. `painting` and `drawing` are deliberately absent -- those name artefacts as well
# as activities, and the artefact is boxable.
ACTIVITY_VETO: List[str] = [
    "tae kwon do", "tai chi", "karate", "yoga", "aerobics",
    "boxing", "archery", "triathlon",
    "marathon", "swimming", "diving", "skiing", 
    "cycling", "sailing", "climbing", "hiking", 
    "running", "cooking", "reading", "writing", "shopping",
    "fishing", "hunting", "camping", "picnic", "protest", "graduation", "birthday",
    "christmas", "halloween", "easter", "thanksgiving", "carnival", "rodeo",
    "circus", "safari", "vacation", 
]

# Collectives (a box around many things, not one branded object) and compounds that merely
# contain a family word. These are real enough to stay in OTHER.
OTHER_VETO: List[str] = [
    "team", "squad", "fleet", "collection",
    "meatball", "snowball", "eyebrow", "ballroom",
]

# person terms the margin gets wrong: body parts and attributes that are not people, and
# collective nouns that box many people at once.
PERSON_VETO: List[str] = [
    "forehead", "manicure", "hairstyle", "haircut", "hair", "brunette",
    "neckline", "mane", "fang", "arm", "leg", "hand", "foot", "shoulder",
    "crowd", "crowded", "team", "squad", "group", "family", "couple", "infantry",
    "parliament", "government", "management", "staff", "band", "orchestra", "choir",
    # abstractions the mining files under person because people are what they evoke
    "traffic", "arrest", "applause", "seminar", "perform", "professional", "profile",
    "mannequin", "snowman", "strawman", "statue", "doll", "figurine", "ghost",
    "photo", "portrait session", "publicity portrait", "documentary",
]

# person terms worth forcing in regardless of what the margin said.
PERSON_FORCE: List[str] = [
    "person", "man", "woman", "boy", "girl", "child", "baby", "toddler",
    "teenager", "adult", "player", "athlete",
    "guitarist", "drummer", "bassist", "violinist", "pianist", 
    # "speaker" is deliberately absent. It is genuinely ambiguous -- a person addressing a
    # room, or a loudspeaker -- and the two scoring modes resolve it differently on purpose.
    # In label mode it came from the person.role prompt tier, so it scores as person. In
    # coverage mode it is mostly the prompt-free vocabularies talking, where it means the
    # audio device, so the electronics family claims it.
    "referee", "spectator", "host", "anchor", "dancer",
    "singer", "musician", "coach", "commentator", "model", "fashion model",
    "supermodel", "participant", "passenger", "pedestrian", "shopper", "customer",
    "worker", "officer", "soldier", "nurse", "doctor", "chef", "waiter", "driver",
    "pilot", "student", "teacher", "journalist", "actor",
    "artist", "performer", "bride", "groom", "monk", "cheerlead", "swimmer",
    "runner", "skier", "surfer", "boxer", "wrestler", "golfer",
    "pitcher", "catcher", "goalkeeper", "guard", "lifeguard", "fireman",
    "businessman", "business woman", "motorcyclist", "face", "head",
]

ALL_LISTS = {
    **{f"BRAND_FAMILIES.{k}": v for k, v in BRAND_FAMILIES.items()},
    "VENUE_VETO": VENUE_VETO, "EVENT_VETO": EVENT_VETO, "OTHER_VETO": OTHER_VETO,
    "ACTIVITY_VETO": ACTIVITY_VETO,
    "PERSON_VETO": PERSON_VETO, "PERSON_FORCE": PERSON_FORCE,
}


def _norm(tag: str) -> str:
    return " ".join(str(tag).lower().split())


def _words(tag: str) -> Set[str]:
    return set(_norm(tag).replace("-", " ").split())


def _matches(tag: str, needles: List[str]) -> bool:
    """Whole-word containment; needles containing a space match as a phrase."""
    words = _words(tag)
    text = _norm(tag)
    for needle in needles:
        if " " in needle:
            if needle in text:
                return True
        elif needle in words:
            return True
    return False


def brand_family(tag: str) -> str:
    for family, needles in BRAND_FAMILIES.items():
        if _matches(tag, needles):
            return family
    return ""


def bucket_for(tag: str, mined: Dict[str, str]) -> Tuple[str, str]:
    """Return (bucket, brand_family_or_empty)."""
    # Every explicit rule is consulted before the mined heuristic, and the mined `person`
    # bucket is the LAST thing tried. An earlier ordering asked it first, which meant
    # anything the mining associated with people short-circuited: "rock concert",
    # "nightclub", "marching band" and "meeting" all came back as person, and "guzheng" was
    # person rather than a musical instrument. Ordering the explicit rules first fixes that
    # class of error structurally, instead of needing a veto entry per mistake.
    #
    # PERSON_FORCE leads because "basketball player" is a person and must not be captured by
    # the sports_equipment family via "basketball".
    # Venues and events are not objects, whichever family word they happen to contain --
    # including a person word, which is why this is tested alongside PERSON_FORCE rather
    # than after it. "pedestrian street" is a street, not a pedestrian.
    is_place = (_matches(tag, VENUE_VETO) or _matches(tag, EVENT_VETO)
                or _matches(tag, ACTIVITY_VETO))
    if not is_place and _matches(tag, PERSON_FORCE) and not _matches(tag, PERSON_VETO):
        return "person", ""
    if is_place:
        return "NONOBJECT", ""
    if _matches(tag, OTHER_VETO):
        return "OTHER", ""
    family = brand_family(tag)
    if family:
        # Under the mark definition, `brand` is the MARK ITSELF -- the logo, wordmark, emblem
        # or badge -- not the object carrying it. The other 15 families name brand-BEARING
        # objects, which was the earlier definition; they are real objects, so they stay in
        # OTHER rather than being discarded.
        if BRAND_DEFINITION == "mark" and family != "mark":
            return "OTHER", family
        return "brand", family
    if mined.get(tag) == "person" and not _matches(tag, PERSON_VETO):
        return "person", ""
    return ("NONOBJECT" if mined.get(tag) == "NONOBJECT" else "OTHER"), ""


def load_corpus(vocab_dump: str, extra: List[str] = ()) -> List[str]:
    with open(vocab_dump) as handle:
        dump = json.load(handle)
    corpus = []
    for key in ("yoloe_pf", "coco80"):
        for term in dump.get(key, []):
            corpus.append(_norm(term))
    corpus.extend(_norm(t) for t in extra)
    return corpus


def audit(corpus: List[str]) -> int:
    """Report needles that cannot match anything in the vocabularies."""
    words: Set[str] = set()
    for term in corpus:
        words |= _words(term)
    def reachable(needle: str) -> bool:
        if " " in needle:
            return any(needle in term for term in corpus)
        return needle in words

    dead_total = 0
    for name, needles in ALL_LISTS.items():
        dead = [n for n in needles if not reachable(n)]
        dead_total += len(dead)
        if dead:
            print(f"  {name:26} {len(dead):2}/{len(needles):2} dead: {', '.join(dead)}")
    print(f"{dead_total} dead needles of {sum(len(v) for v in ALL_LISTS.values())}")
    return dead_total


def emitted_tags(runs_glob: str) -> Dict[str, int]:
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
                tag = _norm(data.get("tag", ""))
                if tag:
                    seen[tag] = seen.get(tag, 0) + 1
    return seen


def main() -> int:
    global BRAND_DEFINITION
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mined", default=os.path.join(paths.TOOLS, "tag_map_narrow.json"),
                        help="mining proposal; only its `person` and `NONOBJECT` buckets are "
                             "used (brand's margin was too low to trust)")
    parser.add_argument("--vocab-dump", default=paths.VOCAB_DUMP)
    parser.add_argument("--runs", default=os.path.join(paths.EXPERIMENTS, "*", "runs*", "*", "out.jsonl"))
    parser.add_argument("--out", default=paths.TAG_MAP)
    parser.add_argument("--audit", action="store_true",
                        help="check every needle is reachable in the vocabularies, then exit")
    parser.add_argument("--brand-definition", default=BRAND_DEFINITION,
                        choices=["mark", "bearing"],
                        help="mark: only the logo/emblem itself counts as brand (default, "
                             "matches the schema). bearing: any brand-carrying object, the "
                             "superseded definition.")
    parser.add_argument("--review", default=None,
                        choices=["brand", "person", "OTHER", "NONOBJECT"])
    args = parser.parse_args()
    BRAND_DEFINITION = args.brand_definition

    counts = emitted_tags(args.runs)
    corpus = load_corpus(args.vocab_dump, extra=list(counts))

    if args.audit:
        return 1 if audit(corpus) else 0

    terms: List[str] = []
    for term in corpus:
        if term not in terms:
            terms.append(term)

    mined: Dict[str, str] = {}
    if os.path.exists(args.mined):
        with open(args.mined) as handle:
            mined = {_norm(k): v for k, v in json.load(handle)["map"].items()}

    mapping: Dict[str, str] = {}
    families: Dict[str, str] = {}
    for term in terms:
        bucket, family = bucket_for(term, mined)
        mapping[term] = bucket
        if family:
            # Recorded even when the bucket is OTHER: a reviewer needs to see that "banner"
            # landed in OTHER *because* it is a mark_surface, not by accident.
            families[term] = family

    bucket_counts = {b: sum(1 for v in mapping.values() if v == b)
                     for b in ("brand", "person", "OTHER", "NONOBJECT")}
    fired = {b: sum(1 for t, v in mapping.items() if v == b and counts.get(t))
             for b in bucket_counts}
    family_counts: Dict[str, int] = {}
    for term, family in families.items():
        key = family if mapping[term] == "brand" else f"{family} (->OTHER)"
        family_counts[key] = family_counts.get(key, 0) + 1
    print(f"{len(terms)} terms -> {bucket_counts}")
    print(f"of which actually fired on this footage: {fired}")
    print(f"brand families: {family_counts}")

    with open(args.out, "w") as handle:
        json.dump({
            "note": "brand = BRAND-BEARING (an object whose identity could be a brand), built "
                    "from explicit category families in build_tag_map.py. person is taken "
                    "from the SigLIP 2 mining proposal plus force/veto lists. Venues and "
                    "events are NONOBJECT, not OTHER.",
            "counts": bucket_counts,
            "fired_on_footage": fired,
            "brand_family_counts": family_counts,
            "brand_families": families,
            "map": mapping,
        }, handle, indent=1)
    print(f"wrote {args.out}")

    if args.review:
        rows = sorted((t for t in terms if mapping[t] == args.review),
                      key=lambda t: (-counts.get(t, 0), t))
        shown = [t for t in rows if counts.get(t)]
        print(f"\n=== {args.review}: {len(rows)} terms, {len(shown)} fired on this footage ===")
        for term in shown:
            extra = f"  [{families[term]}]" if term in families else ""
            print(f"   {counts.get(term, 0):5}  {term}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
