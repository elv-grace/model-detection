#!/usr/bin/env python3
"""Grounding DINO and OWLv2 over the frozen frames (steps 6a / 6b).

Why these two
-------------
Both are Apache-2.0 (unlike ultralytics' AGPL-3.0) and both live in HF transformers, so either
would let the whole stack drop the AGPL dependency. Grounding DINO is the architectural
counter-proposal to YOLOE: a real BERT text encoder cross-attending to image features, rather
than a CLIP-style dot product against a prompt embedding, which is the mechanism that should
handle compositional phrases better.

An important limitation for "general object detection"
------------------------------------------------------
Neither has a prompt-free mode. YOLOE's -pf checkpoint carries a built-in 4585-class
vocabulary and will catalogue a frame with no input; these two are open-vocabulary but
*query-driven* — you must name what you want. Grounding DINO is additionally bounded by BERT's
context (phrases are concatenated into one string), so a 4585-term vocabulary is not merely
slow, it does not fit.

So "general" here means a fixed vocabulary of common concrete nouns, not open-ended
cataloguing. Two vocabularies are supported:

    schema   the 29 terms behind the 8 labelled classes — directly comparable to the
             ultralytics runs on macro F1
    general  those terms plus ~35 more common objects, to test breadth. Derived from what
             yoloe11-pf actually emitted on this footage (>=0.15 confidence), then filtered to
             concrete nouns — the raw list was full of scene and attribute terms ("darkness",
             "tournament", "brunette") that are not objects and cannot be boxed meaningfully.

Usage
-----
    python eval/run_hf_detectors.py --models gdino owlv2 --vocab schema
    python eval/run_hf_detectors.py --models gdino owlv2 --vocab general
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "gdino": "IDEA-Research/grounding-dino-base",
    "owlv2": "google/owlv2-base-patch16-ensemble",
}

# Concrete objects beyond the labelled schema. Grounded in what yoloe11-pf detected on this
# content, so the vocabulary reflects the footage rather than a generic COCO-ish guess.
EXTRA_GENERAL = [
    "hat", "chair", "cup", "glasses", "lamp", "bracelet", "necklace", "helmet",
    "handbag", "camera", "shoe", "glove", "laptop", "table", "speaker", "cake",
    "watch", "picture", "frame", "flag", "podium", "drum", "belt", "jacket",
    "street light", "bench", "desk", "microphone", "book", "phone", "bag",
    "bicycle", "tie", "suit", "shirt", "curtain", "window", "door", "bowl", "clock",
]


def load_manifest():
    with open(os.path.join(HERE, "frames.json")) as handle:
        data = json.load(handle)
    return data["frames"], data["classes"]


def vocabulary(classes: List[Dict], which: str) -> List[str]:
    terms: List[str] = []
    for spec in classes:
        for term in spec["prompts"]:
            if term not in terms:
                terms.append(term)
    if which == "general":
        for term in EXTRA_GENERAL:
            if term not in terms:
                terms.append(term)
    return terms


def write_row(handle, frame, tag, score, box, model_name, vocab_name) -> None:
    handle.write(json.dumps({"type": "tag", "data": {
        "start_time": 0, "end_time": 0,
        "source_media": os.path.join("/elv/test", frame["id"] + ".png"),
        "tag": tag, "track": "",
        "frame_info": {"frame_idx": frame["frame_idx"], "box": box},
        "additional_info": {"kind": "crop", "prompt": tag, "score": round(float(score), 4),
                            "detector": model_name, "vocab": vocab_name},
    }}) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["gdino", "owlv2"], choices=list(MODELS))
    parser.add_argument("--vocab", default="schema", choices=["schema", "general"])
    parser.add_argument("--conf", type=float, default=0.05,
                        help="these models score on a different scale from YOLO; the scorer "
                             "still sweeps thresholds per run (default 0.05)")
    parser.add_argument("--chunk", type=int, default=8,
                        help="Grounding DINO phrases per pass. It concatenates phrases into a\n"
                             "single BERT string and attributes each box to a token span; with\n"
                             "29 phrases 94%% of boxes came back with an EMPTY label and some\n"
                             "as raw wordpieces ('##board'). Fewer phrases per pass makes the\n"
                             "spans separable, at the cost of one pass per chunk.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--out", default=os.path.join(HERE, "runs"))
    args = parser.parse_args()

    import torch
    from PIL import Image

    frames, classes = load_manifest()
    terms = vocabulary(classes, args.vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{len(frames)} frames, {len(terms)} terms ({args.vocab} vocabulary), device={device}")

    for key in args.models:
        model_id = MODELS[key]
        name = f"{key}-{args.vocab}"
        print(f"\n  {name}  ({model_id})")
        started = time.time()
        written = 0
        path = os.path.join(args.out, name, "out.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        try:
            if key == "gdino":
                from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
                processor = AutoProcessor.from_pretrained(model_id)
                model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
                model.eval()
                # One lowercase string of phrases separated by " . ", in chunks (see --chunk).
                groups = [terms[i : i + args.chunk] for i in range(0, len(terms), args.chunk)]
                empty = 0
                with open(path, "w") as handle:
                    for start in range(0, len(frames), args.batch):
                        chunk = frames[start : start + args.batch]
                        images = [Image.open(os.path.join(HERE, f["frame"])).convert("RGB")
                                  for f in chunk]
                        for group in groups:
                            text = ". ".join(group) + "."
                            inputs = processor(images=images, text=[text] * len(images),
                                               return_tensors="pt", padding=True).to(device)
                            with torch.no_grad():
                                outputs = model(**inputs)
                            results = processor.post_process_grounded_object_detection(
                                outputs, inputs.input_ids, threshold=args.conf,
                                target_sizes=[img.size[::-1] for img in images],
                            )
                            for frame, image, result in zip(chunk, images, results):
                                width, height = image.size
                                labels = result.get("text_labels", result.get("labels", []))
                                for box, score, label in zip(result["boxes"], result["scores"],
                                                             labels):
                                    tag = str(label).strip()
                                    if not tag:
                                        # Unattributable span: dropping it is honest, counting
                                        # it as a detection of nothing is not.
                                        empty += 1
                                        continue
                                    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                                    write_row(handle, frame, tag, float(score), {
                                        "x1": round(max(0.0, x1 / width), 4),
                                        "y1": round(max(0.0, y1 / height), 4),
                                        "x2": round(min(1.0, x2 / width), 4),
                                        "y2": round(min(1.0, y2 / height), 4)}, name, args.vocab)
                                    written += 1
                        print(f"    {min(start + args.batch, len(frames))}/{len(frames)}, "
                              f"{written} dets ({empty} unlabelled)", end="\r", flush=True)
            else:
                from transformers import Owlv2ForObjectDetection, Owlv2Processor
                processor = Owlv2Processor.from_pretrained(model_id)
                model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device)
                model.eval()
                with open(path, "w") as handle:
                    for start in range(0, len(frames), args.batch):
                        chunk = frames[start : start + args.batch]
                        images = [Image.open(os.path.join(HERE, f["frame"])).convert("RGB")
                                  for f in chunk]
                        inputs = processor(text=[terms] * len(images), images=images,
                                           return_tensors="pt").to(device)
                        with torch.no_grad():
                            outputs = model(**inputs)
                        results = processor.post_process_grounded_object_detection(
                            outputs=outputs, threshold=args.conf,
                            target_sizes=torch.tensor([img.size[::-1] for img in images]).to(device),
                        )
                        for frame, image, result in zip(chunk, images, results):
                            width, height = image.size
                            for box, score, label in zip(result["boxes"], result["scores"],
                                                         result["labels"]):
                                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                                write_row(handle, frame, terms[int(label)], float(score), {
                                    "x1": round(max(0.0, x1 / width), 4),
                                    "y1": round(max(0.0, y1 / height), 4),
                                    "x2": round(min(1.0, x2 / width), 4),
                                    "y2": round(min(1.0, y2 / height), 4)}, name, args.vocab)
                                written += 1
                        print(f"    {min(start + args.batch, len(frames))}/{len(frames)}, "
                              f"{written} dets", end="\r", flush=True)
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            continue

        elapsed = time.time() - started
        print(f"    {len(frames)} frames, {written} detections, {elapsed:.0f}s "
              f"({elapsed / len(frames) * 1000:.0f} ms/frame)      ")

    return 0


if __name__ == "__main__":
    sys.exit(main())
