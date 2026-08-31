#!/usr/bin/env python3
"""Run detector backends over the frozen frame set and emit tagger-format JSONL.

Runs inside the general_detection container (needs ultralytics + torch).

Two deliberate differences from the production tagger
----------------------------------------------------
1. **No embedding.** Comparing detectors needs boxes, scores and labels — not vectors. Running
   SigLIP 2 over every candidate crop would dominate the cost and change nothing about the
   ranking. Whichever detector wins gets wired into general_detection/detector.py afterwards.

2. **Ungated.** Every backend runs at conf~0 with a high detection cap, and each detection's
   score is written out. Detector scores are *not* comparable across models — YOLOE's
   text-similarity scores, the prompt-free head's scores and YOLO11's sigmoid class scores have
   different scales and calibration — so fixing one threshold for all of them would rank the
   models by score inflation. The scorer instead sweeps thresholds per detector and reports
   each at its own best operating point.

Scope: the backends here are all *general* detectors, which is what the presence labels can
rank. Visual-prompted YOLOE is deliberately not here — it detects only what you give exemplars
for, so it is a per-class enhancement rather than a general-detector candidate, and it needs
exemplars drawn from OUTSIDE the frozen frames (cropping them from the eval set would be
leakage). It lands with Grounding DINO and OWLv2, which also need their own code paths.

Usage
-----
    python eval/run_detectors.py --backends yoloe11-text yoloe11-pf yolo11
    python eval/run_detectors.py --list
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
# Ultralytics resolves bare checkpoint names relative to the CWD; keep them in the mounted cache.
WEIGHTS_DIR = os.environ.get("ELV_WEIGHTS_DIR", "/root/.cache/detection")

# name -> (family, checkpoint). `family` selects how prompts are applied.
BACKENDS: Dict[str, Dict] = {
    # YOLOE, text-prompted with the schema's vocabulary terms
    "yoloe11-text": {"family": "yoloe-text", "weights": "yoloe-11l-seg.pt"},
    "yoloe26-text": {"family": "yoloe-text", "weights": "yoloe-26l-seg.pt"},
    # YOLOE prompt-free: built-in 4585-class vocabulary, no set_classes call
    "yoloe11-pf": {"family": "yoloe-pf", "weights": "yoloe-11l-seg-pf.pt"},
    "yoloe26-pf": {"family": "yoloe-pf", "weights": "yoloe-26l-seg-pf.pt"},
    # YOLO-World v2: set_classes takes the names directly (no separate get_text_pe step)
    "world-text": {"family": "world-text", "weights": "yolov8l-worldv2.pt"},
    # Closed-vocabulary COCO-80 baseline. Can only attempt classes COCO contains
    # (person, car, truck, bus, bottle, cup, chair, tv) — `logo` is structurally impossible.
    "yolo11": {"family": "closed", "weights": "yolo11l.pt"},
}


def load_manifest():
    with open(os.path.join(HERE, "frames.json")) as handle:
        data = json.load(handle)
    return data["frames"], data["classes"]


def prompt_list(classes: List[Dict]) -> List[str]:
    """Flat, de-duplicated list of every vocabulary term in the schema."""
    seen: List[str] = []
    for spec in classes:
        for term in spec["prompts"]:
            if term not in seen:
                seen.append(term)
    return seen


def build(backend: str, prompts: List[str]):
    """Load a backend with CWD in the weights cache, so bare asset names download there
    rather than into the container's ephemeral WORKDIR."""
    spec = BACKENDS[backend]
    family, weights = spec["family"], spec["weights"]

    previous = os.getcwd()
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.chdir(WEIGHTS_DIR)
    try:
        if family in ("yoloe-text", "yoloe-pf"):
            from ultralytics import YOLOE
            model = YOLOE(weights)
            if family == "yoloe-text":
                # get_text_pe may pull the MobileCLIP text encoder on first use.
                model.set_classes(prompts, model.get_text_pe(prompts))
        elif family == "world-text":
            from ultralytics import YOLOWorld
            model = YOLOWorld(weights)
            model.set_classes(prompts)      # no separate text-PE step in this API
        elif family == "closed":
            from ultralytics import YOLO
            model = YOLO(weights)
        else:
            raise ValueError(f"unknown family {family}")
    finally:
        os.chdir(previous)
    return model


def run_backend(
    backend: str, frames: List[Dict], prompts: List[str], out_dir: str,
    imgsz: int, conf: float, max_det: int, batch: int,
) -> Dict:
    model = build(backend, prompts)
    names = model.model.names
    if not isinstance(names, dict):
        names = {i: n for i, n in enumerate(names)}

    path = os.path.join(out_dir, backend, "out.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    started = time.time()
    written = 0
    with open(path, "w") as handle:
        for start in range(0, len(frames), batch):
            chunk = frames[start : start + batch]
            sources = [os.path.join(HERE, f["frame"]) for f in chunk]
            results = model.predict(
                sources, imgsz=imgsz, conf=conf, iou=0.7, max_det=max_det, verbose=False
            )
            for frame, result in zip(chunk, results):
                height, width = result.orig_shape
                boxes = result.boxes
                if boxes is None:
                    continue
                for xyxy, cls, score in zip(
                    boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()
                ):
                    tag = str(names.get(int(cls), int(cls)))
                    x1, y1, x2, y2 = xyxy
                    handle.write(json.dumps({"type": "tag", "data": {
                        "start_time": 0, "end_time": 0,
                        # basename maps back to the manifest frame id
                        "source_media": os.path.join("/elv/test", frame["id"] + ".png"),
                        "tag": tag, "track": "",
                        "frame_info": {"frame_idx": frame["frame_idx"], "box": {
                            "x1": round(max(0.0, x1 / width), 4),
                            "y1": round(max(0.0, y1 / height), 4),
                            "x2": round(min(1.0, x2 / width), 4),
                            "y2": round(min(1.0, y2 / height), 4)}},
                        "additional_info": {"kind": "crop", "prompt": tag,
                                            "score": round(float(score), 4),
                                            "detector": backend},
                    }}) + "\n")
                    written += 1
            print(f"    {min(start + batch, len(frames))}/{len(frames)} frames, "
                  f"{written} detections", end="\r", flush=True)

    elapsed = time.time() - started
    print(f"    {len(frames)} frames, {written} detections, {elapsed:.1f}s "
          f"({elapsed / len(frames) * 1000:.0f} ms/frame)      ")
    return {"backend": backend, "detections": written, "seconds": round(elapsed, 1),
            "vocab_size": len(names)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backends", nargs="*", default=list(BACKENDS),
                        help="which backends to run (default: all)")
    parser.add_argument("--list", action="store_true", help="list backends and exit")
    parser.add_argument("--out", default=os.path.join(HERE, "runs"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.005,
                        help="near-zero on purpose: the scorer picks each detector's own "
                             "operating point (default 0.005)")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    if args.list:
        for name, spec in BACKENDS.items():
            print(f"  {name:<14} {spec['family']:<12} {spec['weights']}")
        return 0

    frames, classes = load_manifest()
    prompts = prompt_list(classes)
    print(f"{len(frames)} frames, {len(prompts)} prompts across {len(classes)} classes")
    print(f"prompts: {', '.join(prompts)}\n")

    os.makedirs(args.out, exist_ok=True)
    summary = []
    for backend in args.backends:
        if backend not in BACKENDS:
            print(f"unknown backend {backend!r}; --list to see options", file=sys.stderr)
            return 2
        print(f"  {backend} ({BACKENDS[backend]['weights']})")
        try:
            summary.append(run_backend(backend, frames, prompts, args.out,
                                       args.imgsz, args.conf, args.max_det, args.batch))
        except Exception as exc:
            # One backend failing (a missing checkpoint, an API difference) must not lose the
            # runs that already succeeded.
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            summary.append({"backend": backend, "error": f"{type(exc).__name__}: {exc}"})

    with open(os.path.join(args.out, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nwrote {args.out}/<backend>/out.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
