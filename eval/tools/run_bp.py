#!/usr/bin/env python3
"""Run every detector backend over the frozen frames under the brand/person schema.

Writes to <--out>/<backend>/out.jsonl, defaulting to the current experiment's runs/ directory.
Each experiment gets its own output tree and nothing is ever overwritten in place, so the
earlier sweeps remain scoreable against the same frames.

Prompts come from the schema, not from here. This script has no notion of brand or person --
it takes a prompt list, runs it through eight backends and writes boxes. That is why changing
the brand definition from 101 concrete nouns to six mark terms required no change to this
file: the definition lives in schema_brand_person.py, and this is the harness around it.

Ungated on purpose
------------------
Every backend runs at conf~0 with a high detection cap and each detection's score is written
out. Detector scores are NOT comparable across models -- YOLOE's text-similarity scores, the
prompt-free head's scores, YOLO11's sigmoid class scores and OWLv2's and Grounding DINO's
query scores have different scales and calibration -- so fixing one threshold for all of them
would rank the models by score inflation. The scorer sweeps thresholds per detector and
reports each at its own best operating point.

The Grounding DINO fix
----------------------
The 8-class sweep recorded Grounding DINO at macro F1 0.27 with 94% of its detections
carrying an EMPTY label, and blamed prompt-batching. That was wrong on both counts. The cause
is visible in post_process_grounded_object_detection:

    keep = scores > threshold                                    # we passed 0.05
    label_ids = get_phrases_from_posmap(prob > text_threshold, ...)   # DEFAULTED to 0.25

Any box scoring between 0.05 and 0.25 keeps the box but produces an empty token span, hence
an empty label. Box threshold five times lower than the text threshold: 94% empty, exactly.
Chunking the prompts could never have fixed it.

The second defect is that labels are always decoded from a BERT token span -- the processor's
own docstring says the `text_labels` argument is "NOT used" -- which is where "##board" (a
raw wordpiece) and "suv police car" (one span crossing two phrases) came from.

Both are fixed here. text_threshold is bound to the box threshold, and attribution is
computed directly rather than decoded: each phrase's character span in the prompt string is
mapped to token positions via the fast tokenizer's offset mapping, and a box is assigned the
phrase whose token positions carry its highest probability. That yields exactly one clean
phrase label per box, which is what every other backend produces.

Latency and throughput are both measured
----------------------------------------
They answer different questions and diverge by a lot on these models. Throughput is measured
batched, which is how a tagger actually consumes video. Latency is measured at batch size 1
with the CUDA queue synchronised, which is what a single-image request would pay. Warm-up
frames are excluded from both: first-call cost includes autograd graph setup and cuDNN
algorithm selection and is not representative.

Usage
-----
    python eval/tools/run_bp.py --backends yoloe11-text owlv2 gdino
    python eval/tools/run_bp.py --list
    python eval/tools/run_bp.py --prompts logo person --out eval/experiments/NN_name/runs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paths  # noqa: E402
from schema_brand_person import DEFAULT_PROMPTS  # noqa: E402

WEIGHTS_DIR = os.environ.get("ELV_WEIGHTS_DIR", "/root/.cache/detection")

BACKENDS: Dict[str, Dict] = {
    "yoloe11-text": {"family": "yoloe-text", "weights": "yoloe-11l-seg.pt"},
    "yoloe26-text": {"family": "yoloe-text", "weights": "yoloe-26l-seg.pt"},
    "yoloe11-pf":   {"family": "yoloe-pf",   "weights": "yoloe-11l-seg-pf.pt"},
    "yoloe26-pf":   {"family": "yoloe-pf",   "weights": "yoloe-26l-seg-pf.pt"},
    "world-text":   {"family": "world-text", "weights": "yolov8l-worldv2.pt"},
    # Closed-vocabulary COCO-80 baseline. It can only attempt classes COCO contains, so it is
    # structurally unable to name a brand mark; it is here to bound what a fixed general
    # detector gives you for free under the coverage metric.
    "yolo11":       {"family": "closed",     "weights": "yolo11l.pt"},
    "owlv2":        {"family": "hf-owlv2",   "weights": "google/owlv2-base-patch16-ensemble"},
    "gdino":        {"family": "hf-gdino",   "weights": "IDEA-Research/grounding-dino-base"},
    # Swin-T backbone instead of base's Swin-B. Not a distilled model -- a smaller one -- and
    # the question it answers is whether the coverage that makes `coverage` mode worth its cost
    # survives the smaller backbone.
    "gdino-tiny":   {"family": "hf-gdino",   "weights": "IDEA-Research/grounding-dino-tiny"},
}


def load_frames() -> List[Dict]:
    with open(paths.FRAMES_JSON) as handle:
        return json.load(handle)["frames"]


def row(frame, tag, score, box, backend) -> str:
    return json.dumps({"type": "tag", "data": {
        "start_time": 0, "end_time": 0,
        "source_media": os.path.join("/elv/test", frame["id"] + ".png"),
        "tag": tag, "track": "",
        "frame_info": {"frame_idx": frame["frame_idx"], "box": box},
        "additional_info": {"kind": "crop", "prompt": tag,
                            "score": round(float(score), 4), "detector": backend},
    }}) + "\n"


def norm_box(x1, y1, x2, y2, width, height) -> Dict[str, float]:
    return {"x1": round(max(0.0, x1 / width), 4), "y1": round(max(0.0, y1 / height), 4),
            "x2": round(min(1.0, x2 / width), 4), "y2": round(min(1.0, y2 / height), 4)}


# --------------------------------------------------------------------------------------
# Grounding DINO phrase attribution
# --------------------------------------------------------------------------------------
def phrase_token_spans(tokenizer, prompts: List[str]):
    """Return (text, [(lo, hi) token index range per prompt]).

    Computed from the fast tokenizer's character offsets rather than by decoding, so a box is
    attributed to a whole prompt phrase and never to a wordpiece fragment or to a span that
    straddles two phrases.
    """
    text = ". ".join(prompts) + "."
    spans_chars, cursor = [], 0
    for phrase in prompts:
        start = text.index(phrase, cursor)
        spans_chars.append((start, start + len(phrase)))
        cursor = start + len(phrase)

    encoded = tokenizer(text, return_offsets_mapping=True, return_tensors="pt",
                        truncation=True, max_length=512)
    offsets = encoded["offset_mapping"][0].tolist()
    spans = []
    for start, end in spans_chars:
        idx = [i for i, (a, b) in enumerate(offsets)
               if b > a and a >= start and b <= end]
        spans.append((min(idx), max(idx) + 1) if idx else (0, 0))
    return text, spans, encoded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backends", nargs="*", default=list(BACKENDS))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--prompts", nargs="*", default=None,
                        help="run this prompt list instead of the schema default. This is how "
                             "every prompt ablation is done -- comparing bare words (--prompts "
                             "brand person), the six brand mark terms, brand marks plus "
                             "surfaces, brand marks plus 'symbol'. Only the five promptable "
                             "backends respond to what is passed here; yoloe*-pf and yolo11 "
                             "answer from fixed vocabularies and emit byte-identical output "
                             "whatever the prompt list is.")
    parser.add_argument("--suffix", default="", help="appended to each output directory name")
    parser.add_argument("--out", default=os.path.join(paths.CURRENT, "runs"))
    parser.add_argument("--imgsz", type=int, default=640,
                        help="input size for the ultralytics backends (yoloe*, world, yolo11)")
    parser.add_argument("--hf-imgsz", type=int, default=0,
                        help="override the input resolution of the HF backends (owlv2, gdino), "
                             "which otherwise use their processors' own defaults -- gdino "
                             "shortest_edge 800 / longest_edge 1333, owlv2 a fixed 960 square. "
                             "Without this a resolution sweep would silently move only the "
                             "ultralytics models and compare them against fixed HF baselines. "
                             "For gdino the value sets shortest_edge with longest_edge scaled "
                             "by the same 1333/800 ratio; for owlv2 it sets the square side.")
    parser.add_argument("--conf", type=float, default=0.005,
                        help="near-zero on purpose; the scorer picks each detector's own "
                             "operating point")
    parser.add_argument("--hf-conf", type=float, default=0.05,
                        help="OWLv2 and Grounding DINO score on a different scale and produce "
                             "far more boxes at 0.005; 0.05 keeps the files tractable while "
                             "still sitting well below any sensible operating point")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8,
                        help="frames per forward pass. The HF backends scale their activation "
                             "memory with the square of the input size, so a resolution sweep "
                             "needs this lowered: gdino OOMs on a 24 GB card at batch 8 above "
                             "about 1100px.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--latency-frames", type=int, default=20,
                        help="frames timed one at a time for the latency figure")
    args = parser.parse_args()

    if args.list:
        for name, spec in BACKENDS.items():
            print(f"  {name:14} {spec['family']:12} {spec['weights']}")
        return 0

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    frames = load_frames()
    source = "explicit list" if args.prompts else "schema default"
    print(f"{len(frames)} frames, {len(prompts)} prompts from {source}: "
          f"{', '.join(prompts[:8])}{' ...' if len(prompts) > 8 else ''}\n")

    import torch
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    summary = []

    for backend in args.backends:
        if backend not in BACKENDS:
            print(f"unknown backend {backend!r}", file=sys.stderr)
            return 2
        spec = BACKENDS[backend]
        family, weights = spec["family"], spec["weights"]
        out_name = backend + args.suffix
        print(f"  {out_name} ({weights})")
        path = os.path.join(args.out, out_name, "out.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        written = 0

        try:
            # ---------------- build ----------------
            if family in ("yoloe-text", "yoloe-pf", "world-text", "closed"):
                previous = os.getcwd()
                os.makedirs(WEIGHTS_DIR, exist_ok=True)
                os.chdir(WEIGHTS_DIR)   # ultralytics resolves bare names against the CWD
                try:
                    if family in ("yoloe-text", "yoloe-pf"):
                        from ultralytics import YOLOE
                        model = YOLOE(weights)
                        if family == "yoloe-text":
                            model.set_classes(prompts, model.get_text_pe(prompts))
                    elif family == "world-text":
                        from ultralytics import YOLOWorld
                        model = YOLOWorld(weights)
                        model.set_classes(prompts)
                    else:
                        from ultralytics import YOLO
                        model = YOLO(weights)
                finally:
                    os.chdir(previous)
                names = model.model.names
                if not isinstance(names, dict):
                    names = {i: n for i, n in enumerate(names)}

                def infer(images_paths):
                    return model.predict(images_paths, imgsz=args.imgsz, conf=args.conf,
                                         iou=0.7, max_det=args.max_det, verbose=False)

                def emit(handle, chunk, results):
                    count = 0
                    for frame, result in zip(chunk, results):
                        height, width = result.orig_shape
                        if result.boxes is None:
                            continue
                        for xyxy, cls, score in zip(result.boxes.xyxy.tolist(),
                                                    result.boxes.cls.tolist(),
                                                    result.boxes.conf.tolist()):
                            handle.write(row(frame, str(names.get(int(cls), int(cls))), score,
                                             norm_box(*xyxy, width, height), backend))
                            count += 1
                    return count

                def prep(chunk):
                    return [os.path.join(paths.FRAMESET, f["frame"]) for f in chunk]

            elif family == "hf-owlv2":
                from transformers import Owlv2ForObjectDetection, Owlv2Processor
                processor = Owlv2Processor.from_pretrained(weights)
                if args.hf_imgsz:
                    # OWLv2's ViT has LEARNED position embeddings sized to its native 960 square
                    # (60x60 patches + 1). Changing the input size without asking the model to
                    # interpolate them raises a shape mismatch -- 2501 against 3601 at 800px --
                    # so `interpolate_pos_encoding=True` below is required, not optional.
                    processor.image_processor.size = {"height": args.hf_imgsz,
                                                      "width": args.hf_imgsz}
                model = Owlv2ForObjectDetection.from_pretrained(weights).to(device).eval()

                def prep(chunk):
                    return [Image.open(os.path.join(paths.FRAMESET, f["frame"])).convert("RGB")
                            for f in chunk]

                def infer(images):
                    inputs = processor(text=[prompts] * len(images), images=images,
                                       return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model(**inputs,
                                        interpolate_pos_encoding=bool(args.hf_imgsz))
                    sizes = torch.tensor([im.size[::-1] for im in images]).to(device)
                    return processor.post_process_grounded_object_detection(
                        outputs=outputs, threshold=args.hf_conf, target_sizes=sizes)

                def emit(handle, chunk, results, images=None):
                    count = 0
                    for frame, image, result in zip(chunk, images, results):
                        width, height = image.size
                        for box, score, label in zip(result["boxes"], result["scores"],
                                                     result["labels"]):
                            count += 1
                            handle.write(row(frame, prompts[int(label)], float(score),
                                             norm_box(*[float(v) for v in box.tolist()],
                                                      width, height), backend))
                    return count

            elif family == "hf-gdino":
                from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
                processor = AutoProcessor.from_pretrained(weights)
                if args.hf_imgsz:
                    # Swin is windowed and fully convolutional in its stage transitions, so it
                    # accepts a larger input directly. Keep the stock 1333/800 aspect cap so a
                    # 16:9 frame is not letterboxed differently at each rung of the sweep.
                    processor.image_processor.size = {
                        "shortest_edge": args.hf_imgsz,
                        "longest_edge": int(round(args.hf_imgsz * 1333 / 800)),
                    }
                model = (AutoModelForZeroShotObjectDetection
                         .from_pretrained(weights).to(device).eval())
                text, spans, _ = phrase_token_spans(processor.tokenizer, prompts)
                empty_spans = sum(1 for lo, hi in spans if hi <= lo)
                if empty_spans:
                    print(f"    WARNING: {empty_spans} prompt(s) had no token span")

                def prep(chunk):
                    return [Image.open(os.path.join(paths.FRAMESET, f["frame"])).convert("RGB")
                            for f in chunk]

                def infer(images):
                    inputs = processor(images=images, text=[text] * len(images),
                                       return_tensors="pt", padding=True).to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    return outputs

                def emit(handle, chunk, outputs, images=None):
                    # Attribution done here rather than by the processor: probs is
                    # (batch, queries, text_len), so the highest probability inside a phrase's
                    # own token span is that phrase's score for that box.
                    probs = torch.sigmoid(outputs.logits)
                    boxes_all = outputs.pred_boxes
                    count = 0
                    for i, (frame, image) in enumerate(zip(chunk, images)):
                        width, height = image.size
                        prob = probs[i]
                        keep = (prob.max(dim=-1).values > args.hf_conf).nonzero().flatten()
                        if keep.numel() == 0:
                            continue
                        phrase_scores = torch.stack(
                            [prob[keep, lo:hi].max(dim=-1).values if hi > lo
                             else torch.zeros(keep.numel(), device=prob.device)
                             for lo, hi in spans], dim=-1)
                        best = phrase_scores.argmax(dim=-1)
                        for j, q in enumerate(keep.tolist()):
                            cx, cy, bw, bh = boxes_all[i][q].tolist()
                            x1, y1 = (cx - bw / 2) * width, (cy - bh / 2) * height
                            x2, y2 = (cx + bw / 2) * width, (cy + bh / 2) * height
                            handle.write(row(frame, prompts[int(best[j])],
                                             float(phrase_scores[j, best[j]]),
                                             norm_box(x1, y1, x2, y2, width, height), backend))
                            count += 1
                    return count
            else:
                raise ValueError(f"unknown family {family}")

            # ---------------- warm-up ----------------
            warm = frames[: args.warmup]
            if warm:
                inputs = prep(warm)
                infer(inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            # ---------------- throughput (batched) ----------------
            started = time.perf_counter()
            with open(path, "w") as handle:
                for start in range(0, len(frames), args.batch):
                    chunk = frames[start : start + args.batch]
                    inputs = prep(chunk)
                    results = infer(inputs)
                    if family in ("hf-owlv2", "hf-gdino"):
                        written += emit(handle, chunk, results, images=inputs)
                    else:
                        written += emit(handle, chunk, results)
                    print(f"    {min(start + args.batch, len(frames))}/{len(frames)} frames, "
                          f"{written} detections", end="\r", flush=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started

            # ---------------- latency (batch of 1) ----------------
            times = []
            for frame in frames[: args.latency_frames]:
                inputs = prep([frame])
                if device.type == "cuda":
                    torch.cuda.synchronize()
                tic = time.perf_counter()
                infer(inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - tic) * 1000.0)
            times.sort()
            median = times[len(times) // 2] if times else float("nan")
            p90 = times[int(len(times) * 0.9)] if times else float("nan")

            print(f"    {len(frames)} frames, {written} detections | "
                  f"throughput {elapsed / len(frames) * 1000:.0f} ms/frame "
                  f"({len(frames) / elapsed:.1f} fps) | "
                  f"latency median {median:.0f} ms, p90 {p90:.0f} ms      ")
            summary.append({
                "backend": out_name, "detections": written,
                "throughput_ms_per_frame": round(elapsed / len(frames) * 1000, 1),
                "throughput_fps": round(len(frames) / elapsed, 2),
                "latency_ms_median": round(median, 1), "latency_ms_p90": round(p90, 1),
                "prompts": len(prompts), "batch": args.batch,
            })
        except Exception as exc:
            # One backend failing must not lose the runs that already succeeded.
            import traceback
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            summary.append({"backend": out_name, "error": f"{type(exc).__name__}: {exc}"})

    out_summary = os.path.join(args.out, "summary.json")
    existing = []
    if os.path.exists(out_summary):
        with open(out_summary) as handle:
            existing = json.load(handle)
    done = {entry["backend"] for entry in summary}
    merged = [entry for entry in existing if entry.get("backend") not in done] + summary
    with open(out_summary, "w") as handle:
        json.dump(merged, handle, indent=2)
    print(f"\nwrote {args.out}/<backend>/out.jsonl and summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
