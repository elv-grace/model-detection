#!/usr/bin/env python3
"""Measure SigLIP 2 crop-embedding throughput, so Phase A cost estimates are not guesses.

The detector timings answer half of "how long does a two-hour video take". The other half is the
embedder, and its cost does NOT scale with frames -- it scales with CROPS, which is frames times
detections per frame. A detector that emits 30 crops a frame can cost more downstream than one
twice as slow that emits 10.

Crops are synthesised at the size distribution measured from the real runs rather than from
fixed squares, because NaFlex is resolution-adaptive: it preserves aspect ratio and picks a patch
count per image, so a batch of 20x60 px marks and a batch of 400x400 px people do not cost the
same. Feeding it uniform squares would measure the wrong thing.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/siglip2-base-patch16-naflex")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    from transformers import AutoModel, AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    # Sizes taken from the measured crop distributions: brand marks are small and often wide
    # (wordmarks), people are large and tall.
    profiles = {
        "brand marks (median 37px, wordmark aspect)": [(64, 24), (40, 40), (96, 32), (28, 28)],
        "people (median 78px, tall)":                 [(80, 200), (120, 300), (60, 160)],
        "mixed, as a real run emits":                 [(64, 24), (40, 40), (80, 200), (120, 300)],
    }

    print(f"{args.model} on {device}, batch {args.batch}\n")
    print(f"{'crop profile':<44}{'ms/batch':>10}{'ms/crop':>10}{'crops/s':>10}")
    print("-" * 74)
    for name, sizes in profiles.items():
        images = [Image.new("RGB", sizes[i % len(sizes)], (128, 128, 128))
                  for i in range(args.batch)]
        times = []
        for i in range(args.warmup + args.iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                inputs = processor(images=images, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model.get_image_features(**inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= args.warmup:
                times.append((time.perf_counter() - start) * 1000)
        per_batch = statistics.median(times)
        print(f"{name:<44}{per_batch:>10.1f}{per_batch / args.batch:>10.2f}"
              f"{1000 / (per_batch / args.batch):>10.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
