"""Crop-and-embed entity tagger: detect, crop, embed with SigLIP 2, one vector tag per detection.
Tags stay per-frame with their box intact so bbox video-editor overlay works.

Two detectors, one embedder
---------------------------
`person` and `brand` are served by different models, because the measurements say they want
different ones: a closed COCO detector wins person outright (AP 0.752, mean IoU 0.89, and the
cheapest model in the study) while only open-vocabulary models can find brand marks at all.
Targets are routed by `prompts.split_by_detector`, and a detector with nothing routed to it is
never constructed -- a person-only request never pays to load the brand model.

They run sequentially, and the order does not matter. On one GPU two detectors do not overlap
usefully: they compete for the same SMs, so running them concurrently costs the same wall-clock
plus scheduling overhead. What does overlap is decode (CPU) against inference (GPU), which is the
frame pipeline's business rather than this module's.

Both detectors' crops are embedded in ONE batch. That matters: the embedder is the larger cost at
low detection counts, and batching across detectors keeps it near its throughput rather than its
latency.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import torch
from dacite import from_dict
from loguru import logger

from common_ml.tagging.models.frame_based import FrameModel
from common_ml.tagging.models.tag_types import FrameTag

from general_detection.config import RuntimeConfig
from general_detection.detector import (
    Detection,
    build_brand_detector,
    build_person_detector,
)
from general_detection.embedder import Siglip2CropEmbedder
from general_detection.prompts import expand_target, split_by_detector

# Used for the optional whole-frame vector: anchored to the full image in normalized coords.
_WHOLE_FRAME_BOX = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}


class EntityDetector(FrameModel):
    """Detects the configured targets and emits one SigLIP 2 embedding per detection."""

    def __init__(
        self,
        cfg: RuntimeConfig,
        embedder_model_id: str,
        cache_dir: str,
        embedder_revision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = cfg
        self.cache_dir = cache_dir
        self.device = device
        self._brand = None
        self._person = None
        self._brand_mode: Optional[str] = None
        self._apply_targets(cfg)
        # Same device as the detectors. Without threading it through, an explicit device (say
        # "cuda:1") would place the detectors there and the embedder on cuda:0, which works but
        # copies every crop across devices and silently occupies a card the caller did not ask
        # for. None keeps the embedder's own auto-detection.
        self.embedder = Siglip2CropEmbedder(
            embedder_model_id,
            revision=embedder_revision,
            device=torch.device(device) if device else None,
        )

    # ---- configuration ----------------------------------------------------------

    def _resolve_prompts(self, cfg: RuntimeConfig) -> Dict[str, List[str]]:
        """`detect_target` wins when set; otherwise `class_prompts` (whose default is the
        brand+person schema)."""
        if cfg.detect_target:
            return expand_target(cfg.detect_target)
        return {k: list(v) for k, v in cfg.class_prompts.items()}

    def _apply_targets(self, cfg: RuntimeConfig) -> None:
        """Build/refresh only the detectors the current target actually needs.

        Loading is lazy and per-role, so a target that routes to one side never constructs the
        other. Rebuilding is confined to a real backend change (`brand_detector` switching
        between "fast" and "coverage"); a prompt-only change re-encodes text, which is seconds,
        rather than reloading weights.
        """
        open_vocab, closed = split_by_detector(self._resolve_prompts(cfg))

        if open_vocab:
            if self._brand is None or self._brand_mode != cfg.brand_detector:
                self._brand = build_brand_detector(cfg.brand_detector, self.cache_dir, cfg,
                                                   self.device)
                self._brand_mode = cfg.brand_detector
            self._brand.set_prompts(open_vocab)
        else:
            # Released rather than kept idle: these are the large weights, and a caller that
            # narrowed its target to person should get the memory back.
            self._brand, self._brand_mode = None, None

        if closed:
            if self._person is None:
                self._person = build_person_detector(self.cache_dir, cfg, self.device)
            self._person.set_prompts(closed)
        else:
            self._person = None

        logger.info(
            f"targets: open-vocab={sorted(open_vocab) or '-'} "
            f"({cfg.brand_detector if open_vocab else 'not loaded'}), "
            f"closed={sorted(closed) or '-'}"
        )

    def set_config(self, config: dict) -> None:
        self.config = from_dict(RuntimeConfig, config)
        self._apply_targets(self.config)

    def get_config(self) -> dict:
        return asdict(self.config)

    # ---- tagging ----------------------------------------------------------------

    def tag_frame(self, img: np.ndarray) -> List[FrameTag]:
        """img: (H, W, 3) uint8 RGB. One FrameTag per detection: `tag` is the parent term,
        `vector` the crop embedding, `box` the normalized un-padded detection box."""
        cfg = self.config

        detections: List[Detection] = []
        for detector in (self._person, self._brand):
            # Person first only because it is the cheap one, so a frame that is going to fail
            # some later guard fails sooner. Nothing depends on the order.
            if detector is not None:
                detections.extend(detector.detect(img, cfg))

        crops = [d.crop for d in detections]
        if cfg.embed_whole_frame:
            crops.append(np.ascontiguousarray(img))
        if not crops:
            return []

        vectors, upscales = self.embedder.embed(crops, cfg)

        tags: List[FrameTag] = []
        for i, detection in enumerate(detections):
            tags.append(
                FrameTag(
                    tag=detection.label,
                    vector=vectors[i].tolist(),
                    box=detection.box,
                    additional_info={
                        "kind": "crop",
                        "prompt": detection.prompt,
                        "score": detection.score,
                        # crop_padding changes the vector, so it is provenance, not trivia:
                        # vectors built at different padding are not comparable.
                        "crop_padding": cfg.crop_padding,
                        "upscale": upscales[i],
                        # Which backend found it. With two detectors in play this is no longer
                        # constant per run, so it is recorded per tag rather than per config.
                        "detector": detection.detector,
                        **self._embedder_info(),
                    },
                )
            )

        if cfg.embed_whole_frame:
            tags.append(
                FrameTag(
                    tag="",  # no class: this is the frame itself, not a detected entity
                    vector=vectors[-1].tolist(),
                    # dict(...) so the tag owns its box and the module constant is never mutated
                    box=dict(_WHOLE_FRAME_BOX),
                    additional_info={
                        "kind": "frame",
                        "upscale": upscales[-1],
                        **self._embedder_info(),
                    },
                )
            )

        return tags

    def _embedder_info(self) -> Dict:
        """Provenance stamped on every tag so the index can validate what it is storing and
        so a checkpoint or budget change is visible after the fact rather than silent."""
        return {
            "embedder": self.embedder.model_id,
            "dim": self.embedder.dim,
            "max_num_patches": self.config.max_num_patches,
        }
