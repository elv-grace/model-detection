"""SigLIP 2 NaFlex crop embedder, loaded directly from HuggingFace transformers.

NaFlex ("native aspect ratio, flexible resolution") resizes each crop to a *patch budget*
rather than to a fixed square, preserving aspect ratio to within one patch. That matters
far more for crops than for whole frames: a 20x160 banner crop becomes 96x672 here, where a
fixed-resolution checkpoint (e.g. `-patch16-384`) would squash it 8x horizontally — and
text, signage, and wordmarks are precisely what a non-uniform squash destroys.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from transformers import Siglip2ImageProcessor, Siglip2VisionModel

from general_detection.config import RuntimeConfig

# Fixed by the patch16 checkpoints; NaFlex varies the grid, not the patch size.
PATCH_SIZE = 16


class Siglip2CropEmbedder:
    """Embeds a list of RGB crops into pooled SigLIP 2 vectors.

    The emitted dimension is read from the loaded checkpoint (`config.hidden_size`) rather
    than hardcoded, and is stamped into every tag's `additional_info.dim` so the index can
    validate it."""

    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,   # hub commit to pin; None -> default branch
        dtype: Optional[torch.dtype] = None,  # None -> auto (bf16/fp16 on GPU, fp32 on CPU)
        device: Optional[torch.device] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if dtype is None:
            if self.device.type == "cuda":
                # bf16 needs Ampere+ (compute capability >= 8.0); fall back to fp16.
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                logger.warning("cuda not available, embedding on cpu (slow)")
                dtype = torch.float32  # half precision is unstable / slow on CPU
        self.dtype = dtype

        logger.info(f"loading {model_id} (revision={revision}, dtype={dtype}, device={self.device})")
        # `revision` pins the processor and the model to the same hub commit so the whole
        # snapshot is reproducible.
        #
        # Vision tower only — no text embeddings are produced here. Loading a sub-model
        # makes transformers log the checkpoint's text-tower keys as UNEXPECTED; that
        # report is the discarded half of the checkpoint and is expected.
        self.processor = Siglip2ImageProcessor.from_pretrained(model_id, revision=revision)
        self.model = Siglip2VisionModel.from_pretrained(
            model_id, revision=revision, dtype=dtype
        ).to(self.device)
        self.model.eval()

        self.dim = int(self.model.config.hidden_size)
        logger.info(f"embedder ready: {self.dim}-d")

    def embed(
        self, crops: List[np.ndarray], cfg: RuntimeConfig
    ) -> Tuple[np.ndarray, List[float]]:
        """Return ((N, dim) float32 vectors, per-crop NaFlex upscale factors).

        The upscale factor is the linear scale the processor actually applied. It is
        reported per crop so the heavily-interpolated tail can be filtered downstream
        without re-tagging — see RuntimeConfig.min_crop_pixels.
        """
        vectors = np.zeros((len(crops), self.dim), dtype=np.float32)
        upscales: List[float] = [0.0] * len(crops)
        if not crops:
            return vectors, upscales

        # Group by patch budget. With max_upscale unset every crop lands in one bucket, so
        # this is a no-op and batching is maximally efficient.
        buckets: Dict[int, List[int]] = {}
        for i, crop in enumerate(crops):
            buckets.setdefault(self._budget(crop, cfg), []).append(i)

        for budget, indices in buckets.items():
            for start in range(0, len(indices), cfg.embed_batch_size):
                chunk = indices[start : start + cfg.embed_batch_size]
                batch = [crops[i] for i in chunk]
                batch_vectors, batch_upscales = self._forward(batch, budget, cfg.normalize)
                for slot, i in enumerate(chunk):
                    vectors[i] = batch_vectors[slot]
                    upscales[i] = batch_upscales[slot]

        return vectors, upscales

    def _budget(self, crop: np.ndarray, cfg: RuntimeConfig) -> int:
        """Patch budget for one crop, honouring cfg.max_upscale when set."""
        if cfg.max_upscale is None:
            return cfg.max_num_patches
        height, width = crop.shape[:2]
        native = math.ceil(height / PATCH_SIZE) * math.ceil(width / PATCH_SIZE)
        # Patch count scales with area, so a linear upscale cap of k allows k^2 patches.
        allowed = int(native * cfg.max_upscale ** 2)
        return max(1, min(cfg.max_num_patches, allowed))

    def _forward(
        self, batch: List[np.ndarray], budget: int, normalize: bool
    ) -> Tuple[np.ndarray, List[float]]:
        # Crops of differing sizes batch fine at one budget: each is padded along the patch
        # axis to `budget` and masked, which is what NaFlex is for.
        inputs = self.processor(
            images=[Image.fromarray(crop) for crop in batch],
            return_tensors="pt",
            max_num_patches=budget,
        )

        # spatial_shapes is (num_patches_h, num_patches_w) per image, so the applied scale
        # is recoverable from the public output — no private helper import needed.
        spatial_shapes = inputs["spatial_shapes"]
        upscales = [
            round(float(int(spatial_shapes[i][0]) * PATCH_SIZE / batch[i].shape[0]), 3)
            for i in range(len(batch))
        ]

        # Only pixel_values is float; pixel_attention_mask and spatial_shapes are integer
        # bookkeeping and must keep their own dtypes.
        model_inputs = {k: v.to(self.device) for k, v in inputs.items()}
        model_inputs["pixel_values"] = model_inputs["pixel_values"].to(self.dtype)

        with torch.no_grad():
            # .float() before normalizing: dividing in bf16 lands ~0.1% off unit length,
            # which a cosine index reads as a real score difference.
            pooled = self.model(**model_inputs).pooler_output.float()
            if normalize:
                # The pooling head's output is not unit length. SigLIP 2's trained
                # similarity is cosine, so this is lossless.
                pooled = F.normalize(pooled, p=2, dim=-1)

        return pooled.cpu().numpy(), upscales
