"""Query-side SigLIP 2 embedder: text and image into the index's space.

The whole point of this module is *parity with the tagger*. `general_detection/embedder.py`
stores `Siglip2VisionModel(...).pooler_output`, L2-normalized. Here the full `Siglip2Model`
is loaded instead of the two towers separately, for one reason: it carries `logit_scale`
and `logit_bias`, which are what turn a text->crop cosine into a calibrated probability
(see `text_probability` below).

That swap is safe, and it is worth being explicit about why, because it would be an easy
place to silently produce vectors from a different space. In transformers, SigLIP 2 has no
projection head on either tower:

    Siglip2Model.get_image_features(...) -> self.vision_model(...)
    Siglip2Model.get_text_features(...)  -> self.text_model(...)

so `Siglip2Model.vision_model(...).pooler_output` is the same tensor
`Siglip2VisionModel(...).pooler_output` returns. (CLIP is *not* like this — it projects
both towers — so the reasoning does not transfer to a CLIP checkpoint.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import IO, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image, ImageOps
from transformers import Siglip2Model, Siglip2Processor

# Fixed by the patch16 checkpoints; NaFlex varies the grid, not the patch size.
PATCH_SIZE = 16


@dataclass(frozen=True)
class QueryVector:
    """An embedded query, plus what it took to produce it.

    `vector` is padded to the index width. `raw_dim` is the model's own width before
    padding, which the caller checks against the index's `additional_info.dim`.
    """

    vector: List[float]
    raw_dim: int
    modality: str  # "text" | "image"


class Siglip2QueryEmbedder:
    """Embeds text or image queries into the space `model-detection` indexed crops into."""

    def __init__(
        self,
        model_id: str,
        revision: Optional[str],
        max_num_patches: int,
        normalize: bool,
        target_size: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if max_num_patches < 1:
            # Guard before pulling several GB of weights, so a bad param fails fast.
            raise ValueError(f"max_num_patches must be >= 1, got {max_num_patches!r}")

        self.model_id = model_id
        self.max_num_patches = max_num_patches
        self.normalize = normalize
        self.target_size = target_size
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
        self.processor = Siglip2Processor.from_pretrained(model_id, revision=revision)
        self.model = Siglip2Model.from_pretrained(model_id, revision=revision, dtype=dtype).to(
            self.device
        )
        self.model.eval()

        self.dim = int(self.model.config.vision_config.hidden_size)
        if self.dim > target_size:
            raise ValueError(
                f"{model_id} emits {self.dim}-d, which cannot be padded down to the index's "
                f"{target_size}-d — the query encoder and the index disagree on the checkpoint"
            )
        logger.info(f"query embedder ready: {self.dim}-d, padding to {target_size}-d")

    # -- text -------------------------------------------------------------------------

    def embed_text(self, query: str) -> QueryVector:
        # Siglip2Processor rather than a bare tokenizer: its text defaults are
        # padding="max_length", truncation=True, max_length=64, which is the fixed-length
        # padding SigLIP was trained on. Tokenized any other way the query vector moves far
        # enough that results collapse onto whatever content dominates the index.
        inputs = self.processor(text=[query], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # .float() before normalizing: dividing in bf16 lands ~0.1% off unit length,
            # which a cosine index reads as a real score difference.
            pooled = self.model.get_text_features(**inputs).pooler_output.float()
            if self.normalize:
                pooled = F.normalize(pooled, p=2, dim=-1)

        return self._to_query_vector(pooled.squeeze(0).cpu().numpy(), "text")

    def text_probability(self, cosine: float) -> float:
        """SigLIP 2's calibrated match probability for a (text, image) cosine.

        Text->crop and crop->crop are different score regimes — the modality gap puts a
        perfect text match around cos 0.05-0.3 where a good image match sits at 0.5-0.9 —
        so a single cosine threshold cannot serve both. SigLIP 2's sigmoid loss trains a
        per-pair probability with no softmax over a batch, so this number *is* comparable
        across queries in a way the raw cosine is not.

        `logit_scale` is stored as the log of the scale and must be exponentiated.
        """
        with torch.no_grad():
            scale = self.model.logit_scale.exp().float().item()
            bias = self.model.logit_bias.float().item()
        return 1.0 / (1.0 + math.exp(-(cosine * scale + bias)))

    # -- image ------------------------------------------------------------------------

    def embed_image(
        self, image: IO[bytes], crop: Optional[Tuple[float, float, float, float]] = None
    ) -> Tuple[QueryVector, "Image.Image"]:
        """Embed an uploaded image, optionally cropped first.

        `crop` is `(x, y, w, h)` in *fractions* of the image, which is what the browser's
        drag-select produces and what keeps this independent of display scaling. Returns
        the vector and the image actually embedded, so the UI can show what it searched
        with rather than what was uploaded.
        """
        decoded = decode_image(image)
        if crop is not None:
            decoded = apply_crop(decoded, crop)

        # Query-by-example goes through the vision tower, so it must be preprocessed the
        # way the indexed crops were: same checkpoint, same patch budget, same normalize.
        inputs = self.processor(
            images=decoded, return_tensors="pt", max_num_patches=self.max_num_patches
        )
        # Only pixel_values is float; pixel_attention_mask and spatial_shapes are integer
        # bookkeeping and must keep their own dtypes.
        model_inputs = {k: v.to(self.device) for k, v in inputs.items()}
        model_inputs["pixel_values"] = model_inputs["pixel_values"].to(self.dtype)

        with torch.no_grad():
            pooled = self.model.get_image_features(**model_inputs).pooler_output.float()
            if self.normalize:
                pooled = F.normalize(pooled, p=2, dim=-1)

        return self._to_query_vector(pooled.squeeze(0).cpu().numpy(), "image"), decoded

    # -- shared -----------------------------------------------------------------------

    def _to_query_vector(self, vector: np.ndarray, modality: str) -> QueryVector:
        values = [float(x) for x in vector]
        raw_dim = len(values)
        if raw_dim < self.target_size:
            # Right-zero-pad. Trailing zeros leave the dot product and both norms
            # unchanged, so cosine ranking is identical to the unpadded space.
            values = values + [0.0] * (self.target_size - raw_dim)
        return QueryVector(vector=values, raw_dim=raw_dim, modality=modality)


def decode_image(image: IO[bytes]) -> Image.Image:
    """Decode an upload into RGB."""
    decoded = Image.open(image)
    # A phone photo records its rotation in EXIF rather than in the pixel data, and
    # Image.open does not apply it, so without this an upload can be embedded sideways.
    decoded = ImageOps.exif_transpose(decoded) or decoded
    # CMYK, grayscale and RGBA uploads all have to reach the model as 3-channel RGB.
    return decoded.convert("RGB")


def apply_crop(image: Image.Image, crop: Tuple[float, float, float, float]) -> Image.Image:
    """Crop by fractional (x, y, w, h), clamped to the image."""
    width, height = image.size
    x, y, w, h = crop
    left = max(0, min(width - 1, int(round(x * width))))
    top = max(0, min(height - 1, int(round(y * height))))
    right = max(left + 1, min(width, int(round((x + w) * width))))
    bottom = max(top + 1, min(height, int(round((y + h) * height))))
    return image.crop((left, top, right, bottom))


def pad_crop(
    crop: Tuple[float, float, float, float], padding: float
) -> Tuple[float, float, float, float]:
    """Expand a fractional crop by `padding` of its own size on each side.

    The indexed crops were taken with `crop_padding` context around the detection box, so
    an object fills 1/(1+2p) of the crop's linear extent rather than all of it. A query
    box drawn tightly around a logo has different composition from everything it is being
    compared to, and SigLIP encodes the whole crop — so the vectors move apart for a
    reason that has nothing to do with the logo. Expanding here restores the framing.
    """
    if padding <= 0:
        return crop
    x, y, w, h = crop
    return (x - w * padding, y - h * padding, w * (1 + 2 * padding), h * (1 + 2 * padding))
