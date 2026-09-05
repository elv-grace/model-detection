"""Demo settings.

Everything here is a *default*: the index qid and auth token arrive per request from the
browser, so one running instance can be pointed at any index without a restart.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # --- vectorstore ---------------------------------------------------------------
    vectorstore_url: str = os.environ.get("VECTORSTORE_URL", "http://localhost:8108")
    # The tagstore is where per-detection boxes are expected to land (track "detection").
    # It is optional: the demo renders whole frames when no box is available. See boxes.py.
    tagstore_url: str = os.environ.get("TAGSTORE_URL", "http://localhost:8102")

    # --- fabric --------------------------------------------------------------------
    fabric_config_url: str = os.environ.get(
        "FABRIC_CONFIG_URL", "https://main.net955305.contentfabric.io/config"
    )
    # Frame images are pulled straight from the fabric by the browser, so the offering
    # has to be named. "default" is what EVIE assumes too.
    offering: str = os.environ.get("FABRIC_OFFERING", "default")
    # Grid thumbnails: the source frames here are 4K (~640 KB each), and a 50-result page
    # of those is 30 MB. The frame_extract rep resizes server-side for ~50 KB instead.
    thumbnail_height: int = 360

    # --- embedder ------------------------------------------------------------------
    # Must match what the index was tagged with. The /api/index_info route reads the
    # index's own additional_info and warns in the UI if these disagree, rather than
    # silently returning vectors from a different space.
    model_id: str = os.environ.get("SIGLIP_MODEL_ID", "google/siglip2-base-patch16-naflex")
    # Pinned to the same hub commit content-search pins, so the query side cannot drift
    # onto a different snapshot of the checkpoint than the index was built with.
    revision: str | None = os.environ.get(
        "SIGLIP_REVISION", "b53b807d3a2d5e2b3911292f2d69e5341cdc064c"
    )
    max_num_patches: int = int(os.environ.get("SIGLIP_MAX_NUM_PATCHES", "256"))
    normalize: bool = True
    # SigLIP 2 base emits 768-d; the vectorstore index is 1024-d and the tagger
    # right-zero-padded to reach it. Trailing zeros leave the dot product and both norms
    # unchanged, so cosine is preserved exactly.
    target_size: int = int(os.environ.get("INDEX_VECTOR_SIZE", "1024"))
    # The padding the indexed crops were taken with. An image query drawn as a tight box
    # is expanded by this much so it matches the framing of what it is searching against;
    # a mismatch of +-0.06 measured at 8% of top-1 retrievals lost. See model-detection's
    # README, "crop_padding changes the vector (measured)".
    index_crop_padding: float = float(os.environ.get("INDEX_CROP_PADDING", "0.06"))

    # --- search --------------------------------------------------------------------
    # The store returns a top-K with no offset and postprocessing can drop rows, so the
    # pool fetched is larger than the page rendered.
    default_limit: int = 50
    max_limit: int = 200
    overfetch_factor: int = 3
    max_pool: int = 1000

    http_timeout: float = 60.0


settings = Settings()
