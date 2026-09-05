"""Content-fabric access: turn a search hit into a frame image the browser can load.

A hit gives `(qid, start_time, frame_idx)`. A frame image needs `(node, versionHash, t)`,
so the only real work here is resolving the object and picking `t`.

The frame URL shape is the one EVIE uses (`src/stores/AIStore.js`):

    {node}/q/{versionHash}/rep/frame_extract/{offering}/video
        ?t={seconds}&exact=true&ignore_trimming=true&authorization={token}

Auth goes in a *query parameter*, not a header, which is what makes a plain `<img src>`
work: the browser fetches the frame straight from the fabric with no proxy and no CORS
preflight.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlencode

import requests
from loguru import logger


@dataclass(frozen=True)
class ContentObject:
    object_id: str
    version_hash: str
    library_id: Optional[str]


class FabricClient:
    """Resolves objects and builds frame URLs. Caches per (node, token) process-wide."""

    def __init__(self, config_url: str, offering: str, timeout: float = 60.0) -> None:
        self.config_url = config_url
        self.offering = offering
        self.timeout = timeout
        self._node: Optional[str] = None
        self._objects: Dict[str, ContentObject] = {}
        self._frame_rates: Dict[str, Optional[float]] = {}
        self._lock = threading.Lock()

    # -- node -------------------------------------------------------------------------

    @property
    def node(self) -> str:
        with self._lock:
            if self._node is None:
                config = requests.get(self.config_url, timeout=self.timeout).json()
                uris = config.get("network", {}).get("services", {}).get("fabric_api", [])
                if not uris:
                    raise RuntimeError(f"no fabric_api nodes in {self.config_url}")
                self._node = uris[0].rstrip("/")
                logger.info("fabric node: {}", self._node)
            return self._node

    # -- objects ----------------------------------------------------------------------

    def content_object(self, object_id: str, token: str) -> ContentObject:
        """Resolve an object id to its latest version hash.

        Deliberately `GET /q/{objectId}` rather than `GET /qid/{objectId}`: the latter
        needs the `q.read.versions` permission, which a content-scoped token generally
        does not carry, while this one only needs read.
        """
        with self._lock:
            cached = self._objects.get(object_id)
        if cached is not None:
            return cached

        response = requests.get(
            f"{self.node}/q/{object_id}", params={"authorization": token}, timeout=self.timeout
        )
        response.raise_for_status()
        body = response.json() or {}
        resolved = ContentObject(
            object_id=object_id,
            version_hash=body.get("hash") or object_id,
            library_id=body.get("qlib_id"),
        )
        with self._lock:
            self._objects[object_id] = resolved
        return resolved

    def frame_rate(self, object_id: str, token: str) -> Optional[float]:
        """The offering's video frame rate, as a float, or None if it cannot be read.

        Only needed to *display* a frame index for a hit that has no `start_time`; the
        frame URL itself never depends on it. That distinction matters — see `frame_time`.
        """
        with self._lock:
            if object_id in self._frame_rates:
                return self._frame_rates[object_id]

        rate: Optional[float] = None
        try:
            resolved = self.content_object(object_id, token)
            response = requests.get(
                f"{self.node}/q/{resolved.version_hash}/meta/offerings/{self.offering}"
                f"/media_struct/streams/video/rate",
                params={"authorization": token},
                timeout=self.timeout,
            )
            if response.ok:
                rate = _parse_rate(response.json())
        except Exception as exc:  # a missing rate is not worth failing a search over
            logger.warning("could not read frame rate for {}: {}", object_id, exc)

        with self._lock:
            self._frame_rates[object_id] = rate
        return rate

    # -- frames -----------------------------------------------------------------------

    def frame_url(
        self,
        object_id: str,
        token: str,
        seconds: float,
        height: Optional[int] = None,
    ) -> str:
        resolved = self.content_object(object_id, token)
        params = {
            "authorization": token,
            "t": f"{seconds:.3f}",
            # Without this the fabric returns the nearest keyframe, which for a 30-second
            # GOP can be a completely different shot from the one that was tagged.
            "exact": "true",
            # Frame indices are counted from the start of the mezzanine, so a trimmed
            # offering would shift every t by the trim offset.
            "ignore_trimming": "true",
        }
        if height:
            # Source frames here are 4K (~640 KB); a page of 50 is 30 MB. Resizing
            # server-side makes a thumbnail ~50 KB. Full resolution is fetched only when
            # a result is opened.
            params["height"] = str(height)
        return (
            f"{self.node}/q/{resolved.version_hash}/rep/frame_extract/{self.offering}/video"
            f"?{urlencode(params)}"
        )


def frame_time(start_time_ms: Optional[int], frame_idx: Optional[int], fps: Optional[float]) -> Optional[float]:
    """Seconds to extract for a hit, preferring `start_time` over `frame_idx / fps`.

    Both are stamped by `common_ml` and on a well-formed index they agree exactly —
    verified on this data, where frame 164735 at 6870822 ms implies 23.9760 fps against a
    declared 24000/1001. `start_time` is preferred anyway because it needs no fps at all:
    reconstructing the time from `frame_idx` means guessing a frame rate, and EVIE's
    fallback of 24000/1001 is silently wrong on 25 or 30 fps content — it does not error,
    it just extracts a frame from somewhere else in the video.
    """
    if start_time_ms is not None:
        return start_time_ms / 1000.0
    if frame_idx is not None and fps:
        return frame_idx / fps
    return None


def evie_url(evie_base: str, object_id: str, seconds: float) -> str:
    """Deep link into EVIE's tag view at this frame, for a closer look than a grid gives."""
    return f"{evie_base.rstrip('/')}/#/{object_id}/tags?it={seconds:.3f}&isolate="


def _parse_rate(value) -> Optional[float]:
    """Frame rates come back as a rational string like "24000/1001"."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().strip('"')
        try:
            if "/" in text:
                numerator, denominator = text.split("/", 1)
                return float(numerator) / float(denominator)
            return float(text)
        except (ValueError, ZeroDivisionError):
            return None
    return None
