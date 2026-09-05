"""Resolve the bounding box for a detection hit.

`model-detection` emits a box per detection (`FrameTag.box`, surfaced as
`Tag.frame_info.box`), and it is the box that makes a result readable — a frame is 4K and a
`logo` hit might be a 22 px wordmark somewhere inside it.

`model-detection` now also stamps the box into `additional_info`, which the vectorstore does
carry back, so a search row against a freshly built index needs no lookup at all —
`box_from_info` reads it straight off the row. Everything below is the fallback for indexes
built before that, whose rows have provenance and no geometry: the box has to be joined back
from wherever the tags were written.

Three places are tried, in this order, and all three are checked because they fail
differently:

1. **tagstore, keyed on the content qid** — where a per-content tag track lands.
2. **tagstore, keyed on the index qid** — an index aggregates tags from its content, and on
   the index tested this is the only key that returns anything at all.
3. **fabric `video_tags` metadata** — what EVIE itself reads (`VideoStore.LoadMetadata`
   selects `video_tags`, and `Helpers.js` walks `video_tags.metadata_tags[track].tags`), so
   anything visible in EVIE's tag tracks is visible here.

**On the index tested, all three come back empty**, and that is a property of the data, not
of this file: the tagging run wrote vectors to the vectorstore without writing a tag track
back to the content objects (`model-detection`'s `output_tags` defaults to false). Verified
with tokens that fully open the content objects — the tagstore reports zero tracks for them
and `video_tags` is `{}`. The index object *does* carry tag tracks, but they belong to other
taggers (`logo_detection`, `object_detection`, `celebrity_detection`) and not to this index's
`detection` vectors.

So results off that index render as whole frames. `status()` reports which sources were tried and
what each said, so the UI can state the reason rather than showing an unexplained absence.
`resolve` never raises: a box is an enhancement, and losing it must not lose the result.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests
from loguru import logger

BOX_KEYS = ("x1", "y1", "x2", "y2")


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def as_dict(self) -> Dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass
class SourceStatus:
    """What one lookup source had to say, for reporting rather than for control flow."""

    source: str
    tags: int = 0
    detail: str = ""


@dataclass
class LookupResult:
    by_frame: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)
    statuses: List[SourceStatus] = field(default_factory=list)


class BoxResolver:
    """Finds per-detection boxes, cached per (content qid, index qid, track)."""

    def __init__(
        self,
        tagstore_url: str,
        fabric=None,
        timeout: float = 60.0,
        page_limit: int = 5000,
    ) -> None:
        self.tagstore_url = tagstore_url.rstrip("/")
        self.fabric = fabric
        self.timeout = timeout
        self.page_limit = page_limit
        self._cache: Dict[Tuple[str, str, str], LookupResult] = {}
        self._lock = threading.Lock()

    def resolve(
        self,
        qid: str,
        track: str,
        frame_idx: Optional[int],
        token: str,
        index_qid: str = "",
        tag_hint: Optional[str] = None,
    ) -> Optional[Box]:
        """Best box for a hit, or None.

        `tag_hint` is the detection's parent term (`logo`, `person`, ...) when known; with
        several detections in one frame it picks the right one rather than the first.
        """
        if frame_idx is None:
            return None
        try:
            candidates = self._lookup(qid, index_qid, track, token).by_frame.get(frame_idx, [])
            if not candidates:
                return None
            if tag_hint:
                matching = [t for t in candidates if t.get("tag") == tag_hint]
                candidates = matching or candidates
            return _extract_box(candidates[0])
        except Exception as exc:
            logger.debug("box lookup failed for {} frame {}: {}", qid, frame_idx, exc)
            return None

    def status(self, qid: str, index_qid: str, track: str, token: str) -> List[Dict[str, Any]]:
        result = self._lookup(qid, index_qid, track, token)
        return [
            {"source": s.source, "tags": s.tags, "detail": s.detail} for s in result.statuses
        ]

    # -- lookup -----------------------------------------------------------------------

    def _lookup(self, qid: str, index_qid: str, track: str, token: str) -> LookupResult:
        key = (qid, index_qid, track)
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        sources = [(f"tagstore[{qid}]", lambda: self._from_tagstore(qid, track, token))]
        if index_qid and index_qid != qid:
            sources.append(
                (f"tagstore[{index_qid}]", lambda: self._from_tagstore(index_qid, track, token))
            )
        sources.append((f"fabric video_tags[{qid}]", lambda: self._from_fabric(qid, track, token)))

        result = LookupResult()
        for name, call in sources:
            try:
                tags, detail = call()
            except Exception as exc:
                result.statuses.append(SourceStatus(name, 0, f"error: {exc}"))
                continue
            result.statuses.append(SourceStatus(name, len(tags), detail))
            for tag in tags:
                frame_idx = _frame_idx_of(tag)
                if frame_idx is not None:
                    result.by_frame.setdefault(frame_idx, []).append(tag)
            if result.by_frame:
                # First source with usable geometry wins; no reason to pay for the rest.
                break

        if not result.by_frame:
            logger.info(
                "no '{}' boxes for {} — {}",
                track,
                qid,
                "; ".join(f"{s.source}: {s.tags} tags {s.detail}".strip() for s in result.statuses),
            )

        with self._lock:
            self._cache[key] = result
        return result

    def _from_tagstore(self, qid: str, track: str, token: str) -> Tuple[List[Dict[str, Any]], str]:
        tags: List[Dict[str, Any]] = []
        start = 0
        while True:
            response = requests.get(
                f"{self.tagstore_url}/{qid}/tags",
                params={"track": track, "start": start, "limit": self.page_limit},
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                timeout=self.timeout,
            )
            if response.status_code in (401, 403):
                # Worth distinguishing from "no tags": a 401 here means the token lacks
                # tagstore permission on this qid, which is fixable by supplying another.
                return [], f"HTTP {response.status_code} (token lacks tagstore access)"
            response.raise_for_status()
            body = response.json() or {}
            page = body.get("tags") or []
            tags.extend(page)
            start += len(page)
            if not page or start >= int((body.get("meta") or {}).get("total", start)):
                break
        return tags, ""

    def _from_fabric(self, qid: str, track: str, token: str) -> Tuple[List[Dict[str, Any]], str]:
        """Read the tag track EVIE reads: `video_tags.metadata_tags[track].tags`."""
        if self.fabric is None:
            return [], "fabric client not configured"
        resolved = self.fabric.content_object(qid, token)
        response = requests.get(
            f"{self.fabric.node}/q/{resolved.version_hash}/meta/video_tags",
            params={
                "authorization": token,
                # video_tags holds links to tag files rather than the tags themselves, so
                # without resolution this always looks empty even when it is not.
                "resolve": "true",
                "link_depth": "2",
                "resolve_ignore_errors": "true",
            },
            timeout=self.timeout,
        )
        if response.status_code in (401, 403):
            return [], f"HTTP {response.status_code}"
        response.raise_for_status()
        body = response.json() or {}
        metadata_tags = body.get("metadata_tags") or {}
        if not metadata_tags:
            return [], "video_tags is empty"

        tags: List[Dict[str, Any]] = []
        # Shape is {track_key: {"tags": [...]}}, sometimes nested one level under a source
        # key, so both layouts are accepted.
        for key, value in metadata_tags.items():
            if not isinstance(value, dict):
                continue
            if key == track or value.get("label") == track:
                tags.extend(value.get("tags") or [])
            nested = value.get("metadata_tags")
            if isinstance(nested, dict) and track in nested:
                tags.extend((nested[track] or {}).get("tags") or [])
        return tags, "" if tags else f"no '{track}' track in video_tags"


def box_from_info(additional_info: Optional[Dict[str, Any]]) -> Optional[Box]:
    """The box the tagger stamped onto the vector row itself, when it is there.

    Preferred over `BoxResolver`: no request, and no risk of joining to the wrong detection.
    Whole-frame vectors are skipped — their box is the entire frame, so outlining it says
    nothing and cropping to it returns the frame again.
    """
    if not additional_info or additional_info.get("kind") == "frame":
        return None
    return _extract_box(additional_info)


def _frame_idx_of(tag: Dict[str, Any]) -> Optional[int]:
    """Tolerant of where the frame index sits, since the tag shape is not yet pinned."""
    for candidate in (tag.get("frame_idx"), (tag.get("frame_info") or {}).get("frame_idx")):
        if candidate is not None:
            try:
                return int(candidate)
            except (TypeError, ValueError):
                continue
    return None


def _extract_box(tag: Dict[str, Any]) -> Optional[Box]:
    """Pull a normalized box out of a tag, wherever it is carried."""
    for source in (tag.get("box"), (tag.get("frame_info") or {}).get("box"), tag):
        if isinstance(source, dict) and all(k in source for k in BOX_KEYS):
            try:
                return Box(*(float(source[k]) for k in BOX_KEYS))
            except (TypeError, ValueError):
                continue
    return None
