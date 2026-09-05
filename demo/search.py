"""Search orchestration: embed the query, hit the store, attach frames and boxes.

The postprocessing here differs from `content-search`'s frame search in one deliberate way.
Frame search dedupes by `(qid, frame_idx)` because a frame contributes exactly one vector.
A *detection* index contributes up to `max_detections` vectors per frame, and several
logos in one frame are several real findings — collapsing them would hide hits, which is
the opposite of what an evaluation tool should do. So the default is no collapsing at all,
and both collapse modes are opt-in.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from loguru import logger

from .boxes import BoxResolver, box_from_info
from .config import Settings
from .embedder import QueryVector, Siglip2QueryEmbedder
from .fabric import evie_url, frame_time
from .tokens import TokenRegistry
from .vectorstore import VectorHit, VectorStoreClient

DETECTION_TRACK = "detection"


@dataclass
class SearchRequest:
    index_qid: str
    # The token used against the vectorstore. Frames are fetched with whichever of
    # `content_tokens` opens the result's own content object, which is often a different
    # one — see tokens.py.
    token: str
    content_tokens: List[str] = field(default_factory=list)
    explicit_tokens: Dict[str, str] = field(default_factory=dict)
    limit: int = 50
    track: Optional[str] = DETECTION_TRACK
    qids: List[str] = field(default_factory=list)
    # Two floors rather than one, because text and image scores live on different scales.
    # A cosine floor that is sane for an image query (say 0.6) discards every text result
    # ever returned, since text tops out near 0.3. The UI sends whichever fits the query.
    min_similarity: Optional[float] = None
    min_probability: Optional[float] = None
    collapse: str = "none"  # "none" | "frame" | "time"
    collapse_gap_ms: int = 1000
    evie_base: Optional[str] = None


@dataclass
class SearchResult:
    qid: str
    vector_id: int
    similarity: float
    # Only meaningful for text queries; see Siglip2QueryEmbedder.text_probability.
    probability: Optional[float]
    frame_idx: Optional[int]
    start_time_ms: Optional[int]
    seconds: Optional[float]
    # All three point at this app's own /api/frame proxy rather than at the fabric, so a
    # denied frame comes back as a labelled image instead of a blank tile. See frames.py.
    thumbnail_url: Optional[str]
    full_url: Optional[str]
    # Only set when a box is known: the frame cropped to this detection, which is what
    # tells two results on the same frame apart.
    crop_url: Optional[str]
    evie_url: Optional[str]
    box: Optional[Dict[str, float]]
    additional_info: Dict[str, Any]
    # How many other hits landed on this same frame — the signal that a card is one of
    # several detections in one image rather than a lone match.
    crops_in_frame: int = 1
    error: Optional[str] = None


@dataclass
class SearchResponse:
    results: List[SearchResult]
    meta: Dict[str, Any]


class SearchService:
    def __init__(
        self,
        settings: Settings,
        embedder: Siglip2QueryEmbedder,
        store: VectorStoreClient,
        fabric,
        boxes: BoxResolver,
        tokens: TokenRegistry,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.fabric = fabric
        self.boxes = boxes
        self.tokens = tokens

    def search(self, request: SearchRequest, query: QueryVector) -> SearchResponse:
        limit = max(1, min(request.limit, self.settings.max_limit))
        # The store has no offset and postprocessing drops rows, so over-fetch. With
        # collapse off this is nearly a no-op, but a min_similarity floor can still cut
        # deeply into a page.
        pool = min(limit * self.settings.overfetch_factor, self.settings.max_pool)

        hits = self.store.search(
            index_qid=request.index_qid,
            token=request.token,
            vector=query.vector,
            limit=pool,
            track=request.track,
            qids=request.qids or None,
        )
        pool_size = len(hits)

        if request.min_similarity is not None:
            hits = [h for h in hits if h.similarity >= request.min_similarity]
        if request.min_probability is not None and query.modality == "text":
            hits = [
                h
                for h in hits
                if self.embedder.text_probability(h.similarity) >= request.min_probability
            ]

        # Counted before collapsing, so a collapsed card can still say how many crops in
        # its frame matched.
        per_frame: Dict[tuple, int] = {}
        for hit in hits:
            per_frame[(hit.qid, hit.frame_idx)] = per_frame.get((hit.qid, hit.frame_idx), 0) + 1

        if request.collapse == "frame":
            hits = _collapse_by_frame(hits)
        elif request.collapse == "time":
            hits = _collapse_by_time(hits, request.collapse_gap_ms)

        page = hits[:limit]
        results = [
            self._to_result(hit, request, query, per_frame.get((hit.qid, hit.frame_idx), 1))
            for hit in page
        ]

        logger.info(
            "{} query -> {} rows, {} after filtering, {} rendered",
            query.modality,
            pool_size,
            len(hits),
            len(results),
        )
        return SearchResponse(
            results=results,
            meta={
                "modality": query.modality,
                "pool": pool_size,
                "matched": len(hits),
                "count": len(results),
                "limit": limit,
                "track": request.track,
                "query_dim": query.raw_dim,
                # Which content objects the page touched, whether a token opened each, and
                # what every box source said. Without this a missing frame or a missing
                # box is indistinguishable from a bad query.
                "content": self._diagnostics(results, request),
            },
        )

    def _diagnostics(self, results: List[SearchResult], request: SearchRequest) -> List[Dict[str, Any]]:
        seen: Dict[str, Dict[str, Any]] = {}
        for result in results:
            if result.qid in seen:
                seen[result.qid]["results"] += 1
                continue
            resolution = self.tokens.resolve(
                result.qid, request.content_tokens or [request.token], request.explicit_tokens
            )
            on_row = box_from_info(result.additional_info) is not None
            seen[result.qid] = {
                "qid": result.qid,
                "results": 1,
                "token": resolution.ok,
                "token_reason": resolution.reason,
                "boxes": [{"source": "additional_info", "tags": 1, "detail": ""}]
                if on_row
                else self.boxes.status(
                    result.qid,
                    request.index_qid,
                    request.track or DETECTION_TRACK,
                    resolution.token or request.token,
                ),
            }
        return list(seen.values())

    def _to_result(
        self, hit: VectorHit, request: SearchRequest, query: QueryVector, crops_in_frame: int
    ) -> SearchResult:
        # Whichever supplied token opens *this* content object — not necessarily the one
        # that opened the index. A page routinely spans several objects.
        resolution = self.tokens.resolve(
            hit.qid, request.content_tokens or [request.token], request.explicit_tokens
        )
        frame_token = resolution.token
        error = resolution.reason

        seconds = frame_time(hit.start_time, hit.frame_idx, None)
        if seconds is None and frame_token:
            # Only reached on an index whose rows carry no start_time; costs one metadata
            # read per object, cached.
            fps = self.fabric.frame_rate(hit.qid, frame_token)
            seconds = frame_time(hit.start_time, hit.frame_idx, fps)

        track = hit.track or DETECTION_TRACK
        # The row's own box when the tagger stamped one; the join-back is for older indexes.
        box = box_from_info(hit.additional_info) or self.boxes.resolve(
            qid=hit.qid,
            track=track,
            frame_idx=hit.frame_idx,
            token=frame_token or request.token,
            index_qid=request.index_qid,
            tag_hint=hit.additional_info.get("tag"),
        )

        thumbnail = full = crop = link = None
        if seconds is not None and frame_token:
            common = {"qid": hit.qid, "token": frame_token, "t": f"{seconds:.3f}"}
            if box:
                common["box"] = f"{box.x1},{box.y1},{box.x2},{box.y2}"
            thumbnail = _frame_url({**common, "height": str(self.settings.thumbnail_height)})
            full = _frame_url(common)
            if box:
                crop = _frame_url({**common, "mode": "crop"})
            if request.evie_base:
                link = evie_url(request.evie_base, hit.qid, seconds)
        elif seconds is None:
            error = error or "no start_time or frame rate — cannot locate the frame"

        return SearchResult(
            qid=hit.qid,
            vector_id=hit.id,
            similarity=round(hit.similarity, 6),
            probability=(
                round(self.embedder.text_probability(hit.similarity), 6)
                if query.modality == "text"
                else None
            ),
            frame_idx=hit.frame_idx,
            start_time_ms=hit.start_time,
            seconds=round(seconds, 3) if seconds is not None else None,
            thumbnail_url=thumbnail,
            full_url=full,
            crop_url=crop,
            evie_url=link,
            box=box.as_dict() if box else None,
            additional_info=hit.additional_info,
            crops_in_frame=crops_in_frame,
            error=error,
        )


def _collapse_by_frame(hits: List[VectorHit]) -> List[VectorHit]:
    """Keep the best hit per (qid, frame_idx), preserving rank order."""
    best: Dict[tuple, VectorHit] = {}
    for hit in hits:  # already best-first
        best.setdefault((hit.qid, hit.frame_idx), hit)
    return list(best.values())


def _collapse_by_time(hits: List[VectorHit], gap_ms: int) -> List[VectorHit]:
    """Drop hits within `gap_ms` of an already-kept hit from the same content.

    Frames are sampled at roughly 1 fps, so consecutive frames of one shot are
    near-identical vectors and a page can otherwise spend every slot on neighbours of its
    own top hit. Hits with no start_time are kept — they have no timeline to compare on.
    """
    if gap_ms <= 0:
        return list(hits)
    kept: List[VectorHit] = []
    kept_times: Dict[str, List[int]] = {}
    for hit in hits:
        if hit.start_time is None:
            kept.append(hit)
            continue
        times = kept_times.setdefault(hit.qid, [])
        if any(abs(hit.start_time - t) < gap_ms for t in times):
            continue
        times.append(hit.start_time)
        kept.append(hit)
    return kept


def _frame_url(params: Dict[str, str]) -> str:
    """Point at this app's own proxy rather than at the fabric.

    Going direct is one hop fewer, but a denied frame then arrives as a JSON error body
    inside an `<img>`, which browsers block as an opaque response — the tile just goes
    blank with nothing in the console. Through the proxy every reply is an image.
    """
    return f"/api/frame?{urlencode(params)}"


def result_to_dict(result: SearchResult) -> Dict[str, Any]:
    return asdict(result)
