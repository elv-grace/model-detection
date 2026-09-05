"""FastAPI app for the model-detection search demo.

Run from the repository root:

    pip install -r demo/requirements.txt
    uvicorn demo.app:app --host 0.0.0.0 --port 8300

The index qid and auth token are supplied per request by the browser, so one process can
serve any index. The SigLIP 2 checkpoint is loaded once, lazily, on the first query.
"""
from __future__ import annotations

import base64
import io
import json
import threading
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pathlib import Path
from pydantic import BaseModel, Field

from .boxes import BoxResolver
from .config import settings
from .embedder import Siglip2QueryEmbedder, pad_crop
from .fabric import FabricClient
from .frames import FrameRenderer, parse_box, placeholder
from .search import DETECTION_TRACK, SearchRequest, SearchService, result_to_dict
from .tokens import TokenRegistry, parse_tokens
from .vectorstore import VectorStoreClient

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="model-detection search demo", docs_url="/api/docs")

store = VectorStoreClient(settings.vectorstore_url, settings.http_timeout)
fabric = FabricClient(settings.fabric_config_url, settings.offering, settings.http_timeout)
boxes = BoxResolver(settings.tagstore_url, fabric, settings.http_timeout)
# The node getter is a callable so the registry shares FabricClient's lazily-resolved node
# instead of fetching the network config a second time.
tokens = TokenRegistry(lambda: fabric.node, settings.http_timeout)
renderer = FrameRenderer(fabric, settings.http_timeout)

_embedder: Optional[Siglip2QueryEmbedder] = None
_embedder_lock = threading.Lock()


def get_service() -> SearchService:
    """Load the checkpoint on first use so startup is instant and `--reload` stays usable."""
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            _embedder = Siglip2QueryEmbedder(
                model_id=settings.model_id,
                revision=settings.revision,
                max_num_patches=settings.max_num_patches,
                normalize=settings.normalize,
                target_size=settings.target_size,
            )
    return SearchService(settings, _embedder, store, fabric, boxes, tokens)


# --------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------


class TextSearchBody(BaseModel):
    index_qid: str
    token: str
    # Additional tokens for the content objects the index spans. A token is scoped to one
    # object, so without these most frames in a multi-object index cannot be fetched.
    # Accepts bare tokens (the qid each opens is probed) or `iq__... = token` lines.
    content_tokens: str = ""
    query: str
    limit: int = settings.default_limit
    track: Optional[str] = DETECTION_TRACK
    qids: List[str] = Field(default_factory=list)
    min_similarity: Optional[float] = None
    min_probability: Optional[float] = None
    collapse: str = "none"
    collapse_gap_ms: int = 1000
    evie_base: Optional[str] = None


# --------------------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "model_loaded": _embedder is not None,
        "model_id": settings.model_id,
        "revision": settings.revision,
        "vectorstore": settings.vectorstore_url,
        "tagstore": settings.tagstore_url,
    }


@app.get("/api/index_info")
def index_info(index_qid: str, token: str) -> Dict[str, Any]:
    """Index width, tracks, and a parity check against how the demo will embed queries.

    The check is the point of this route. A query embedded with a different checkpoint,
    patch budget or padding than the index was built with still returns a confidently
    ranked page — it is simply the wrong page, with nothing in the output to say so. So
    the index's own `additional_info` is read back from one throwaway search and compared
    against this process's configuration.
    """
    try:
        info = store.index_info(index_qid, token)
        tracks = store.tracks(index_qid, token)
    except requests.HTTPError as exc:
        raise _downstream_error(exc) from exc

    vector_size = info.get("vector_size")
    probe: Dict[str, Any] = {}
    warnings: List[str] = []

    track_names = [t.get("name") for t in tracks]
    track = DETECTION_TRACK if DETECTION_TRACK in track_names else (track_names[0] if track_names else None)

    if vector_size and track:
        try:
            # An arbitrary unit-ish vector: the ranking is meaningless, the metadata is not.
            sample = store.search(
                index_qid, token, [1.0 / (vector_size ** 0.5)] * vector_size, limit=1, track=track
            )
            if sample:
                probe = sample[0].additional_info or {}
        except requests.HTTPError as exc:
            logger.warning("probe search failed: {}", exc)

    if probe:
        if probe.get("embedder") and probe["embedder"] != settings.model_id:
            warnings.append(
                f"index was built with {probe['embedder']} but this demo embeds queries with "
                f"{settings.model_id} — scores are not comparable"
            )
        if probe.get("max_num_patches") and probe["max_num_patches"] != settings.max_num_patches:
            warnings.append(
                f"index used max_num_patches={probe['max_num_patches']}, demo uses "
                f"{settings.max_num_patches} — image queries will land off-distribution"
            )
        if probe.get("dim") and vector_size and probe["dim"] > vector_size:
            warnings.append(f"index reports {probe['dim']}-d vectors in a {vector_size}-d space")
        padding = probe.get("crop_padding")
        if padding is not None and abs(float(padding) - settings.index_crop_padding) > 1e-9:
            warnings.append(
                f"index crops used crop_padding={padding}, demo pads image queries to "
                f"{settings.index_crop_padding} — set INDEX_CROP_PADDING to match"
            )
    if vector_size and vector_size != settings.target_size:
        warnings.append(
            f"index is {vector_size}-d but the demo pads queries to {settings.target_size}-d; "
            f"set INDEX_VECTOR_SIZE={vector_size}"
        )

    return {
        "index_qid": index_qid,
        "vector_size": vector_size,
        "tracks": tracks,
        "selected_track": track,
        "index_provenance": probe,
        "demo_config": {
            "model_id": settings.model_id,
            "revision": settings.revision,
            "max_num_patches": settings.max_num_patches,
            "target_size": settings.target_size,
            "crop_padding": settings.index_crop_padding,
        },
        "warnings": warnings,
    }


@app.post("/api/search/text")
def search_text(body: TextSearchBody) -> JSONResponse:
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must be non-empty")

    service = get_service()
    try:
        vector = service.embedder.embed_text(query)
        response = service.search(_to_request(body), vector)
    except requests.HTTPError as exc:
        raise _downstream_error(exc) from exc

    payload = {
        "results": [result_to_dict(r) for r in response.results],
        "meta": {**response.meta, "query": query},
    }
    return JSONResponse(payload)


@app.post("/api/search/image")
def search_image(
    file: UploadFile = File(...),
    index_qid: str = Form(...),
    token: str = Form(...),
    content_tokens: str = Form(""),
    limit: int = Form(settings.default_limit),
    track: Optional[str] = Form(DETECTION_TRACK),
    qids: str = Form(""),
    min_similarity: Optional[float] = Form(None),
    collapse: str = Form("none"),
    collapse_gap_ms: int = Form(1000),
    evie_base: Optional[str] = Form(None),
    # Fractional (x, y, w, h) drag-select from the browser. Fractions rather than pixels
    # so the crop is independent of how the image was displayed.
    crop: Optional[str] = Form(None),
    # Whether to expand the drawn box to the index's crop_padding. On by default because
    # the indexed crops all carry that context and a tight query box does not.
    match_index_padding: bool = Form(True),
) -> JSONResponse:
    service = get_service()

    crop_box = _parse_crop(crop)
    if crop_box and match_index_padding:
        crop_box = pad_crop(crop_box, settings.index_crop_padding)

    try:
        vector, embedded = service.embedder.embed_image(io.BytesIO(file.file.read()), crop_box)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not decode image: {exc}") from exc

    body = TextSearchBody(
        index_qid=index_qid,
        token=token,
        content_tokens=content_tokens,
        query="",
        limit=limit,
        track=track,
        qids=[q for q in (qids or "").split(",") if q.strip()],
        min_similarity=min_similarity,
        collapse=collapse,
        collapse_gap_ms=collapse_gap_ms,
        evie_base=evie_base,
    )
    try:
        response = service.search(_to_request(body), vector)
    except requests.HTTPError as exc:
        raise _downstream_error(exc) from exc

    payload = {
        "results": [result_to_dict(r) for r in response.results],
        "meta": {
            **response.meta,
            # What was actually embedded, after cropping and padding — so the UI can show
            # the query beside the results instead of the original upload.
            "query_image": _thumbnail_data_url(embedded),
            "crop_applied": list(crop_box) if crop_box else None,
        },
    }
    return JSONResponse(payload)


@app.get("/api/frame")
def frame(
    qid: str,
    token: str,
    t: float,
    mode: str = Query("full", pattern="^(full|crop)$"),
    box: Optional[str] = None,
    height: Optional[int] = None,
) -> Response:
    """Serve one frame as an image — always an image, including on failure.

    This proxies rather than letting the browser hit the fabric directly, because a denied
    frame comes back as a JSON error body and a browser refuses to render that inside an
    `<img>` (Chrome logs `ERR_BLOCKED_BY_ORB` and shows nothing). A labelled placeholder
    that says "HTTP 403" is far more useful than a blank rectangle.

    `mode=crop` needs `box` and returns just that detection, scaled up with a little
    context — the only way to tell apart several results that share one frame.
    """
    parsed = parse_box(box)
    try:
        data, content_type = renderer.render(
            qid=qid, token=token, seconds=t, mode=mode, box=parsed, height=height
        )
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        reason = "token does not open this content object" if status == 403 else f"HTTP {status}"
        logger.warning("frame {} @ {}s failed: {}", qid, t, reason)
        data, content_type = placeholder(f"{qid}\n{reason}"), "image/jpeg"
    except Exception as exc:
        logger.warning("frame {} @ {}s failed: {}", qid, t, exc)
        data, content_type = placeholder(f"{qid}  {exc}"), "image/jpeg"

    return Response(
        content=data,
        media_type=content_type,
        # Frames are immutable for a given (qid, t), and a grid re-requests them on every
        # re-render, so let the browser keep them.
        headers={"Cache-Control": "private, max-age=3600"},
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _to_request(body: TextSearchBody) -> SearchRequest:
    candidates, explicit = parse_tokens(body.content_tokens)
    # The index token is tried last rather than first: it is the one least likely to open
    # a *content* object, and probing costs a round trip per attempt.
    if body.token and body.token not in candidates:
        candidates.append(body.token)
    return SearchRequest(
        index_qid=body.index_qid,
        token=body.token,
        content_tokens=candidates,
        explicit_tokens=explicit,
        limit=body.limit,
        track=body.track or None,
        qids=body.qids,
        min_similarity=body.min_similarity,
        min_probability=body.min_probability,
        collapse=body.collapse,
        collapse_gap_ms=body.collapse_gap_ms,
        evie_base=body.evie_base,
    )


def _parse_crop(raw: Optional[str]):
    if not raw:
        return None
    try:
        values = json.loads(raw)
        x, y, w, h = (float(v) for v in values)
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)


def _thumbnail_data_url(image, max_side: int = 320) -> str:
    preview = image.copy()
    preview.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _downstream_error(exc: requests.HTTPError) -> HTTPException:
    """Forward a vectorstore or fabric failure with its own status and body.

    A 403 from the store means the token does not cover that index, and a 404 means the
    qid is not an index at all. Both are things the user can fix, so the detail is passed
    through rather than flattened into a generic 502.
    """
    response = exc.response
    status = response.status_code if response is not None else 502
    detail: Any = str(exc)
    if response is not None:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
    return HTTPException(status_code=status, detail=detail)
