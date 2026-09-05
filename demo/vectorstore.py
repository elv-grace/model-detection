"""Thin client for the vectorstore HTTP API.

Mirrors `content-search`'s `HttpVectorStoreClient` for the two calls this demo makes, plus
`tracks`, which content-search does not use but which is how the UI discovers that an index
holds a "detection" track at all.

  GET  /indexes/{qid}          -> {"qid", "vector_size"}
  GET  /indexes/{qid}/tracks   -> {"tracks": [{"name", "count"}]}
  POST /indexes/{qid}/search   -> {"results": [{"distance", "vector": {...}}]}

https://docs.eluv.io/api/vectorstore/vectors/vectorstore-search/
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
from loguru import logger


@dataclass(frozen=True)
class VectorHit:
    """One row of a search response.

    Note there is no bounding box *column*: the vectorstore's embedding schema has none, so
    the geometry arrives inside `additional_info`, where `model-detection` stamps it. On an
    index built before that it is absent and has to be joined back — see boxes.py.
    """

    id: int
    qid: str
    track: str
    similarity: float
    frame_idx: Optional[int] = None
    start_time: Optional[int] = None  # ms
    end_time: Optional[int] = None  # ms
    source: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)


class VectorStoreClient:
    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def index_info(self, index_qid: str, token: str) -> Dict[str, Any]:
        return self._get(f"/indexes/{index_qid}", token)

    def tracks(self, index_qid: str, token: str) -> List[Dict[str, Any]]:
        return self._get(f"/indexes/{index_qid}/tracks", token).get("tracks", [])

    def search(
        self,
        index_qid: str,
        token: str,
        vector: List[float],
        limit: int,
        track: Optional[str] = None,
        qids: Optional[List[str]] = None,
    ) -> List[VectorHit]:
        body: Dict[str, Any] = {"vector": vector, "limit": limit}
        if track:
            # An index can hold more than one track, and whole-frame vectors and crop
            # vectors are different enough distributions that ranking them against each
            # other is meaningless. Filtering keeps one query inside one space.
            body["track"] = track
        if qids:
            body["qids"] = qids

        response = requests.post(
            f"{self.base_url}/indexes/{index_qid}/search",
            json=body,
            headers=self._headers(token),
            timeout=self.timeout,
        )
        response.raise_for_status()
        results = (response.json() or {}).get("results", [])
        logger.info("vectorstore returned {} rows (limit={})", len(results), limit)
        return [_to_hit(row) for row in results]

    def _get(self, path: str, token: str) -> Dict[str, Any]:
        response = requests.get(
            f"{self.base_url}{path}", headers=self._headers(token), timeout=self.timeout
        )
        response.raise_for_status()
        return response.json() or {}

    @staticmethod
    def _headers(token: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }


def _to_hit(row: Dict[str, Any]) -> VectorHit:
    vector = row.get("vector") or {}
    return VectorHit(
        id=vector.get("id"),
        qid=vector.get("qid"),
        track=vector.get("track", ""),
        # The store returns cosine *distance*; everything user-facing here is similarity.
        similarity=1.0 - float(row.get("distance", 1.0)),
        frame_idx=vector.get("frame_idx"),
        start_time=vector.get("start_time"),
        end_time=vector.get("end_time"),
        source=vector.get("source"),
        additional_info=vector.get("additional_info") or {},
    )
