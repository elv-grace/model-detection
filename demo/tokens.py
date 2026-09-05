"""Which token opens which content object.

A vectorstore index spans many content objects, and a fabric auth token is scoped to one.
So a single-token demo can only render frames for a fraction of its own results — the rest
come back 403, and (because the fabric answers with a JSON error body) a browser's Opaque
Response Blocking rejects them inside an `<img>`, leaving a silently blank tile.

The fix is to accept *several* tokens and work out which one covers each object. Tokens are
opaque — signed binary, no readable qid inside — so the mapping cannot be derived and has
to be probed: `GET /q/{qid}` with each candidate until one answers 200. That is one cheap
request per (qid, token) pair, cached for the life of the process, and a page of results
usually spans a handful of objects.

A caller may also state the mapping outright as `iq__... = token`, which skips probing.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import requests
from loguru import logger


@dataclass(frozen=True)
class TokenResolution:
    """The outcome of looking for a token that opens one content object."""

    qid: str
    token: Optional[str]
    # Human-readable reason when `token` is None, shown on the result card rather than
    # left as an unexplained blank tile.
    reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.token is not None


def parse_tokens(raw: Optional[str]) -> Tuple[List[str], Dict[str, str]]:
    """Split a textarea into (candidate tokens, explicit qid -> token).

    Accepts one entry per line, either a bare token or `iq__... = token`. Blank lines and
    `#` comments are ignored so a list can be annotated.
    """
    candidates: List[str] = []
    explicit: Dict[str, str] = {}
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            qid, _, token = line.partition("=")
            qid, token = qid.strip(), token.strip()
            if qid and token:
                explicit[qid] = token
                candidates.append(token)
                continue
        candidates.append(line)
    return candidates, explicit


class TokenRegistry:
    """Resolves a usable token per content object, probing and caching as it goes."""

    def __init__(self, node_getter, timeout: float = 30.0) -> None:
        # A callable rather than a URL so the fabric node is resolved lazily and shared
        # with FabricClient instead of being fetched twice.
        self._node = node_getter
        self.timeout = timeout
        self._resolved: Dict[str, str] = {}
        # Remembering failures matters as much as remembering successes: without it every
        # result from an unreachable object re-probes every token on every search.
        self._failed: Dict[str, str] = {}
        # Which candidate set produced the cached failure, so that supplying a new token
        # retries instead of returning the stale "nothing opens this" answer.
        self._tried: Dict[str, Tuple[str, ...]] = {}
        self._lock = threading.Lock()

    def resolve(
        self,
        qid: str,
        candidates: Sequence[str],
        explicit: Optional[Dict[str, str]] = None,
    ) -> TokenResolution:
        explicit = explicit or {}
        if qid in explicit:
            return TokenResolution(qid, explicit[qid])

        with self._lock:
            if qid in self._resolved:
                token = self._resolved[qid]
                # Only trust the cache while the token is still on offer; a caller that
                # removed it should stop getting results rendered with it.
                if token in candidates:
                    return TokenResolution(qid, token)
                del self._resolved[qid]
            cached_failure = self._failed.get(qid)

        untried = [t for t in candidates if t]
        if cached_failure is not None and self._same_candidates(qid, untried):
            return TokenResolution(qid, None, cached_failure)

        last_reason = "no token supplied"
        for token in untried:
            ok, reason = self._probe(qid, token)
            if ok:
                with self._lock:
                    self._resolved[qid] = token
                    self._failed.pop(qid, None)
                    self._tried[qid] = tuple(untried)
                return TokenResolution(qid, token)
            last_reason = reason

        reason = f"no supplied token opens {qid} ({last_reason})"
        with self._lock:
            self._failed[qid] = reason
            self._tried[qid] = tuple(untried)
        return TokenResolution(qid, None, reason)

    def _same_candidates(self, qid: str, candidates: Sequence[str]) -> bool:
        with self._lock:
            return self._tried.get(qid) == tuple(candidates)

    def _probe(self, qid: str, token: str) -> Tuple[bool, str]:
        """`GET /q/{qid}` — read access only, unlike `/qid/{qid}` which needs versions."""
        try:
            response = requests.get(
                f"{self._node()}/q/{qid}", params={"authorization": token}, timeout=self.timeout
            )
        except Exception as exc:
            return False, f"fabric unreachable: {exc}"
        if response.ok:
            logger.debug("token …{} opens {}", token[-8:], qid)
            return True, "ok"
        return False, f"HTTP {response.status_code}"
