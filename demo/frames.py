"""Fetch a frame, and render it the way a result wants to be looked at.

Frames used to be loaded straight from the fabric by the browser. That is one fewer hop,
but it breaks in two ways this module exists to fix:

1. **A 403 renders as a blank tile.** A token is scoped to one content object and an index
   spans several, so some results always come back denied. The fabric answers with a JSON
   error body, and Chrome's Opaque Response Blocking refuses a JSON payload delivered to an
   `<img>` — `net::ERR_BLOCKED_BY_ORB`, no console error, nothing on screen. Proxying means
   every request answers with an *image*, including the failures, which say why on the tile.

2. **A whole frame cannot show which crop matched.** With no dedupe, several results can
   share one frame_idx and look identical. Given a box, `crop` mode returns just that
   region and the results become distinguishable.
"""
from __future__ import annotations

import io
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple

import requests
from loguru import logger
from PIL import Image, ImageDraw

# Fraction of the box's own size kept around it in crop mode. A wordmark cropped to its
# exact bounds is unreadable out of context — you cannot tell a jersey badge from a
# hoarding — and this is for judging by eye, not for embedding, so context is free here.
CROP_CONTEXT = 0.35

# Smallest crop returned, in px. Detected marks have a median short side around 22 px;
# shown at native size they are invisible in a grid, so small crops are scaled up.
MIN_CROP_SIDE = 240


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


class FrameRenderer:
    """Fetches frames from the fabric and returns JPEG bytes, with a small LRU cache.

    The cache is keyed on the *source* request, not the rendered output, so switching a
    card between whole-frame and box-crop does not re-download the frame.
    """

    def __init__(self, fabric, timeout: float = 60.0, cache_size: int = 64) -> None:
        self.fabric = fabric
        self.timeout = timeout
        self.cache_size = cache_size
        self._cache: "OrderedDict[Tuple[str, float, Optional[int]], bytes]" = OrderedDict()
        self._lock = threading.Lock()

    def render(
        self,
        qid: str,
        token: str,
        seconds: float,
        mode: str = "full",
        box: Optional[Box] = None,
        height: Optional[int] = None,
        draw_box: bool = True,
    ) -> Tuple[bytes, str]:
        """Return (jpeg bytes, content type). Raises only on a genuinely unexpected error."""
        if mode == "crop" and box is not None:
            # Crop from the full-resolution frame: a 22 px mark inside a frame already
            # downscaled to 360 px high is a handful of pixels and nothing survives.
            source = self._fetch(qid, token, seconds, None)
            return _encode(_crop_to_box(Image.open(io.BytesIO(source)), box)), "image/jpeg"

        source = self._fetch(qid, token, seconds, height)
        if box is not None and draw_box:
            return _encode(_draw_box(Image.open(io.BytesIO(source)), box)), "image/jpeg"
        # Untouched bytes when there is nothing to draw — no decode/re-encode round trip.
        return source, "image/jpeg"

    def _fetch(self, qid: str, token: str, seconds: float, height: Optional[int]) -> bytes:
        key = (qid, round(seconds, 3), height)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        url = self.fabric.frame_url(qid, token, seconds, height=height)
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        data = response.content

        with self._lock:
            self._cache[key] = data
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return data


def placeholder(message: str, width: int = 640, height: int = 360) -> bytes:
    """A labelled tile, so a failure reads as a reason instead of a blank rectangle."""
    image = Image.new("RGB", (width, height), (18, 17, 26))
    draw = ImageDraw.Draw(image)
    draw.rectangle([(0, 0), (width - 1, height - 1)], outline=(60, 56, 80))

    # Wrapped by character count rather than measured text: the default bitmap font is
    # fixed-width, and a demo placeholder does not justify loading a TTF.
    words, lines, line = message.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > 46:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    lines.append(line)
    lines = lines[:6]

    y = height // 2 - len(lines) * 7
    for text in lines:
        draw.text((width // 2 - len(text) * 3, y), text, fill=(150, 145, 175))
        y += 14
    return _encode(image)


def _crop_to_box(image: Image.Image, box: Box) -> Image.Image:
    width, height = image.size
    x1, y1 = box.x1 * width, box.y1 * height
    x2, y2 = box.x2 * width, box.y2 * height
    pad_x, pad_y = (x2 - x1) * CROP_CONTEXT, (y2 - y1) * CROP_CONTEXT

    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(width, int(x2 + pad_x))
    bottom = min(height, int(y2 + pad_y))
    if right <= left or bottom <= top:
        return image

    cropped = image.crop((left, top, right, bottom))
    # The box is drawn *after* cropping, so the exact detection bounds stay visible inside
    # the context margin rather than being cropped away with it.
    inner = Box(
        (x1 - left) / cropped.width,
        (y1 - top) / cropped.height,
        (x2 - left) / cropped.width,
        (y2 - top) / cropped.height,
    )
    cropped = _draw_box(cropped, inner, width_px=max(1, cropped.width // 200))

    scale = MIN_CROP_SIDE / min(cropped.size)
    if scale > 1:
        # NEAREST would show the pixel grid, which reads as detail that is not there.
        cropped = cropped.resize(
            (int(cropped.width * scale), int(cropped.height * scale)), Image.LANCZOS
        )
    return cropped


def _draw_box(image: Image.Image, box: Box, width_px: Optional[int] = None) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    coords = [
        (box.x1 * width, box.y1 * height),
        (box.x2 * width, box.y2 * height),
    ]
    line = width_px or max(2, width // 400)
    # Dark outline under the accent so the box stays visible on light content too.
    draw.rectangle(coords, outline=(10, 8, 16), width=line + 2)
    draw.rectangle(coords, outline=(139, 123, 247), width=line)
    return image


def _encode(image: Image.Image, quality: int = 88) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def parse_box(raw: Optional[str]) -> Optional[Box]:
    """`x1,y1,x2,y2` normalized, as it travels in a query string."""
    if not raw:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in raw.split(","))
    except (ValueError, TypeError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return Box(x1, y1, x2, y2)
