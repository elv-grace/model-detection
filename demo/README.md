# demo — search GUI for the detection vector index

A single-page tool for answering one question by eye: **given a text phrase or an image,
does the index return crops that actually match?**

Text or an image goes in, gets embedded by the same SigLIP 2 checkpoint the tagger indexed
crops with, is sent to the [vectorstore search
API](https://docs.eluv.io/api/vectorstore/vectors/vectorstore-search/), and each hit comes
back as the video frame it was detected in, fetched from the fabric, so a page of results can
be judged in one pass.

```
text  ──► Siglip2Model.get_text_features  ─┐
                                           ├─► pad 768→1024 ─► POST /indexes/{qid}/search
image ──► crop ─► get_image_features      ─┘                             │
                                                                          ▼
                          frame card  ◄─  rep/frame_extract  ◄─  (qid, start_time, frame_idx)
```

## Run

```bash
pip install -r demo/requirements.txt
uvicorn demo.app:app --host 0.0.0.0 --port 8300      # from the repo root
```

Open <http://localhost:8300>, paste an index qid and an auth token, press **Connect**, then
search. The checkpoint loads lazily on the first query (~15 s), not at startup.

Settings that are not secrets-in-transit are read from the environment:

| variable | default | |
|---|---|---|
| `VECTORSTORE_URL` | `http://localhost:8108` | |
| `TAGSTORE_URL` | `http://localhost:8102` | where boxes are expected to appear |
| `FABRIC_CONFIG_URL` | `https://main.net955305.contentfabric.io/config` | |
| `INDEX_VECTOR_SIZE` | `1024` | index width to pad queries to |
| `INDEX_CROP_PADDING` | `0.06` | padding the indexed crops were taken with |
| `SIGLIP_MODEL_ID` / `SIGLIP_REVISION` / `SIGLIP_MAX_NUM_PATCHES` | naflex base, pinned | |

## Tokens: one per content object, not one per app

A fabric auth token is scoped to **one content object**, and a vectorstore index spans
several — the index used for testing holds vectors from three. A single token therefore
renders only a fraction of its own results, and the rest fail in a way that is invisible:
the fabric answers `403` with a JSON error body, and a browser refuses to render JSON inside
an `<img>` (Chrome's Opaque Response Blocking, `ERR_BLOCKED_BY_ORB`) — no console error, just
a blank tile. That was the original "frames not showing" bug.

So the **Content tokens** box takes one token per line, either bare or as
`iq__... = token`. Tokens are opaque — signed binary with no readable qid inside, checked —
so a bare token has its object probed with `GET /q/{qid}` and the mapping is cached
([`tokens.py`](tokens.py)). Supplying tokens for two of the three objects took a 50-result
page from 10 rendered frames to 50.

Anything still unreachable renders as a **labelled placeholder image** saying which object
and why, and the sidebar names the objects that need a token. Verified: the same token
authorizes the vectorstore, the tagstore, `GET /q/{qid}` and `rep/frame_extract` for the
object it covers.

> Do not call `GET /qid/{qid}` — it needs `q.read.versions`, which a content-scoped token
> generally lacks. `GET /q/{qid}` needs only read and returns the same version hash.

## Seeing which crop matched

With no dedupe, several results legitimately share one `frame_idx`, and as whole frames they
are pixel-identical — there is nothing to judge. Given a box, each card gets a **Show
detection** toggle that swaps the whole frame for just that detection, cropped from the
*full-resolution* frame (a 22 px mark inside an already-downscaled thumbnail is a handful of
pixels), drawn with its box, kept with 35% context so a wordmark is recognisable, and scaled
up to at least 240 px. The box is rendered into the image server-side rather than overlaid in
CSS, because a percentage overlay only lines up while the frame's aspect ratio matches the
tile's and cannot follow the image into the cropped view.

### Where the box comes from

`model-detection` stamps the geometry into `additional_info.box` as well as
`Tag.frame_info.box`, and `additional_info` is the one thing a vectorstore search row returns —
so on an index built with a current tagger the box arrives with the vector and no lookup
happens at all.

On **older indexes** the row carries provenance and no geometry, and the box has to be joined
back. Three sources are tried per result ([`boxes.py`](boxes.py)), and all three were checked
with tokens that **fully open** the objects — this is a property of the data, not a permissions
problem:

| source | result |
|---|---|
| vectorstore search response | `additional_info` **is** returned (prompt, score, detector, crop_padding, upscale). No geometry on an index built before `additional_info.box`. |
| tagstore, content qid | 200, `{"tracks": []}` for every indexed object |
| tagstore, index qid | 17 tracks, but they belong to other taggers (`logo_detection`, `object_detection`, `celebrity_detection`) — no `detection` track, 0 tags |
| fabric `video_tags` (what EVIE reads) | `{}`, with and without link resolution |
| the vector's `source` part | `403` — part reads need `q.read.files.read` |

On such an index results render as whole frames, and the UI says which sources it tried and
what each replied rather than leaving the absence unexplained. `resolve` never raises: a box is an
enhancement, and losing it must not lose the result.

## Design notes

**Query vectors have to land in the index's space, and nothing downstream would say if they
did not.** A query embedded with the wrong checkpoint or patch budget still returns a
confidently ranked page; it is just the wrong page. So **Connect** reads the index's own
`additional_info` back off a throwaway search and diffs it against this process's config,
warning in the sidebar on any mismatch of embedder, `max_num_patches`, `crop_padding` or
width.

**The full `Siglip2Model` is loaded, not the two towers.** In transformers, SigLIP 2 has no
projection head — `get_image_features` *is* `vision_model(...)` and `get_text_features` *is*
`text_model(...)` — so the pooled outputs are identical to what
[`general_detection/embedder.py`](../general_detection/embedder.py) stores, while
`logit_scale`/`logit_bias` come along for the calibration below. (CLIP projects both towers,
so this reasoning does not transfer to a CLIP checkpoint.)

**Text and image scores are shown differently, because they are not comparable.** The
modality gap puts a perfect text match near cos 0.05–0.3 where a good image match sits at
0.5–0.9 — measured here at 0.14–0.16 for text and 0.69–0.77 for image against the same
index. One threshold cannot serve both, so text results are scored by SigLIP 2's calibrated
sigmoid probability (`sigmoid(cos · logit_scale.exp() + logit_bias)`), which is comparable
across queries, and image results by raw cosine.

**An image query is padded to the index's `crop_padding` by default.** Indexed crops carry
6% context on each side, so an object fills 89% of the crop's linear extent. A tightly drawn
query box has different composition, and the README's measurement puts a ±0.06 mismatch at
8% of top-1 retrievals lost. The checkbox turns it off for comparison.

**Frames are timed from `start_time`, not `frame_idx / fps`.** Both are stamped by
`common_ml` and agree exactly on well-formed data (frame 164735 at 6870822 ms implies
23.9760 fps against a declared 24000/1001), but `start_time` needs no frame rate at all.
Reconstructing time from `frame_idx` means guessing one, and EVIE's 24000/1001 fallback is
silently wrong on 25 or 30 fps content — it does not error, it extracts a different frame.

**Nothing is deduped by frame unless you ask.** Frame search collapses `(qid, frame_idx)`
because a frame contributes one vector; a detection index contributes up to
`max_detections` per frame, and several logos in one frame are several real findings.
Default is `none`; a card shows an *n* crops badge when it shares a frame with others.
`frame` and `time` collapsing are opt-in from the sidebar.

**Thumbnails are resized server-side.** `rep/frame_extract` honours `height`, which turns a
640 KB 4K frame into ~50 KB — a 50-result page goes from 30 MB to 2.5 MB. Full resolution is
fetched only when a card is opened or cropped.

**Frames are proxied through `/api/frame`, not loaded straight from the fabric.** Going
direct is one hop fewer and was the first design, but it makes a denied frame indistinguishable
from a broken app (see ORB, above) and leaves no place to draw a box or cut a crop. The proxy
always answers with an image, caches the source frame in a small LRU so switching a card
between whole-frame and crop does not re-download it, and sets `Cache-Control` because a frame
is immutable for a given `(qid, t)`.

## Layout

| file | |
|---|---|
| [`app.py`](app.py) | FastAPI routes: `/api/index_info`, `/api/search/{text,image}`, `/api/frame` |
| [`embedder.py`](embedder.py) | SigLIP 2 text + image towers, padding, sigmoid calibration |
| [`vectorstore.py`](vectorstore.py) | `/indexes/{qid}` · `/tracks` · `/search` |
| [`fabric.py`](fabric.py) | object resolution, frame URLs, `frame_idx`↔time |
| [`tokens.py`](tokens.py) | which token opens which content object |
| [`frames.py`](frames.py) | fetch, box overlay, crop isolation, failure placeholders |
| [`boxes.py`](boxes.py) | box lookup across three sources (inert until tags are written) |
| [`search.py`](search.py) | embed → search → filter → collapse → enrich |
| [`static/`](static/) | one HTML page, one stylesheet, one script — no build step |

## Testing the UI

Forward port 8300 and open localhost:8300.  
There is no test suite; the UI was driven headlessly with Playwright
(`pip install playwright && python -m playwright install chromium`) to catch exactly the
class of bug that leaves no console error — the ORB-blocked frames were invisible until a
real browser loaded the page and reported `naturalWidth == 0`. Worth reaching for again
before trusting a change to the grid.

Much of the query side mirrors `content-search`'s `frame_search` module, which solves the
same problem for whole-frame vectors. It is duplicated rather than imported so the demo runs
from this repo alone; if the two drift, `content-search` is the one that ships.
