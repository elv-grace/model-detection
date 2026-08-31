#!/usr/bin/env python3
"""Freeze a deterministic evaluation frame set, and generate the presence-labelling tool.

Why freeze frames at all
------------------------
Detectors must be compared on identical pixels. Running each one over the videos re-decodes
them and lets fps/keyframe choices drift between runs, so differences in the results stop
being attributable to the detector. Extracting once to files removes that variance, makes the
set reproducible from the manifest, and is far cheaper — it gets 30 frames of NBA33min.mp4
into the evaluation instead of decoding 2.25 GB per detector.

Why the long side is capped (--max-side, default 1920)
------------------------------------------------------
nfl.jpg is 6243x4376. Two things go wrong if frames that large stay native:

  * Detector: at imgsz=640 the frame is pre-downscaled ~0.10x, so a 60px logo arrives at the
    detector as ~6px and is effectively invisible. The same logo in a 1080p frame arrives at
    ~20px. Left uncapped, the comparison partly measures how each detector copes with a 10x
    pre-downscale rather than how well it detects.
  * Embedder: a full-height crop of a 4376px frame is ~4000px tall, which NaFlex resizes to
    ~256px -- a 0.06x scale (observed exactly this, upscale=0.06). No 1080p frame ever
    produces that; there a person crop is 400-900px (~0.3-0.6x) and a logo crop 40-80px
    (~3-6x upscale).

Capping normalises the whole set to broadcast resolution so both axes are comparable across
frames and results transfer to production content. The trade-off is that genuine detail a
higher imgsz could exploit is discarded, so if high-resolution stills are a real production
input, build a second set with --max-side 0 (no cap) and evaluate it separately.

Outputs
-------
    eval/frames/<id>.png        full-res frames, capped at --max-side; what detectors run on
    eval/thumbs/<id>.jpg        downscaled; what the labelling UI displays
    eval/frames.json            manifest: source file, timestamp, frame index
    eval/label_presence.html    self-contained labelling tool, manifest inlined

Usage
-----
    python eval/freeze_frames.py                # extract frames + build the tool
    python eval/freeze_frames.py --html-only    # rebuild the tool after editing CLASSES
    python eval/freeze_frames.py --max-side 0   # keep native resolution
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TEST_FILES = os.path.join(REPO, "test-files")

# How many frames to take from each source. Totals 100. Weighted by duration and by how much
# distinct content each file contributes, not by file size.
FRAME_BUDGET: Dict[str, int] = {
    "NBA33min.mp4": 30,
    "sn1.mp4": 18,
    "bd2.mp4": 14,
    "bd1.mp4": 12,
    "1.mp4": 8,
    "spiderman30s.mp4": 8,
    "NBAallstar10s.mp4": 6,
    # stills contribute one frame each
    "nfl.jpg": 1,
    "nfl2.jpg": 1,
    "chalkboard.png": 1,
    "cupcakes.png": 1,
}

# The presence-labelling schema.
#
# Every `prompts` entry is an exact term in YOLOE's prompt-free vocabulary (4585 classes), so
# each class is expressible as a text prompt for YOLOE and YOLO-World and is checkable against
# the prompt-free head. Terms absent from that vocabulary ("writing surface", "brand mark",
# "packaged good") are what produced the near-zero scores in the first calibration run.
#
# EDIT THIS BEFORE LABELLING, NOT AFTER — changing it invalidates labels already collected.
#
# Deliberately absent: a generic-object catch-all. "Is a generic object present?" is always
# true, so recall is undefined and presence labels carry no signal. Generic-object performance
# is evaluated instead by precision@K on contact sheets and by vocabulary diversity (how many
# distinct vocabulary terms fire, and whether they are plausible). `ball` and `bottle_or_cup`
# serve as concrete, measurable stand-ins for that bucket.
CLASSES: List[Dict] = [
    {"key": "logo", "label": "Logo / brand mark", "hint": "any brand mark, wordmark, or emblem",
     "prompts": ["logo", "car logo", "letter logo"]},
    {"key": "person", "label": "Person", "hint": "any visible person, including crowd",
     "prompts": ["person"]},
    {"key": "sign_or_text", "label": "Sign / text",
     "hint": "signage, billboard, banner, scoreboard text, captions",
     "prompts": ["text", "sign", "billboard", "banner", "scoreboard"]},
    {"key": "screen", "label": "Screen / monitor",
     "hint": "TV, monitor, phone or laptop screen, jumbotron",
     "prompts": ["screen", "monitor", "television", "computer screen"]},
    {"key": "board", "label": "Writing board", "hint": "whiteboard, blackboard, chalkboard",
     "prompts": ["whiteboard", "blackboard"]},
    {"key": "ball", "label": "Ball", "hint": "basketball, football, any sports ball",
     "prompts": ["ball", "basketball", "football"]},
    {"key": "bottle_or_cup", "label": "Bottle / cup",
     "hint": "drink bottle, can, cup — product stand-in",
     "prompts": ["bottle", "cup", "beer bottle", "coffee cup"]},
    # Boundary stated explicitly: presence labels are only as good as the class definition,
    # and "is that a car?" for a bus or a motorbike would be answered inconsistently across
    # 100 frames. Four wheels, road-going, passenger or light commercial — nothing else.
    {"key": "car", "label": "Car / road vehicle",
     "hint": "car, taxi, SUV, van, pickup — four-wheeled road vehicles only (not buses or bikes)",
     "prompts": ["car", "truck", "van", "taxi", "sports car", "suv", "police car"]},
]


# Matches common_ml.utils.files.get_file_type, so "is this a still" is decided the same way
# the tagger decides it. Necessary because ffprobe reports a single JPEG as a 1-frame video at
# 25fps with duration 0.04s: trusting that made this script seek to t=0.02 in a 0.04s clip,
# landing past the only frame, and ffmpeg then exits 0 having written nothing.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}


def run(cmd: List[str], expect: Optional[str] = None) -> bytes:
    """Run a command, and verify it actually produced `expect`.

    ffmpeg can exit 0 without writing an output file (seeking past the last frame does it), so
    a zero exit code is not sufficient evidence of success. Checking the artifact turns that
    into an error at the point of failure instead of a confusing one further downstream.
    """
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{cmd[0]} failed: {exc.output.decode('utf-8', 'replace')[:400]}")
    if expect is not None and not os.path.exists(expect):
        raise RuntimeError(
            f"{cmd[0]} exited 0 but did not write {os.path.basename(expect)}\n  cmd: {' '.join(cmd)}"
        )
    return out


def probe(path: str) -> Dict:
    """Duration, fps and dimensions via ffprobe."""
    out = json.loads(
        run([
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
            "-print_format", "json", path,
        ])
    )
    stream = out["streams"][0]
    num, _, den = stream["avg_frame_rate"].partition("/")
    fps = float(num) / float(den) if den and float(den) else 0.0
    duration = float(out.get("format", {}).get("duration") or 0.0)
    return {"width": stream["width"], "height": stream["height"], "fps": fps, "duration": duration}


def timestamps(duration: float, count: int) -> List[float]:
    """Evenly spaced sample points, inset from both ends.

    The 2% inset skips leading slates/black frames and trailing fades, which would otherwise
    contribute empty frames that are unlabellable and unfairly penalise every detector.
    """
    if count == 1:
        return [duration / 2]
    start, end = duration * 0.02, duration * 0.98
    step = (end - start) / (count - 1)
    return [round(start + i * step, 3) for i in range(count)]


def extract(source: str, out_png: str, out_jpg: str, max_side: int, ts: Optional[float]) -> None:
    # -ss before -i seeks by keyframe (fast) and is deterministic, which is what matters here.
    seek = ["-ss", f"{ts:.3f}"] if ts is not None else []
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *seek, "-i", source, "-frames:v", "1"]
    if max_side > 0:
        # Downscale only: leave frames already under the cap untouched, and keep dimensions
        # even (-2) so the encoders are happy.
        cmd += ["-vf", (
            f"scale='if(gt(max(iw,ih),{max_side}),if(gt(iw,ih),{max_side},-2),iw)':"
            f"'if(gt(max(iw,ih),{max_side}),if(gt(iw,ih),-2,{max_side}),ih)'"
        )]
    run(cmd + [out_png], expect=out_png)
    run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", out_png,
         "-vf", "scale='if(gt(iw,640),640,iw)':-2", "-q:v", "4", out_jpg], expect=out_jpg)


def is_blank(path: str) -> bool:
    """True when a frame carries no image content at all.

    Judged on pixel *variation*, never brightness. Dark cinematography is not a blank frame:
    a Social Network frame here has mean luma 15 while clearly showing a person and a GAP
    logo, so any brightness threshold that catches real black frames also throws away the
    most valuable dark-scene test cases.
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return False  # optional check; never block extraction on a missing dependency
    array = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return bool(array.std() < 2.0)


def extract_nonblank(
    source: str, out_png: str, out_jpg: str, max_side: int, ts: Optional[float],
    duration: float, attempts: int = 5, nudge: float = 3.0,
) -> Optional[float]:
    """Extract, and if the frame is blank, retry a little later. Returns the timestamp used.

    Evenly spaced sampling lands on fades and cut-to-black often enough to matter: blank frames
    are unlabellable, so they silently shrink the usable evaluation set.
    """
    for attempt in range(attempts):
        candidate = ts if attempt == 0 or ts is None else min(ts + attempt * nudge, max(0.0, duration - 0.1))
        extract(source, out_png, out_jpg, max_side, candidate)
        if not is_blank(out_png):
            return candidate
        if ts is None:
            break  # a still has nowhere else to sample
    return ts


def build_manifest(max_side: int) -> List[Dict]:
    frames_dir = os.path.join(HERE, "frames")
    thumbs_dir = os.path.join(HERE, "thumbs")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(thumbs_dir, exist_ok=True)

    manifest: List[Dict] = []
    for name, count in FRAME_BUDGET.items():
        source = os.path.join(TEST_FILES, name)
        if not os.path.exists(source):
            print(f"  skip {name}: not found in test-files/", file=sys.stderr)
            continue

        info = probe(source)
        # Extension, not ffprobe: a JPEG probes as a 1-frame 25fps "video" of 0.04s.
        is_still = os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
        points = [None] if is_still else timestamps(info["duration"], count)
        stem = os.path.splitext(name)[0]

        for i, ts in enumerate(points):
            frame_id = f"{stem}__{i:03d}"
            png = os.path.join(frames_dir, frame_id + ".png")
            jpg = os.path.join(thumbs_dir, frame_id + ".jpg")
            used = extract_nonblank(source, png, jpg, max_side, ts, info["duration"])
            manifest.append({
                "id": frame_id,
                "source": name,
                # frame_idx ties a frozen frame back to the tagger's frame_info.frame_idx, so
                # a detection here is locatable in the original video.
                "timestamp": used,
                "frame_idx": int(round(used * info["fps"])) if used is not None else 0,
                "frame": f"frames/{frame_id}.png",
                "thumb": f"thumbs/{frame_id}.jpg",
            })
            moved = "" if used == ts else f"  (resampled from {ts:.2f}s: blank)"
            print(f"  {frame_id}  t={used if used is not None else 0:.2f}s{moved}")

    return manifest


def write_html(manifest: List[Dict]) -> str:
    """Emit the labelling tool with the manifest inlined.

    Inlined rather than fetched: a fetch() of a sibling JSON is blocked by CORS under file://,
    but <img src="..."> is not. So the tool works by just opening the file — no server, no
    build step.
    """
    payload = json.dumps({"frames": manifest, "classes": CLASSES})
    html = _HTML_TEMPLATE.replace("__PAYLOAD__", payload)
    path = os.path.join(HERE, "label_presence.html")
    with open(path, "w") as handle:
        handle.write(html)
    return path


_HTML_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Detector Eval — Presence Labelling</title>
<style>
  :root {
    --bg:#f6f7f9; --fg:#14161a; --muted:#5b6270; --line:#d8dce3; --card:#ffffff;
    --accent:#2f6feb; --on-accent:#ffffff; --warn:#b4690e;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg:#14161a; --fg:#e7e9ee; --muted:#9aa2b1; --line:#2c313a; --card:#1b1e24;
      --accent:#5b8cf5; --on-accent:#0b0d10; --warn:#e0a33c;
    }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
  header { position:sticky; top:0; z-index:5; background:var(--card);
           border-bottom:1px solid var(--line); padding:10px 16px;
           display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; font-weight:650; }
  .grow { flex:1 1 auto; }
  .meta { color:var(--muted); font-size:13px; font-variant-numeric:tabular-nums; }
  .bar { height:4px; background:var(--line); border-radius:2px; overflow:hidden; width:180px; }
  .bar > i { display:block; height:100%; background:var(--accent); width:0; }
  main { display:grid; grid-template-columns:minmax(0,1fr) 310px; gap:16px; padding:16px;
         align-items:start; }
  @media (max-width:900px) { main { grid-template-columns:minmax(0,1fr); } }
  .stage { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px; }
  .stage img { display:block; width:100%; height:auto; max-height:70vh;
               object-fit:contain; border-radius:6px; background:#000; }
  .panel { background:var(--card); border:1px solid var(--line); border-radius:10px;
           padding:14px; position:sticky; top:64px; }
  .cls { display:flex; gap:10px; align-items:flex-start; width:100%; text-align:left;
         padding:8px 10px; margin-bottom:6px; border:1px solid var(--line);
         border-radius:8px; background:transparent; color:inherit; cursor:pointer; font:inherit; }
  .cls:hover { border-color:var(--accent); }
  .cls[aria-pressed="true"] { background:var(--accent); color:var(--on-accent);
                              border-color:var(--accent); }
  .cls kbd { flex:0 0 auto; font:600 12px ui-monospace,monospace; border:1px solid currentColor;
             border-radius:4px; padding:0 5px; opacity:.75; }
  .cls b { display:block; font-weight:600; }
  .cls i { display:block; font-size:12px; opacity:.75; font-style:normal; }
  .row { display:flex; gap:8px; margin-top:12px; }
  button.act { flex:1; padding:9px; border-radius:8px; border:1px solid var(--line);
               background:transparent; color:inherit; cursor:pointer; font:inherit; }
  button.act:hover { border-color:var(--accent); }
  button.primary { background:var(--accent); color:var(--on-accent); border-color:var(--accent); }
  .unsure[aria-pressed="true"] { background:var(--warn); color:var(--on-accent);
                                 border-color:var(--warn); }
  .hint { color:var(--muted); font-size:12px; margin-top:12px; }
  .done { color:var(--accent); font-weight:600; }
</style></head><body>

<header>
  <h1>Presence labelling</h1>
  <div class="meta" id="pos"></div>
  <div class="bar"><i id="prog"></i></div>
  <div class="meta" id="count"></div>
  <div class="grow"></div>
  <button class="act primary" id="export" style="flex:0 0 auto">Download labels JSON</button>
</header>

<main>
  <div class="stage">
    <img id="img" alt="evaluation frame">
    <div class="meta" id="src" style="margin-top:8px"></div>
  </div>

  <div class="panel">
    <div id="classes"></div>
    <button class="act unsure" id="unsure" aria-pressed="false">Mark frame unsure (u)</button>
    <div class="row">
      <button class="act" id="prev">&larr; Prev</button>
      <button class="act primary" id="next">Next &rarr;</button>
    </div>
    <div class="hint">
      Mark every class <b>visibly present</b> in the frame, however small.
      Leaving all unchecked is a valid label meaning none are present — a frame counts as
      done once you advance past it.
      <br><br>
      Keys: <kbd>1</kbd>&ndash;<kbd>9</kbd> toggle &middot;
      <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> navigate &middot; <kbd>u</kbd> unsure.
      Progress autosaves to this browser.
    </div>
  </div>
</main>

<script>
const DATA = __PAYLOAD__;
const KEY = "model-detection-presence-v1";
const frames = DATA.frames, classes = DATA.classes;

let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }

let i = 0;
// Resume at the first unlabelled frame so a reopened session continues where it stopped.
while (i < frames.length - 1 && state[frames[i].id]) i++;

const el = id => document.getElementById(id);
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} };

function rec() {
  const id = frames[i].id;
  if (!state[id]) state[id] = { present: [], unsure: false };
  return state[id];
}

function renderClasses() {
  el("classes").innerHTML = classes.map((c, n) => `
    <button class="cls" data-key="${c.key}" aria-pressed="false">
      <kbd>${n + 1}</kbd>
      <span style="flex:1"><b>${c.label}</b><i>${c.hint}</i></span>
    </button>`).join("");
  el("classes").querySelectorAll(".cls").forEach(b => b.onclick = () => toggle(b.dataset.key));
}

function toggle(key) {
  const r = rec();
  const at = r.present.indexOf(key);
  if (at === -1) r.present.push(key); else r.present.splice(at, 1);
  save(); paint();
}

function paint() {
  const f = frames[i], r = state[f.id];
  el("img").src = f.thumb;
  el("src").textContent = `${f.source}  ·  ${f.timestamp === null ? "still" : "t=" + f.timestamp + "s"}`
    + `  ·  frame_idx ${f.frame_idx}  ·  ${f.id}`;
  el("pos").textContent = `frame ${i + 1} / ${frames.length}`;
  const done = frames.filter(x => state[x.id]).length;
  el("count").innerHTML = done === frames.length
    ? `<span class="done">all ${done} labelled</span>` : `${done} labelled`;
  el("prog").style.width = (100 * done / frames.length) + "%";
  el("classes").querySelectorAll(".cls").forEach(b =>
    b.setAttribute("aria-pressed", String(!!r && r.present.includes(b.dataset.key))));
  el("unsure").setAttribute("aria-pressed", String(!!r && r.unsure));
}

function go(d) {
  rec();                 // advancing past a frame is what marks it labelled
  save();
  i = Math.min(frames.length - 1, Math.max(0, i + d));
  paint();
}

el("next").onclick = () => go(1);
el("prev").onclick = () => go(-1);
el("unsure").onclick = () => { const r = rec(); r.unsure = !r.unsure; save(); paint(); };

el("export").onclick = () => {
  const out = {
    schema: classes.map(c => c.key),
    labelled: frames.filter(f => state[f.id]).length,
    total: frames.length,
    labels: Object.fromEntries(frames.map(f =>
      [f.id, state[f.id] || { present: [], unsure: false, skipped: true }])),
  };
  const url = URL.createObjectURL(new Blob([JSON.stringify(out, null, 2)],
    { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url; a.download = "presence_labels.json"; a.click();
  URL.revokeObjectURL(url);
};

addEventListener("keydown", e => {
  if (e.key === "ArrowRight") { go(1); e.preventDefault(); }
  else if (e.key === "ArrowLeft") { go(-1); e.preventDefault(); }
  else if (e.key === "u") { el("unsure").click(); }
  else if (/^[1-9]$/.test(e.key)) { const c = classes[+e.key - 1]; if (c) toggle(c.key); }
});

renderClasses(); paint();
</script></body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-side", type=int, default=1920,
                        help="cap the long side of stored frames; 0 disables (downscale only)")
    parser.add_argument("--html-only", action="store_true",
                        help="regenerate the labelling tool from the existing manifest")
    args = parser.parse_args()

    manifest_path = os.path.join(HERE, "frames.json")

    if args.html_only:
        if not os.path.exists(manifest_path):
            print("no frames.json — run without --html-only first", file=sys.stderr)
            return 1
        with open(manifest_path) as handle:
            existing = json.load(handle)
        manifest = existing["frames"]
        # Rewrite `classes` too, not just the HTML: score_detectors.py and contact_sheet.py
        # both read the schema from frames.json, so a schema edit applied only to the tool
        # would be labelled but never scored — the class would vanish silently.
        if existing.get("classes") != CLASSES:
            before = [c["key"] for c in existing.get("classes", [])]
            after = [c["key"] for c in CLASSES]
            existing["classes"] = CLASSES
            with open(manifest_path, "w") as handle:
                json.dump(existing, handle, indent=2)
            added = [k for k in after if k not in before]
            removed = [k for k in before if k not in after]
            changes = ", ".join(
                filter(None, [
                    f"added {'/'.join(added)}" if added else "",
                    f"removed {'/'.join(removed)}" if removed else "",
                ])
            ) or "reordered/edited"
            print(f"schema changed in frames.json ({changes})")
    else:
        cap = f"max side {args.max_side}px" if args.max_side > 0 else "native resolution"
        print(f"extracting frames ({cap}) ...")
        manifest = build_manifest(args.max_side)
        if not manifest:
            print("no frames extracted", file=sys.stderr)
            return 1
        with open(manifest_path, "w") as handle:
            json.dump({"max_side": args.max_side, "classes": CLASSES, "frames": manifest},
                      handle, indent=2)
        print(f"\n{len(manifest)} frames -> eval/frames/, eval/thumbs/, eval/frames.json")

    path = write_html(manifest)
    print(f"labelling tool -> {os.path.relpath(path, REPO)}")
    print(f"  {len(CLASSES)} classes: {', '.join(c['key'] for c in CLASSES)}")
    print("  open it in a browser; no server needed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
