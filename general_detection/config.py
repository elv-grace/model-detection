from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from general_detection.prompts import DEFAULT_CLASS_PROMPTS


@dataclass
class RuntimeConfig:
    """Runtime tunables for the crop-and-embed entity tagger, injected per-request via
    `--params` in run.py.

    See the README's "Runtime parameters" table for provenance: some defaults are
    inherited from sibling taggers, others are explicitly uncalibrated placeholders and
    are marked as such below.
    """

    # ---- what to detect ---------------------------------------------------------

    # What to look for. None means the default, ["brand", "person"].
    #
    # A term naming a known parent expands to that parent's phrasings, so "brand" becomes the
    # six mark terms rather than the literal word -- which matters, because the bare word is a
    # far weaker prompt and only Grounding DINO grounds it at all. Any other term becomes its
    # own parent with itself as the single phrasing, so ["car"] is a valid target.
    #
    # Targets are ROUTED to detectors: "person" goes to the closed COCO backend, everything else
    # to the open-vocabulary one, and a detector with nothing routed to it is never loaded. So
    # ["person"] never pays for the brand model, and ["car"] never pays for the person model.
    detect_target: Optional[List[str]] = None

    # Which open-vocabulary backend serves non-person targets.
    #
    #   "fast"      YOLOE-26 @1280.       brand AP 0.133, coverage 0.25,  98 ms/frame
    #   "coverage"  Grounding DINO @800.  brand AP 0.205, coverage 0.61, 293 ms/frame
    #
    # These are NOT fast-versus-accurate. At their shipping operating points held-out F1 is
    # 0.267 against 0.286 -- seven percent apart. What separates them is COVERAGE: Grounding
    # DINO finds ~2.4x as many DISTINCT marks, and a mark that is never cropped can never be
    # retrieved. Choose "coverage" when recall of marks matters more than throughput; it costs
    # roughly 2x end to end.
    brand_detector: str = "fast"

    # ---- detection --------------------------------------------------------------

    # Explicit {parent: [phrasings]} mapping. Normally left at the default and driven by
    # `detect_target` instead; set it directly to control phrasings per parent.
    # Overriding it re-encodes the text prompts (seconds); it does not reload the model.
    class_prompts: Dict[str, List[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_CLASS_PROMPTS.items()}
    )

    # Per-parent confidence overrides. EMPTY by default, because each backend now carries its
    # own measured threshold (see detector.py BRAND_BACKENDS / PERSON_BACKEND) and those are the
    # values leave-one-clip-out selection chose.
    # A single shared `conf` was removed rather than retuned. Detector scores are not comparable
    # across backends -- YOLOE's text-similarity scores, YOLO11's sigmoid class scores and
    # Grounding DINO's query scores are on different scales -- so one number could only ever be
    # right for one of them. The old default of 0.25 was ultralytics' closed-vocabulary value and
    # measured badly.
    # Use this to override a specific parent. CAREFUL: passing class_conf via --params REPLACES
    # this dict, it does not merge into it, so pass every class you want gated.
    class_conf: Dict[str, float] = field(default_factory=dict)

    # Ultralytics' internal NMS IoU, applied per class id (i.e. per prompt).
    iou: float = 0.7

    # NMS across the phrasings of one parent term ("logo" vs "letter logo"), applied after
    # detection. The `iou` stage above cannot do this: those are distinct class ids, so
    # ultralytics never compares their boxes, and one jersey badge survives as one box per
    # phrasing — N near-identical crops, N near-identical vectors, N stacked overlay rectangles.
    # 0.6, and chosen by a CONSTRAINT rather than by maximising a score. Measured against box
    # ground truth, the constraint is that coverage must NOT fall.
    nms_iou: float = 0.6

    # NMS across different parent terms. Disabled (1.0) by default, and it should stay disabled
    # under this schema: the only cross-parent overlap available is brand against person, and a
    # mark on a jersey overlapping the player wearing it is TWO real findings, not a duplicate.
    # Enabling this would delete one of them.
    cross_class_nms_iou: float = 1.0

    # Hard cap per frame, applied last, highest score first. Each survivor costs one
    # SigLIP 2 forward pass, so this is the pipeline's primary cost knob. (Ultralytics'
    # own max_det default is 300, which here would mean 300 embeds per frame.)
    max_detections: int = 30

    # Per-detector input size and threshold. None means "use the backend's measured default"
    # (detector.py), which is what you want unless you are deliberately re-tuning.
    # These are per-detector because a single global value would be wrong for at least one
    # backend whatever it was set to. Measured (eval/experiments/06_resolution).
    # Resolution buys recall on small objects; it does not improve the crop an embedder receives, 
    # and it makes the average crop smaller.
    brand_imgsz: Optional[int] = None
    brand_conf: Optional[float] = None
    person_imgsz: Optional[int] = None
    person_conf: Optional[float] = None

    # ---- crop selection ---------------------------------------------------------

    # Drop detections whose normalized box area is below this. Same knob, same default, as
    # model-celeb-vector's RuntimeConfig.
    min_box_size: float = 0.0

    # Drop crops whose shorter side is under this many source pixels, measured on the
    # un-padded detection box. Every tag carries `additional_info.upscale`,
    # which at a fixed budget is a monotone function of crop area, so a downstream consumer can
    # raise the effective floor by filtering. A floor set too HIGH is not recoverable: the crop
    # was never embedded, and getting it back means re-decoding the video and re-detecting.
    min_crop_pixels: int = 16

    # Fraction of the box's width/height added on each side before cropping. The *reported*
    # box stays un-padded, so the overlay draws the detection rather than the crop.
    # Note this changes the emitted vector slightly, and it is best to compare vectors in an index with the same crop_padding.
    # Recorded in `additional_info.crop_padding`.
    crop_padding: float = 0.06

    # ---- embedding --------------------------------------------------------------

    # NaFlex resolution budget: the crop is resized (aspect ratio preserved to within one
    # patch) to cover at most this many 16x16 patches. 256 is the checkpoint's documented
    # budget. Crops are small and often extreme aspect ratios — a 20x160 banner becomes
    # 96x672 here — which is exactly what NaFlex preserves and a square resize destroys.
    max_num_patches: int = 256

    # Optional ceiling on the NaFlex upscale factor, applied by lowering the per-crop patch
    # budget so small crops stay nearer native resolution instead of being interpolated up.
    # None (the default) gives every crop the full `max_num_patches` budget.
    # When set, crops are bucketed by resulting budget before batching, since the processor
    # takes one max_num_patches per call. Padding is masked out, so a crop's vector depends
    # only on its own budget and never on which crops it was batched with.
    max_upscale: Optional[float] = None

    # L2-normalize each emitted vector so cosine similarity reduces to a dot product.
    # SigLIP 2 is trained with a sigmoid loss over scaled dot products of already-normalized
    # embeddings, so cosine *is* its trained similarity and the pooled output's magnitude is
    # an artifact — normalizing is lossless.
    normalize: bool = True

    # Crops per forward pass through the vision tower.
    embed_batch_size: int = 32

    # Additionally emit one whole-frame vector per frame, with an empty tag and a full-frame
    # box. Off by default. Enable if this index needs to answer "find frames that look like
    # this crop" — there is no separate frame embedder feeding this space.
    embed_whole_frame: bool = False

    # ---- output -----------------------------------------------------------------

    # Additionally emit a vector-less FrameTag (to visualize the bounding boxes in EVIE) beside each detection's 
    # vector tag: same label, same box, no `vector`, and no embedder provenance (because there is no associated vector). 
    # Off by default, which is the original one-vector-tag-per-detection output.
    output_tags: bool = False
