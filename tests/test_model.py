"""Unit tests for the crop-and-embed path.

The detector and embedder are stubbed, so these run without downloading YOLOE or SigLIP 2
weights and without a GPU. The end-to-end test at the bottom is opt-in.
"""
import os

import numpy as np
import pytest

pytest.importorskip("ultralytics")
pytest.importorskip("transformers")

from general_detection.config import RuntimeConfig
from general_detection.detector import Detection, YoloeDetector
from general_detection.embedder import Siglip2CropEmbedder
from general_detection.model import EntityDetector

_DIM = 8


class _FakeEmbedder:
    model_id = "fake/siglip2"
    dim = _DIM

    def embed(self, crops, cfg):
        vectors = np.zeros((len(crops), _DIM), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors, [2.5] * len(crops)


class _FakeDetector:
    name = "fake-yoloe"

    def __init__(self, detections=None):
        self._detections = detections

    def detect(self, img, cfg):
        if self._detections is not None:
            return self._detections
        return [
            Detection(
                label="brand",
                prompt="letter logo",
                score=0.9,
                box={"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4},
                crop=np.zeros((40, 60, 3), dtype=np.uint8),
                detector="fake-yoloe",
            )
        ]


def _model(cfg=None, detections=None, person=None):
    # Bypass __init__ so no weights are loaded. The model holds two detector slots and either
    # may be None -- a target routed to only one side never constructs the other.
    model = object.__new__(EntityDetector)
    model.config = cfg or RuntimeConfig()
    model._brand = _FakeDetector(detections)
    model._person = person
    model._brand_mode = "fast"
    model.embedder = _FakeEmbedder()
    return model


# ---- tag shape ------------------------------------------------------------------


def test_tag_frame_emits_one_vector_tag_per_detection():
    tags = _model().tag_frame(np.zeros((1080, 1920, 3), dtype=np.uint8))

    assert len(tags) == 1
    tag = tags[0]
    # the parent term, not the phrasing that fired
    assert tag.tag == "brand"
    assert len(tag.vector) == _DIM
    # normalized box, which is what the video-editor overlay multiplies by canvas size
    assert set(tag.box) == {"x1", "y1", "x2", "y2"}
    assert all(0.0 <= v <= 1.0 for v in tag.box.values())


def test_tag_carries_the_provenance_the_index_needs():
    info = _model().tag_frame(np.zeros((100, 100, 3), dtype=np.uint8))[0].additional_info

    assert info["kind"] == "crop"
    assert info["prompt"] == "letter logo"  # which phrasing fired, for recall tuning
    assert info["score"] == 0.9
    assert info["dim"] == _DIM
    # per TAG, not per run: two detectors are in play, so this rides on the Detection
    assert info["detector"] == "fake-yoloe"
    assert info["embedder"] == "fake/siglip2"
    # crop_padding changes the vector: vectors built at different padding are not comparable
    assert info["crop_padding"] == pytest.approx(RuntimeConfig().crop_padding)
    # lets the heavily-interpolated tail be filtered downstream without re-tagging
    assert info["upscale"] == 2.5


def test_no_detections_emits_nothing():
    assert _model(detections=[]).tag_frame(np.zeros((100, 100, 3), dtype=np.uint8)) == []


def test_embed_whole_frame_adds_one_untagged_full_frame_vector():
    cfg = RuntimeConfig(embed_whole_frame=True)
    tags = _model(cfg).tag_frame(np.zeros((100, 200, 3), dtype=np.uint8))

    assert len(tags) == 2
    frame_tag = tags[-1]
    assert frame_tag.tag == ""   # the frame itself, not a detected entity
    assert frame_tag.box == {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}
    assert frame_tag.additional_info["kind"] == "frame"


def test_embed_whole_frame_still_emits_when_nothing_is_detected():
    cfg = RuntimeConfig(embed_whole_frame=True)
    tags = _model(cfg, detections=[]).tag_frame(np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(tags) == 1
    assert tags[0].additional_info["kind"] == "frame"


# ---- cropping -------------------------------------------------------------------


def test_crop_padding_expands_the_crop_but_not_the_reported_box():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # 40x40 box, 10% padding -> 4px each side
    crop = YoloeDetector._crop(img, 10, 10, 50, 50, 0.1)
    assert crop.shape[:2] == (48, 48)


def test_crop_clamps_at_the_frame_edge():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = YoloeDetector._crop(img, 0, 0, 20, 20, 0.5)
    assert crop.shape[:2] == (30, 30)   # padded to -10..30, clamped to 0..30


def test_crop_is_contiguous_for_pil():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = YoloeDetector._crop(img, 10, 10, 50, 50, 0.0)
    # a sliced view is non-contiguous and PIL.Image.fromarray rejects it
    assert crop.flags["C_CONTIGUOUS"]


def test_degenerate_box_yields_no_crop():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert YoloeDetector._crop(img, 50, 50, 50, 50, 0.0) is None


# ---- dedupe ---------------------------------------------------------------------


def _stub_detector(conf=0.007):
    detector = object.__new__(YoloeDetector)
    detector._prompts = ["logo", "letter logo", "person"]
    detector._labels = ["brand", "brand", "person"]
    detector._group_of_label = {"brand": 0, "person": 1}
    # Normally set by BaseDetector.__init__; the per-backend threshold is the fallback the
    # per-parent gate defers to, so it has to exist even on a stub.
    detector.conf = conf
    detector.imgsz = 1280
    return detector


def _overlapping_detections(sv):
    box = [10.0, 10.0, 110.0, 110.0]
    return sv.Detections(
        xyxy=np.array([box, box, box], dtype=np.float32),
        confidence=np.array([0.9, 0.8, 0.7], dtype=np.float32),
        class_id=np.array([0, 1, 2]),
    )


def test_gate_falls_back_to_the_backend_threshold_and_class_conf_overrides_it():
    """There is no global `conf` any more: each backend carries its own measured threshold.

    Scores are not comparable across backends -- YOLOE's text-similarity scores and Grounding
    DINO's query scores are on different scales -- so one shared number could only ever be right
    for one of them. `class_conf` stays available as a per-parent override on top.
    """
    sv = pytest.importorskip("supervision")
    detector = _stub_detector(conf=0.25)
    box = [10.0, 10.0, 110.0, 110.0]
    dets = sv.Detections(
        xyxy=np.array([box, box, box], dtype=np.float32),
        confidence=np.array([0.3, 0.2, 0.3], dtype=np.float32),
        class_id=np.array([0, 1, 2]),
    )

    # No override: everything falls back to the backend's own 0.25, so both 0.3s survive.
    out = detector._gate(dets, RuntimeConfig())
    assert sorted(detector._labels[int(c)] for c in out.class_id) == ["brand", "person"]

    # With an override, person needs 0.55 and is dropped while brand still clears 0.25.
    out = detector._gate(dets, RuntimeConfig(class_conf={"person": 0.55}))
    assert sorted(detector._labels[int(c)] for c in out.class_id) == ["brand"]


def test_floor_is_the_lowest_gate_any_of_this_backends_classes_uses():
    """The pre-filter handed to the model must not be higher than any per-class gate.

    Passing the backend's own `conf` straight to `predict` would silently drop a class whose
    `class_conf` entry is lower than it, before the per-class gate ever ran.
    """
    detector = _stub_detector(conf=0.25)
    assert detector._floor(RuntimeConfig()) == 0.25
    assert detector._floor(RuntimeConfig(class_conf={"brand": 0.01})) == 0.01
    # An override for a parent this backend does not serve must not move its floor.
    assert detector._floor(RuntimeConfig(class_conf={"car": 0.01})) == 0.25


def test_shipped_backend_thresholds_are_reachable():
    """A gate above a class's whole score range deletes the class instead of trimming it.

    An earlier default ({"object": 0.5}) did exactly that: measured on real content the catch-all
    never exceeded 0.105. `class_conf` now ships empty so that mistake has no vehicle, and the
    thresholds that do ship are the ones leave-one-clip-out selection chose. This pins that they
    stay inside (0, 1) so a typo cannot silently mute a backend.
    """
    from general_detection.detector import BRAND_BACKENDS, PERSON_BACKEND

    assert RuntimeConfig().class_conf == {}, "class_conf should ship empty"
    for name, spec in list(BRAND_BACKENDS.items()) + [("person", PERSON_BACKEND)]:
        assert 0.0 < spec["conf"] < 1.0, f"{name} threshold {spec['conf']} is unreachable"
        assert spec["imgsz"] % 32 == 0 or spec["kind"] == "gdino", \
            f"{name} imgsz {spec['imgsz']} is not a multiple of 32"


def test_dedupe_collapses_synonyms_but_preserves_cross_class_nesting():
    sv = pytest.importorskip("supervision")
    detector = _stub_detector()

    out = detector._dedupe(_overlapping_detections(sv), RuntimeConfig())

    # "logo" and "brand mark" fire on the same pixels and collapse to one; the overlapping
    # "screen display" survives, because a logo inside a screen is real structure.
    labels = sorted(detector._labels[int(i)] for i in out.data["prompt_id"])
    assert labels == ["brand", "person"]


def test_dedupe_keeps_the_highest_scoring_phrasing():
    sv = pytest.importorskip("supervision")
    detector = _stub_detector()

    out = detector._dedupe(_overlapping_detections(sv), RuntimeConfig())

    kept = {detector._prompts[int(i)] for i in out.data["prompt_id"]}
    assert "logo" in kept and "brand mark" not in kept   # 0.9 beats 0.8


def test_cross_class_nms_merges_everything_when_enabled():
    sv = pytest.importorskip("supervision")
    detector = _stub_detector()

    out = detector._dedupe(
        _overlapping_detections(sv), RuntimeConfig(cross_class_nms_iou=0.5)
    )
    assert len(out) == 1


# ---- patch budget ---------------------------------------------------------------


def test_budget_is_the_full_allowance_by_default():
    embedder = object.__new__(Siglip2CropEmbedder)
    crop = np.zeros((32, 32, 3), dtype=np.uint8)
    assert embedder._budget(crop, RuntimeConfig()) == 256


def test_max_upscale_lowers_the_budget_by_the_square_of_the_cap():
    embedder = object.__new__(Siglip2CropEmbedder)
    crop = np.zeros((32, 32, 3), dtype=np.uint8)   # native 2x2 = 4 patches
    # a 4x linear cap allows 4^2 = 16x the native patch count
    assert embedder._budget(crop, RuntimeConfig(max_upscale=4.0)) == 64


def test_max_upscale_never_exceeds_max_num_patches():
    embedder = object.__new__(Siglip2CropEmbedder)
    crop = np.zeros((640, 640, 3), dtype=np.uint8)
    assert embedder._budget(crop, RuntimeConfig(max_upscale=4.0)) == 256


# ---- end to end (opt-in: needs weights, network on first run, and a GPU) ----------


@pytest.mark.skipif(
    not os.getenv("ELV_DETECTION_INTEGRATION"),
    reason="set ELV_DETECTION_INTEGRATION=1 to run against real weights",
)
def test_end_to_end_against_a_test_file():
    from common_ml.tagging.file_tagger import FileTagger

    from config import config

    test_file = os.path.join(os.path.dirname(__file__), "../test-files/1.mp4")
    model = EntityDetector(
        cfg=RuntimeConfig(),
        embedder_model_id=config["model"]["embedder"]["model_id"],
        embedder_revision=config["model"]["embedder"].get("revision"),
        cache_dir=config["storage"]["cache_path"],
    )
    tags = FileTagger.from_frame_model(model).tag(test_file)

    assert len(tags) > 0
    for tag in tags:
        assert tag.source_media == test_file
        assert tag.vector is not None
        assert len(tag.vector) == model.embedder.dim
        # unit-normalized so cosine similarity reduces to a dot product downstream
        assert abs(np.linalg.norm(tag.vector) - 1.0) < 1e-3
        # vector tags pass through per-frame; common_ml does not run-length merge them,
        # which is what keeps the box available for the overlay
        assert tag.frame_info is not None
        assert tag.frame_info.box
