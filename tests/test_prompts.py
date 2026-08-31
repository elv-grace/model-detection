"""Pure-logic tests for prompt flattening. No heavy imports, no weights."""
import pytest

from general_detection.prompts import DEFAULT_CLASS_PROMPTS, flatten


def test_flatten_is_index_aligned_and_order_stable():
    prompts, labels = flatten({"logo": ["logo", "brand mark"], "screen": ["monitor"]})
    assert prompts == ["logo", "brand mark", "monitor"]
    # class id -> prompt -> parent label must stay parallel; the detector indexes into these
    assert labels == ["logo", "logo", "screen"]


def test_flatten_defaults_cover_every_parent_term():
    prompts, labels = flatten(DEFAULT_CLASS_PROMPTS)
    assert len(prompts) == len(labels)
    assert set(labels) == set(DEFAULT_CLASS_PROMPTS)


def test_flatten_rejects_a_prompt_shared_by_two_parents():
    # The detector would emit one class id for it and the parent choice would be arbitrary.
    with pytest.raises(ValueError, match="more than one class"):
        flatten({"logo": ["mark"], "text": ["mark"]})


def test_flatten_rejects_empty_input():
    with pytest.raises(ValueError, match="no prompts"):
        flatten({"logo": []})
    with pytest.raises(ValueError, match="empty"):
        flatten({})
