"""Routing tests: which detector serves which target, and what never loads."""
import pytest
from general_detection.prompts import (
    BRAND_PROMPTS, DEFAULT_CLASS_PROMPTS, expand_target, split_by_detector,
)


def test_default_target_expands_brand_to_the_mark_terms():
    """`brand` must expand, not pass through literally.

    The bare word is a far weaker prompt than the mark list -- only Grounding DINO grounds it
    at all -- so a target of ["brand"] silently becoming the literal string would be a large
    quality regression that nothing else would catch.
    """
    assert expand_target(["brand"])["brand"] == BRAND_PROMPTS
    assert expand_target(["brand", "person"]) == DEFAULT_CLASS_PROMPTS


def test_unknown_target_becomes_its_own_parent():
    assert expand_target(["car"]) == {"car": ["car"]}


def test_blank_terms_are_dropped_and_an_empty_target_is_an_error():
    assert expand_target(["person", "  ", ""]) == {"person": ["person"]}
    with pytest.raises(ValueError):
        expand_target(["", "   "])


@pytest.mark.parametrize("target,open_vocab,closed", [
    (["brand", "person"], ["brand"], ["person"]),
    (["person"],          [],        ["person"]),
    (["car"],             ["car"],   []),
    (["person", "car"],   ["car"],   ["person"]),
])
def test_targets_route_to_the_right_detectors(target, open_vocab, closed):
    o, c = split_by_detector(expand_target(target))
    assert sorted(o) == open_vocab
    assert sorted(c) == closed


def test_a_person_only_target_leaves_the_open_vocab_side_empty():
    """The caller is expected to skip an empty side rather than run it on nothing.

    This is the largest available saving for the common case: a person-only request should
    never construct the brand model, which is the large weight.
    """
    o, _ = split_by_detector(expand_target(["person"]))
    assert not o
