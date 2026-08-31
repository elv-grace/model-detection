#!/usr/bin/env python3
"""Canonical locations inside eval/, so scripts stop guessing from their own __file__.

Before the reorganisation every script assumed everything lived beside it, which was true when
eval/ was one flat directory and false the moment it was not. Resolving paths here means a
script can be moved between tools/, box_gt/ and report/ without touching its path handling.

    frameset/     the 100 frozen frames, the manifest, the presence labels
    tools/        every script, the vocabulary dump, the tag maps
    experiments/  NN_name/ per experiment, holding only that run's outputs
    box_gt/       the box-level ground-truth task, tool and scorer
    report/       report source, build script, built page

Frame paths inside frames.json are stored relative to frameset/ ("frames/NBA33min__000.png"),
so join them against FRAMESET rather than against the calling script's directory.
"""
from __future__ import annotations

import os

TOOLS = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(TOOLS)

FRAMESET = os.path.join(EVAL, "frameset")
EXPERIMENTS = os.path.join(EVAL, "experiments")
BOX_GT = os.path.join(EVAL, "box_gt")
REPORT = os.path.join(EVAL, "report")

FRAMES_JSON = os.path.join(FRAMESET, "frames.json")
PRESENCE_LABELS = os.path.join(FRAMESET, "presence_labels.json")
BOX_LABELS = os.path.join(BOX_GT, "box_labels.json")

TAG_MAP = os.path.join(TOOLS, "tag_map.json")
VOCAB_DUMP = os.path.join(TOOLS, "vocab_dump.json")

# The experiment the current schema belongs to. Scripts default here so a bare invocation
# always operates on the live setup rather than on whichever directory was newest.
CURRENT = os.path.join(EXPERIMENTS, "04_brand_person_mark")


def experiment(name: str) -> str:
    """Resolve an experiment directory from a bare name or an explicit path.

    A name may address a subdirectory ("03_prompt_ablation/runs_mark"), so the presence of a
    separator cannot be what distinguishes a name from a path -- treating it that way returned
    those names unresolved and relative, and they then globbed to nothing from any working
    directory but one. Resolution under EXPERIMENTS is tried first and an existing path wins;
    only something that does not resolve there is taken as a path in its own right.
    """
    if os.path.isabs(name):
        return name
    candidate = os.path.join(EXPERIMENTS, name)
    if os.path.exists(candidate) or not os.path.exists(name):
        return candidate
    return name


def runs_glob(spec: str) -> str:
    """Resolve a --runs argument into a glob over out.jsonl files.

    Accepts an experiment name ("04_brand_person_mark"), a subdirectory of one
    ("03_prompt_ablation/runs_mark"), or an absolute path.
    """
    base = experiment(spec)
    if os.path.isdir(os.path.join(base, "runs")):
        base = os.path.join(base, "runs")
    return os.path.join(base, "*", "out.jsonl")
