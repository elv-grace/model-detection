#!/usr/bin/env python3
"""Build eval/report.html from eval/report_src.html by inlining the contact sheets.

Why the split
-------------
report_src.html is the authored source and carries __SHEET_*__ placeholders. report.html is the
build output with every sheet inlined as a data URI, and it is the ONLY file that gets
published. That matters because an artifact's URL is bound to its file path -- publishing
report_src.html would create a second artifact at a new URL instead of updating the existing
one. Keeping the source separate also keeps it diffable, which a file with megabytes of base64
in it is not.

Inlining is required rather than preferred: the artifact CSP blocks every external host except
Google Fonts, so a sheet referenced by path would silently fail to load for anyone but me.

Placeholders
------------
Each prefix selects a source directory, so one placeholder syntax reaches every experiment:

    __SHEET_<name>__      -> 04_brand_person_mark/sheets/       CURRENT mark schema
    __SHEET_L_<name>__    -> 02_brand_person_101/sheets/        legacy 101-term runs
    __SHEET_V_<name>__    -> 02_brand_person_101/sheets_visual/ per-exemplar image prompts
    __SHEET_B_<name>__    -> 03_prompt_ablation/sheets_bare/    bare-word ablation
    __SHEET_G_<name>__    -> box_gt/                            ground truth itself

The unprefixed form is the current experiment, so the report's main sheets track the live
schema and a superseded one has to be asked for by name.

Usage
-----
    python eval/report/build_report.py
"""
from __future__ import annotations

import argparse
import base64
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Sheet names contain "__" themselves (gdino__brand, yoloe11__person__person_cafe1__person), so
# the closing delimiter is anchored on the attribute's quote rather than on the first "__".
PATTERN = re.compile(r"__SHEET_(V_|B_|L_|G_)?([A-Za-z0-9_.\-]+?)__(?=\")")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default=os.path.join(HERE, "report_src.html"))
    parser.add_argument("--out", default=os.path.join(HERE, "report.html"))
    exp = os.path.join(HERE, "..", "experiments")
    parser.add_argument("--sheets", default=os.path.join(exp, "04_brand_person_mark", "sheets"))
    parser.add_argument("--legacy", default=os.path.join(exp, "02_brand_person_101", "sheets"))
    parser.add_argument("--visual", default=os.path.join(exp, "02_brand_person_101", "sheets_visual"))
    parser.add_argument("--bare", default=os.path.join(exp, "03_prompt_ablation", "sheets_bare"))
    parser.add_argument("--gt", default=os.path.join(HERE, "..", "box_gt"))
    args = parser.parse_args()

    with open(args.src) as handle:
        html = handle.read()

    missing = []
    inlined = []

    def substitute(match: "re.Match") -> str:
        prefix, name = match.group(1), match.group(2)
        base = {"V_": args.visual, "B_": args.bare,
                "L_": args.legacy, "G_": args.gt}.get(prefix, args.sheets)
        path = os.path.join(base, f"{name}.jpg")
        if not os.path.exists(path):
            missing.append(path)
            return match.group(0)
        with open(path, "rb") as handle:
            blob = base64.b64encode(handle.read()).decode("ascii")
        inlined.append((name, os.path.getsize(path)))
        return f"data:image/jpeg;base64,{blob}"

    html = PATTERN.sub(substitute, html)

    if missing:
        for path in missing:
            print(f"MISSING {path}", file=sys.stderr)
        print(f"{len(missing)} sheet(s) missing -- run eval/sheet_bp.py first", file=sys.stderr)
        return 1

    with open(args.out, "w") as handle:
        handle.write(html)

    for name, size in inlined:
        print(f"  inlined {name:52} {size//1024:>5} KB")
    print(f"\n{len(inlined)} sheets -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")
    if os.path.getsize(args.out) > 16e6:
        print("WARNING: over the 16 MB artifact limit", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
