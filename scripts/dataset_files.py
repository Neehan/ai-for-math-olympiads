#!/usr/bin/env python3
"""List the dataset files a stage needs, for run.sh's --dataset-dir mode.

Prints one `<container name> <published name>` pair per line. The container
name is the fixed name docker/entrypoint.sh writes into /run/contest, so the
dataset-specific naming stays in src/constants.py and out of shell code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The repo root must precede the stale `src` that `pip install .` left in
# site-packages; python puts this script's own directory on sys.path first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants import _ACTIVE_DATASET_FILES  # noqa: E402

# The entrypoint always writes these fixed names; the loader reads and deletes
# each one before any agent spawns.
_CONTAINER_NAMES: dict[str, str] = {
    "problems": "problems.jsonl",
    "hints": "hints.jsonl",
    "outlines": "outlines.jsonl",
    "solutions": "solutions.jsonl",
    "selection": "selection.jsonl",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "audit", "state-audit"))
    parser.add_argument(
        "--selection-arm",
        action="store_true",
        help="Also list the frozen selection candidate set",
    )
    args = parser.parse_args()

    kinds = ["problems", "hints", "outlines"]
    # Reference solutions never enter a generation container.
    if args.stage in ("audit", "state-audit"):
        kinds.append("solutions")
    if args.selection_arm:
        kinds.append("selection")

    for kind in kinds:
        published = _ACTIVE_DATASET_FILES[kind]
        if published is None:
            continue
        print(f"{_CONTAINER_NAMES[kind]} {published}")


if __name__ == "__main__":
    main()
