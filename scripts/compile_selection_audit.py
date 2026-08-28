"""Compile deterministic selection verdicts after isolated output merging."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.constants import CONFIG_PATH, MODE_SELECTION, MODE_SELECTION_NO_PROBLEM
from src.storage import compile_arm_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()

    config = dataclasses.replace(load_config(CONFIG_PATH), model=args.model)
    try:
        arm = config.arms[args.arm]
    except KeyError as error:
        raise SystemExit(f"Unknown arm: {args.arm}") from error
    if arm.mode not in {MODE_SELECTION, MODE_SELECTION_NO_PROBLEM}:
        raise SystemExit(f"Arm {args.arm!r} is not a selection arm")
    path, count = compile_arm_audit(
        config,
        arm,
        results_root=args.results_root.resolve(),
    )
    print(f"compiled {count} deterministic selection verdict(s) -> {path}")


if __name__ == "__main__":
    main()
