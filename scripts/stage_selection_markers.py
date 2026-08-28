"""Stage opaque completion markers without exposing prior selector verdicts."""

from __future__ import annotations

import argparse
from pathlib import Path


def stage_markers(
    source_root: Path,
    destination_root: Path,
    model_dir: str,
    arm: str,
) -> int:
    """Mirror only completion-path existence for one selection arm."""
    if arm not in {"selection", "selection-no-problem"}:
        raise ValueError(f"Unsupported selection arm: {arm}")
    source_arm = source_root / model_dir / arm
    count = 0
    for marker in sorted(source_arm.glob("*/seed_*/meta.json")):
        relative = marker.relative_to(source_root)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("{}\n", encoding="utf-8")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    count = stage_markers(
        args.source_root.resolve(),
        args.destination_root.resolve(),
        args.model_dir,
        args.arm,
    )
    print(f"staged {count} opaque selection completion marker(s)")


if __name__ == "__main__":
    main()
