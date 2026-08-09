"""Keep only durable completed attempts/runs in a generation staging tree."""

import shutil
import sys
from pathlib import Path


def prune_staging(root: Path) -> None:
    """Discard resume markers/partial writes while retaining completed bank runs."""
    if not root.is_dir():
        raise ValueError(f"Staging root is not a directory: {root}")
    for seed_dir in root.glob("*/*/*/seed_*"):
        if (seed_dir / "meta.json").is_file() and (
            seed_dir / "solution.md"
        ).is_file():
            continue
        keep: set[Path] = set()
        for run_dir in seed_dir.glob("run_[0-9][0-9]"):
            if (run_dir / "meta.json").is_file() and (
                run_dir / "solution.md"
            ).is_file():
                keep.add(run_dir)
        for child in list(seed_dir.iterdir()):
            if child in keep:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
        if not any(seed_dir.iterdir()):
            seed_dir.rmdir()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: prune_staging.py STAGING_ROOT")
    prune_staging(Path(sys.argv[1]).resolve())
