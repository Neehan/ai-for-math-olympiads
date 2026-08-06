"""Host-side cleanup after staged run outputs have merged into ``results/``.

This module intentionally uses only the standard library: ``run.sh`` must work
on a host that has Docker and Python but not the harness's SDK dependencies.
"""

import argparse
import fcntl
import json
import re
import shutil
from pathlib import Path
from typing import Any

_ATTEMPT_RE = re.compile(r"^[0-9a-f]{24}$")
_SCRATCH_RE = re.compile(r"^[0-9a-f]{8}$")


def cleanup_completed(root: Path, results_root: Path) -> int:
    """Remove checkpoints only after their canonical result marker exists."""
    attempts_root = root / "attempts"
    if not attempts_root.is_dir():
        return 0

    completed: list[tuple[Path, list[Path]]] = []
    for state_path in attempts_root.glob("*/state.json"):
        attempt_dir = state_path.parent
        if not _ATTEMPT_RE.fullmatch(attempt_dir.name):
            raise ValueError(f"Refusing malformed attempt path: {attempt_dir}")
        state: dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8"))
        raw_marker = state.get("completion_marker")
        if raw_marker is None:
            continue
        if not isinstance(raw_marker, str):
            raise ValueError(f"Malformed completion marker in {state_path}")
        marker = Path(raw_marker)
        invalid_part = any(part in {"", ".", ".."} for part in marker.parts)
        if marker.is_absolute() or invalid_part:
            raise ValueError(f"Malformed completion marker in {state_path}")
        # A durable canonical marker is stronger evidence than the checkpoint's
        # completed bit: the process can die immediately after writing either.
        if not (results_root / marker).is_file():
            continue
        roles = state.get("roles", {})
        if not isinstance(roles, dict):
            raise ValueError(f"Malformed roles in {state_path}")
        workspaces: list[Path] = []
        for role in roles.values():
            if not isinstance(role, dict):
                raise ValueError(f"Malformed role in {state_path}")
            scratch_name = role.get("scratch_name")
            if scratch_name is None:
                continue
            if not _SCRATCH_RE.fullmatch(str(scratch_name)):
                raise ValueError(f"Refusing malformed workspace in {state_path}")
            workspaces.append(root / "w" / str(scratch_name))
        completed.append((attempt_dir, workspaces))

    # Validate the entire cleanup set before deleting any of it. Then honor
    # the same advisory lock used by a live container: another run.sh may be
    # finishing this exact attempt while our trap is cleaning the namespace.
    removed = 0
    for attempt_dir, workspaces in completed:
        lock_path = attempt_dir / ".lock"
        with lock_path.open("a+") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            for workspace in workspaces:
                shutil.rmtree(workspace, ignore_errors=True)
            shutil.rmtree(attempt_dir)
            removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("results_root", type=Path)
    args = parser.parse_args()
    cleanup_completed(args.root, args.results_root)


if __name__ == "__main__":
    main()
