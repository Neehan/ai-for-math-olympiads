"""Recover completed proof text lost by an empty SDK aggregate result.

The provider's raw per-attempt transcript is the source of truth.  This tool
only fills an empty committed proof phase (or a demonstrably completed active
phase) from the last finalized assistant message containing ``## Final
Solution``.  It never invents text, starts a model session, or changes token
accounting.  Original JSON files are copied once under ``recovery-backup/``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from src.checkpoint import _atomic_json, phase_record, progress_tool_calls
from src.constants import (
    CHECKPOINT_ROOT_DEFAULT,
    PHASE_CRITIQUE,
    PHASE_PLAN,
    PHASE_PLAN_WRAP_UP,
)
from src.models import PhaseResult, ReconnectEvent

_EXCLUDED = {PHASE_CRITIQUE, PHASE_PLAN, PHASE_PLAN_WRAP_UP}


def _message_text(record: dict[str, Any]) -> str:
    if record.get("type") != "assistant":
        return ""
    message = record.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def finalized_solutions(transcript: Path) -> list[str]:
    """Return finalized assistant proofs, in transcript order."""
    solutions: list[str] = []
    with transcript.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            message = record.get("message")
            stop_reason = (
                message.get("stop_reason") if isinstance(message, dict) else None
            )
            text = _message_text(record).strip()
            if stop_reason == "end_turn" and "## Final Solution" in text:
                solutions.append(text)
    return solutions


def _backup(path: Path, attempt_dir: Path) -> None:
    backup = attempt_dir / "recovery-backup" / path.relative_to(attempt_dir)
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)


def _transcript(root: Path, role_state: dict[str, Any]) -> Path | None:
    scratch = role_state.get("scratch_name")
    session_id = role_state.get("session_id")
    if not scratch or not session_id:
        return None
    matches = list(
        (root / "w" / str(scratch) / ".claude-runtime" / "projects").glob(
            f"**/{session_id}.jsonl"
        )
    )
    return matches[0] if len(matches) == 1 else None


def recover_attempt(
    attempt_dir: Path,
    root: Path,
    *,
    arm: str,
    include_unmarked: bool,
    apply: bool,
) -> str:
    state_path = attempt_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    marker = str(state.get("completion_marker") or "")
    if marker and f"/{arm}/" not in marker:
        return "skip"
    if not marker and not include_unmarked:
        return "skip"
    changed = False
    for role, role_state in state.get("roles", {}).items():
        if not isinstance(role_state, dict):
            continue
        transcript = _transcript(root, role_state)
        if transcript is None:
            continue
        candidates = finalized_solutions(transcript)
        if not candidates:
            continue
        recovered_text = candidates[-1]
        phase_dir = attempt_dir / "phases"
        phase_paths = sorted(phase_dir.glob(f"{role}-*.json"))
        target_path: Path | None = None
        target_record: dict[str, Any] | None = None
        for phase_path in reversed(phase_paths):
            record = json.loads(phase_path.read_text(encoding="utf-8"))
            phase = record.get("phase", {})
            if (
                isinstance(phase, dict)
                and phase.get("label") not in _EXCLUDED
                and not phase.get("budget_exhausted")
                and not str(phase.get("text", "")).strip()
            ):
                target_path, target_record = phase_path, record
                break
        if target_path is not None and target_record is not None:
            target_record["phase"]["text"] = recovered_text
            if apply:
                _backup(target_path, attempt_dir)
                _atomic_json(target_path, target_record)
            changed = True
            continue

        active = role_state.get("active")
        if not isinstance(active, dict) or active.get("label") in _EXCLUDED:
            continue
        progress = active.get("progress")
        if not isinstance(progress, dict):
            continue
        progress_text = "\n".join(
            str(part) for part in progress.get("text_parts", [])
        )
        # Requiring both the finalized raw message and the durable progress
        # prefix prevents treating a merely-started Final Solution as complete.
        if recovered_text not in progress_text:
            continue
        tracker = role_state.get("tracker")
        if not isinstance(tracker, dict):
            continue
        reconnect_start = int(active.get("reconnect_start", 0))
        reconnects = [
            ReconnectEvent(**event)
            for event in role_state.get("reconnects", [])[reconnect_start:]
        ]
        phase = PhaseResult(
            label=str(active["label"]),
            prompt=str(active["prompt"]),
            text=recovered_text,
            output_tokens=int(tracker.get("current_phase_streamed_tokens", 0)),
            cumulative_output_tokens=int(tracker.get("spent", 0)),
            num_turns=0,
            duration_ms=0,
            total_cost_usd=0.0,
            is_error=False,
            stop_reason="recovered_finalized_transcript",
            budget_exhausted=False,
            tool_calls=progress_tool_calls(progress),
            reconnects=reconnects,
            process_resume_count=int(active.get("process_resume_count", 0)),
            discarded_output_text=str(active.get("discarded_output_text", "")),
        )
        sequence = int(active["sequence"])
        phase_path = phase_dir / f"{role}-{sequence:06d}.json"
        record = {
            "sequence": sequence,
            "phase_id": active["phase_id"],
            "tracker": tracker,
            "phase": phase_record(phase),
        }
        if apply:
            _backup(state_path, attempt_dir)
            phase_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(phase_path, record)
            role_state["active"] = None
            role_state["next_sequence"] = sequence + 1
            _atomic_json(state_path, state)
        changed = True
    return "recover" if changed else "unchanged"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root", type=Path, default=CHECKPOINT_ROOT_DEFAULT
    )
    parser.add_argument("--arm", default="baseline")
    parser.add_argument(
        "--include-unmarked",
        action="store_true",
        help="also inspect an active attempt that failed before preparing its marker",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    counts = {"recover": 0, "unchanged": 0, "skip": 0}
    run_roots = [args.checkpoint_root]
    run_roots.extend(
        child
        for child in sorted(args.checkpoint_root.iterdir())
        if child.is_dir() and (child / "attempts").is_dir()
    )
    for run_root in run_roots:
        for attempt in sorted((run_root / "attempts").glob("*")):
            if not (attempt / "state.json").exists():
                continue
            outcome = recover_attempt(
                attempt,
                run_root,
                arm=args.arm,
                include_unmarked=args.include_unmarked,
                apply=args.apply,
            )
            counts[outcome] += 1
            if outcome == "recover":
                print(
                    f"{'recovered' if args.apply else 'would recover'} "
                    f"{run_root.name}/{attempt.name}"
                )
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
