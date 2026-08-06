"""One-time, fail-loud migration of old capped sequential runs.

The legacy controller wrote complete phase logs and retained its Claude
transcript, but stopped after 64 rounds. This importer reconstructs the exact
committed phase ledger and token counter in the new checkpoint format so the
next normal invocation continues with round 65. It never edits legacy inputs
or canonical results.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import zstandard

from src.checkpoint import AttemptCheckpoint
from src.config import load_config
from src.constants import (
    CHECKPOINT_ROOT_ENV,
    CONFIG_PATH,
    LOGS_FILENAME,
    META_FILENAME,
    MODE_SEQUENTIAL,
    PHASE_WRAP_UP,
    SESSION_STATE_SUBDIR,
)
from src.models import PhaseResult, Problem, ReconnectEvent, ToolCall
from src.run import run_checkpoint_identity

_PROBLEM_PREFIX = "\nProblem:\n"
_PROBLEM_SUFFIX = "\n\nWork through it carefully"


def _statement_from_initial_prompt(prompt: str) -> str:
    if prompt.count(_PROBLEM_PREFIX) != 1:
        raise ValueError("Legacy solve prompt has an ambiguous Problem section")
    tail = prompt.split(_PROBLEM_PREFIX, 1)[1]
    if _PROBLEM_SUFFIX not in tail:
        raise ValueError("Legacy solve prompt has no expected statement terminator")
    return tail.split(_PROBLEM_SUFFIX, 1)[0].strip()


def _tool(record: dict[str, Any]) -> ToolCall:
    raw_input = record.get("input", {})
    return ToolCall(
        name=str(record["name"]),
        tool_input=dict(raw_input) if isinstance(raw_input, dict) else {},
        result=str(record.get("result", "")),
        is_error=bool(record.get("is_error", False)),
    )


def _phase(record: dict[str, Any]) -> PhaseResult:
    return PhaseResult(
        label=str(record["label"]),
        prompt=str(record["prompt"]),
        text=str(record["text"]),
        output_tokens=int(record["output_tokens"]),
        cumulative_output_tokens=int(record["cumulative_output_tokens"]),
        num_turns=int(record["num_turns"]),
        duration_ms=int(record["duration_ms"]),
        total_cost_usd=float(record["total_cost_usd"]),
        is_error=bool(record["is_error"]),
        stop_reason=str(record["stop_reason"]),
        budget_exhausted=bool(record["budget_exhausted"]),
        tool_calls=[_tool(item) for item in record.get("tool_calls", [])],
        reconnects=[
            ReconnectEvent(**item)
            for item in record.get("session_reconnects", [])
        ],
    )


def _read_phases(seed_dir: Path) -> list[PhaseResult]:
    compressed = (seed_dir / LOGS_FILENAME).read_bytes()
    text = zstandard.ZstdDecompressor().decompress(compressed).decode("utf-8")
    return [_phase(json.loads(line)) for line in text.splitlines() if line]


def _single_transcript(runtime: Path) -> tuple[str, Path]:
    transcripts = sorted((runtime / "projects").glob("*/*.jsonl"))
    if len(transcripts) != 1:
        raise ValueError(
            f"Expected exactly one provider transcript in {runtime}, "
            f"found {len(transcripts)}"
        )
    transcript = transcripts[0]
    return transcript.stem, transcript


def migrate_one(
    legacy_results: Path,
    legacy_scratch: Path,
    problem_id: str,
    checkpoint_root: Path,
) -> tuple[str, int, str]:
    config = load_config(CONFIG_PATH)
    arm = config.arms["baseline-sequential"]
    if arm.mode != MODE_SEQUENTIAL or arm.budget_units != 8:
        raise ValueError("baseline-sequential is not the expected 8x protocol")

    seed_dir = legacy_results / problem_id / "seed_1"
    meta = json.loads((seed_dir / META_FILENAME).read_text(encoding="utf-8"))
    if meta.get("termination_reason") != "round_limit":
        raise ValueError(f"{problem_id} did not terminate at the legacy round cap")
    if int(meta.get("seed", -1)) != 1:
        raise ValueError(f"{problem_id} is not seed 1")

    phases = _read_phases(seed_dir)
    if len(phases) != 129 or phases[0].label != "solve":
        raise ValueError(f"{problem_id} does not contain solve + 64 full rounds")
    if phases[-1].label != "revise" or any(p.budget_exhausted for p in phases):
        raise ValueError(f"{problem_id} has an incomplete terminal phase")
    spent = sum(phase.output_tokens for phase in phases)
    if spent != int(meta["output_tokens_spent"]):
        raise ValueError(f"{problem_id} phase totals disagree with meta.json")
    if phases[-1].cumulative_output_tokens != spent:
        raise ValueError(f"{problem_id} cumulative token count is inconsistent")

    scratch_name = str(meta["scratch_dir_name"])
    source_scratch = legacy_scratch / scratch_name
    runtime = source_scratch / SESSION_STATE_SUBDIR
    session_id, source_transcript = _single_transcript(runtime)
    statement = _statement_from_initial_prompt(phases[0].prompt)
    problem = Problem(
        problem_id=problem_id,
        statement=statement,
        domain="legacy",
        hint_h1=None,
        hint_h2=None,
        hint_h3=None,
    )

    os.environ[CHECKPOINT_ROOT_ENV] = str(checkpoint_root)
    checkpoint = AttemptCheckpoint(
        run_checkpoint_identity(config, arm, problem, seed=1)
    )
    try:
        if checkpoint.phases("main") or checkpoint.session_id("main") is not None:
            raise FileExistsError(f"Checkpoint already populated for {problem_id}")
        destination = checkpoint.scratch_dir("main")
        shutil.copytree(source_scratch, destination, dirs_exist_ok=True)

        # Claude Code stores transcripts under a cwd-derived project slug.
        # Retain the legacy copy and add the location expected from /c/w/<id>.
        project_dir = (
            destination
            / SESSION_STATE_SUBDIR
            / "projects"
            / f"-c-w-{destination.name}"
        )
        project_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_transcript, project_dir / source_transcript.name)

        reconnects: list[ReconnectEvent] = []
        checkpoint.save_session("main", session_id, reconnects)
        tracker = checkpoint.tracker(
            "main", config.budget_tokens(arm), config.wrap_up_reserve_tokens
        )
        for phase in phases:
            stop_at = (
                tracker.budget_tokens
                if phase.label == PHASE_WRAP_UP
                else tracker.soft_limit_tokens
            )
            checkpoint.begin_phase(
                "main", phase.label, phase.prompt, stop_at, tracker
            )
            tracker.finish_phase(phase.output_tokens)
            if tracker.spent != phase.cumulative_output_tokens:
                raise ValueError(
                    f"{problem_id} diverged at phase {phase.label}: "
                    f"{tracker.spent} != {phase.cumulative_output_tokens}"
                )
            reconnects.extend(phase.reconnects)
            checkpoint.finish_phase(
                "main", phase, tracker, session_id, reconnects
            )

        checkpoint.data()["legacy_migration"] = {
            "source": "64-round controller",
            "problem_id": problem_id,
            "source_scratch_name": scratch_name,
            "imported_phases": len(phases),
            "imported_output_tokens": spent,
        }
        checkpoint.save_data()
        checkpoint_path = checkpoint.path
        workspace_name = destination.name
    finally:
        checkpoint.close()

    # Reopen through the production restoration path and verify every invariant.
    restored = AttemptCheckpoint(
        run_checkpoint_identity(config, arm, problem, seed=1)
    )
    try:
        restored_phases = restored.phases("main")
        restored_tracker = restored.tracker(
            "main", config.budget_tokens(arm), config.wrap_up_reserve_tokens
        )
        if len(restored_phases) != len(phases) or restored_tracker.spent != spent:
            raise RuntimeError(f"Post-import verification failed for {problem_id}")
        if restored.session_id("main") != session_id:
            raise RuntimeError(f"Session UUID verification failed for {problem_id}")
        if not (
            restored.scratch_dir("main")
            / SESSION_STATE_SUBDIR
            / "projects"
            / f"-c-w-{workspace_name}"
            / f"{session_id}.jsonl"
        ).is_file():
            raise RuntimeError(f"Relocated transcript missing for {problem_id}")
    finally:
        restored.close()
    return checkpoint_path.name, spent, session_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-results", required=True, type=Path)
    parser.add_argument("--legacy-scratch", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--problems", required=True)
    args = parser.parse_args()

    for problem_id in args.problems.split(","):
        attempt, spent, session_id = migrate_one(
            args.legacy_results,
            args.legacy_scratch,
            problem_id.strip(),
            args.checkpoint_root,
        )
        print(
            f"{problem_id}: checkpoint={attempt} tokens={spent} "
            f"session={session_id}"
        )


if __name__ == "__main__":
    main()
