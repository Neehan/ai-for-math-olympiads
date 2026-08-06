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
    SCRATCH_SUBDIR,
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


def _transcript_turns(transcript: Path) -> list[tuple[str, str]]:
    """Return top-level prompt/assistant-text pairs from a Claude transcript."""
    turns: list[tuple[str, list[str]]] = []
    for line in transcript.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        message = record.get("message") or {}
        content = message.get("content")
        if record.get("type") == "user" and isinstance(content, str):
            turns.append((content, []))
            continue
        if (
            turns
            and record.get("type") == "assistant"
            and isinstance(content, list)
        ):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    turns[-1][1].append(str(block.get("text", "")))
    return [(prompt, "\n".join(parts)) for prompt, parts in turns]


def _validate_interrupted_boundary(
    transcript: Path,
    committed_phases: list[PhaseResult],
    pending_phase: PhaseResult,
) -> None:
    """Prove the snapshot ends after the committed phase and pending prompt."""
    turns = _transcript_turns(transcript)
    if len(turns) < 2:
        raise ValueError("Snapshot transcript has no verifiable phase boundary")
    completed_prompt, completed_text = turns[-2]
    pending_prompt, pending_text = turns[-1]
    if completed_prompt != committed_phases[-1].prompt:
        raise ValueError("Snapshot's last completed prompt does not match ledger")
    if completed_text != committed_phases[-1].text:
        raise ValueError("Snapshot's last completed response does not match ledger")
    if pending_prompt != pending_phase.prompt:
        raise ValueError("Snapshot's pending prompt does not match next ledger phase")
    if pending_text.strip():
        raise ValueError("Snapshot pending turn already contains assistant text")


def _discarded_progress(phases: list[PhaseResult]) -> dict[str, object]:
    """Preserve every lost-tail response and tool call as audit evidence."""
    tool_uses: dict[str, object] = {}
    tool_results: dict[str, object] = {}
    for phase_index, phase in enumerate(phases):
        for call_index, call in enumerate(phase.tool_calls):
            call_id = f"legacy-{phase_index}-{call_index}"
            tool_uses[call_id] = {
                "name": call.name,
                "input": call.tool_input,
            }
            tool_results[call_id] = {
                "result": call.result,
                "is_error": call.is_error,
            }
    return {
        "text_parts": [phase.text for phase in phases if phase.text],
        "tool_uses": tool_uses,
        "tool_results": tool_results,
    }


def migrate_one(
    legacy_results: Path,
    legacy_scratch: Path,
    problem_id: str,
    checkpoint_root: Path,
    completed_phases: int | None = None,
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

    all_phases = _read_phases(seed_dir)
    if len(all_phases) != 129 or all_phases[0].label != "solve":
        raise ValueError(f"{problem_id} does not contain solve + 64 full rounds")
    if all_phases[-1].label != "revise" or any(
        phase.budget_exhausted for phase in all_phases
    ):
        raise ValueError(f"{problem_id} has an incomplete terminal phase")
    full_spent = sum(phase.output_tokens for phase in all_phases)
    if full_spent != int(meta["output_tokens_spent"]):
        raise ValueError(f"{problem_id} phase totals disagree with meta.json")
    if all_phases[-1].cumulative_output_tokens != full_spent:
        raise ValueError(f"{problem_id} cumulative token count is inconsistent")

    if completed_phases is None:
        phases = all_phases
        pending_phase: PhaseResult | None = None
    else:
        if not 1 <= completed_phases < len(all_phases):
            raise ValueError("completed_phases must select a nonempty proper prefix")
        phases = all_phases[:completed_phases]
        pending_phase = all_phases[completed_phases]
    spent = sum(phase.output_tokens for phase in phases)
    if phases[-1].cumulative_output_tokens != spent:
        raise ValueError(f"{problem_id} prefix token count is inconsistent")

    scratch_name = str(meta["scratch_dir_name"])
    source_scratch = legacy_scratch / scratch_name
    runtime = source_scratch / SESSION_STATE_SUBDIR
    session_id, source_transcript = _single_transcript(runtime)
    if pending_phase is not None:
        _validate_interrupted_boundary(
            source_transcript, phases, pending_phase
        )
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
        # Files created after the transcript snapshot are still legitimate
        # products of charged model compute. Preserve them while retaining the
        # older, internally consistent provider transcript.
        final_scratch = seed_dir / SCRATCH_SUBDIR
        if pending_phase is not None:
            if not final_scratch.is_dir():
                raise FileNotFoundError(final_scratch)
            shutil.copytree(final_scratch, destination, dirs_exist_ok=True)

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

        discarded_tail_tokens = 0
        if pending_phase is not None:
            discarded_tail = all_phases[completed_phases:]
            discarded_tail_tokens = full_spent - spent
            stop_at = (
                tracker.budget_tokens
                if pending_phase.label == PHASE_WRAP_UP
                else tracker.soft_limit_tokens
            )
            checkpoint.begin_phase(
                "main",
                pending_phase.label,
                pending_phase.prompt,
                stop_at,
                tracker,
            )
            # Charge the entire post-snapshot branch as discarded compute. On
            # resume, the pending response is regenerated from the exact saved
            # transcript boundary without granting additional budget.
            tracker.add(None, {"output_tokens": discarded_tail_tokens})
            for phase in discarded_tail:
                reconnects.extend(phase.reconnects)
            checkpoint.save_progress(
                "main",
                tracker,
                session_id,
                reconnects,
                _discarded_progress(discarded_tail),
            )

        checkpoint.data()["legacy_migration"] = {
            "source": "64-round controller",
            "problem_id": problem_id,
            "source_scratch_name": scratch_name,
            "imported_phases": len(phases),
            "imported_output_tokens": spent,
            "discarded_tail_output_tokens": discarded_tail_tokens,
            "full_legacy_output_tokens": full_spent,
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
        expected_spent = full_spent if pending_phase is not None else spent
        if (
            len(restored_phases) != len(phases)
            or restored_tracker.spent != expected_spent
        ):
            raise RuntimeError(f"Post-import verification failed for {problem_id}")
        restored_active = restored.active("main")
        if pending_phase is None and restored_active is not None:
            raise RuntimeError(f"Unexpected active phase for {problem_id}")
        if pending_phase is not None and (
            restored_active is None
            or restored_active.get("label") != pending_phase.label
            or restored_active.get("prompt") != pending_phase.prompt
        ):
            raise RuntimeError(f"Pending phase verification failed for {problem_id}")
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
    return checkpoint_path.name, expected_spent, session_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-results", required=True, type=Path)
    parser.add_argument("--legacy-scratch", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--problems", required=True)
    parser.add_argument(
        "--completed-phases",
        type=int,
        help=(
            "Import only this exact transcript-backed phase prefix and charge "
            "the remaining legacy tail as discarded compute (one problem only)"
        ),
    )
    args = parser.parse_args()

    problem_ids = [item.strip() for item in args.problems.split(",") if item.strip()]
    if args.completed_phases is not None and len(problem_ids) != 1:
        raise ValueError("--completed-phases requires exactly one problem")
    for problem_id in problem_ids:
        attempt, spent, session_id = migrate_one(
            args.legacy_results,
            args.legacy_scratch,
            problem_id,
            args.checkpoint_root,
            completed_phases=args.completed_phases,
        )
        print(
            f"{problem_id}: checkpoint={attempt} tokens={spent} "
            f"session={session_id}"
        )


if __name__ == "__main__":
    main()
