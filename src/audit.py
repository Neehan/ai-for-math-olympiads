"""Audit entrypoint: grade completed attempts of one arm with a judge model.

Usage:
    python -m src.audit --arm baseline
    python -m src.audit --arm hint --problems usamo-2026-3

Per the paper's grading protocol: the judge is a frontier model OTHER than the
solution's author (enforced in config), is given the problem statement,
the fixed verified reference solution, and the standalone solution.md (the hint and
solver scratch are not included), grades only the '## Final Solution' section,
and scores 7 (complete and rigorous), 6/5
(complete in essence, small obviously-fixable gap), or 0 (anything else — no
other partial credit) with a written note (why valid, or what is
missing/wrong). Each attempt's verdict is audit.json in its seed dir
(resumable marker); the per-arm compiled file is
results/<model>/<arm>/audit.jsonl, one line per (problem, seed).
"""

import argparse
import dataclasses
import hashlib
import json
import logging
import os
from math import comb
from pathlib import Path
from typing import Any

import anyio

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ResultMessage,
)

from src.checkpoint import AttemptCheckpoint, protocol_fingerprint
from src.concurrency import run_all
from src.config import load_config, override_models
from src.constants import (
    ALLOWED_TOOLS,
    AUDIT_SCORE_INVALID,
    AUDIT_SCORES,
    CLI_PATH_ENV,
    CONFIG_PATH,
    LOG_FORMAT,
    LOG_LEVEL,
    MAX_OUTPUT_TOKENS_PER_RESPONSE,
    META_FILENAME,
    MODE_PARALLEL,
    MODE_SEQUENTIAL,
    MODE_UNIFORM_STRATEGY,
    MODE_UNIFORM_STRATEGY_ONLY,
    MODE_UNIFORM_COMPRESS,
    MODE_SELECTION,
    MODE_SELECTION_NO_PROBLEM,
    PARALLEL_BANK_PROTOCOL,
    PERMISSION_MODE,
    RESULTS_ROOT,
    SEED_AUDIT_FILENAME,
    UNIFORM_STRATEGIES_FILENAME,
)
from src.models import ArmConfig, ExperimentConfig, Problem, ReconnectEvent
from src.prompts import audit_prompt
from src.run import select_problems, select_seeds
from src.solver import (
    ResumableClaudeSession,
    StderrTail,
    agent_runtime_policy,
    agent_settings_path,
    auto_compact_window,
    disallowed_tools_for_model,
    isolated_session_env,
    process_recovery_prompt,
    provider_model_name,
    token_env_name,
    uses_vllm,
)
from src.storage import (
    all_budget_cut_multipliers,
    archive_audit_scratches,
    bank_run_output_dir,
    budget_cut_multipliers,
    compile_arm_audit,
    cut_solution_path,
    load_audit_references,
    load_problems,
    materialize_budget_cut_snapshots,
    parallel_bank_audited,
    parallel_bank_done,
    seed_audited,
    seed_done,
    seed_output_dir,
    uniform_strategy_bank_done,
    seed_solution_text,
    write_seed_audit,
)
from src.token_pool import TokenPool

log = logging.getLogger("audit")

# Structured output contract for the judge; enforced by the API, so a verdict
# always parses. Scores restricted to the protocol's 0/5/6/7 scale.
AUDIT_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": list(AUDIT_SCORES)},
        "note": {"type": "string", "minLength": 1},
    },
    "required": ["score", "note"],
    "additionalProperties": False,
}


def _audit_options(
    config: ExperimentConfig,
    oauth_token: str,
    scratch_dir: str,
    stderr_tail: StderrTail,
    *,
    session_id: str | None = None,
    resume_session_id: str | None = None,
    output_schema: dict[str, object] = AUDIT_OUTPUT_SCHEMA,
    allow_tools: bool = True,
    max_output_tokens_per_response: int = MAX_OUTPUT_TOKENS_PER_RESPONSE,
    max_turns: int | None = None,
) -> ClaudeAgentOptions:
    """Judge session options: scratch tools to CHECK (audit, not solve),
    structured 0/5/6/7 output, opaque scratch cwd (never the repo root).

    Same tool policy as the solver — the judge may verify a computation but,
    per prompts/audit.md, a passing check never substitutes for written proof.
    setting_sources exclusion needs extra_args (a falsy [] is dropped).
    """
    return ClaudeAgentOptions(
        model=provider_model_name(config.audit_model),
        cli_path=os.environ.get(CLI_PATH_ENV),
        effort=config.effort,  # type: ignore[arg-type]
        stderr=stderr_tail,
        env=isolated_session_env(
            config.audit_model,
            oauth_token,
            scratch_dir,
            max_output_tokens_per_response,
        ),
        allowed_tools=list(ALLOWED_TOOLS) if allow_tools else [],
        disallowed_tools=[
            *disallowed_tools_for_model(config.audit_model),
            *([] if allow_tools else ALLOWED_TOOLS),
        ],
        settings=str(agent_settings_path(config.audit_model)),
        extra_args={
            "setting-sources": "",
            **(
                {"autocompact": auto_compact_window(config.audit_model)}
                if uses_vllm(config.audit_model)
                else {}
            ),
        },
        permission_mode=PERMISSION_MODE,
        max_turns=max_turns if max_turns is not None else config.audit_max_turns,
        cwd=scratch_dir,
        output_format={"type": "json_schema", "schema": output_schema},
        session_id=session_id,
        resume=resume_session_id,
    )


def _completed_without_verdict(result: ResultMessage) -> bool:
    """Whether a bounded judge call ended after model work but made no decision."""
    if isinstance(result.structured_output, dict):
        return False
    limit_reached = (
        result.stop_reason == "max_tokens"
        or result.subtype == "error_max_turns"
        or any("max turn" in error.casefold() for error in (result.errors or []))
    )
    return limit_reached or (
        not result.is_error
        and result.num_turns > 0
        and not isinstance(result.structured_output, dict)
    )


async def _judge(
    config: ExperimentConfig,
    prompt: str,
    pool: TokenPool,
    scratch_dir: str,
    checkpoint: AttemptCheckpoint,
    role: str,
    *,
    output_schema: dict[str, object] = AUDIT_OUTPUT_SCHEMA,
    allow_tools: bool = True,
    max_output_tokens_per_response: int = MAX_OUTPUT_TOKENS_PER_RESPONSE,
    max_turns: int | None = None,
    terminal_no_verdict: bool = False,
) -> tuple[dict[str, object], list[ReconnectEvent]]:
    """Run one judge call and return its validated structured verdict."""
    saved = checkpoint.call_result(role)
    if saved is not None:
        raw_verdict = saved.get("verdict")
        if not isinstance(raw_verdict, dict):
            raise TypeError("Checkpoint judge verdict is corrupt")
        saved_verdict: dict[str, object] = dict(raw_verdict)
        reconnects = [ReconnectEvent(**item) for item in saved.get("reconnects", [])]
        return saved_verdict, reconnects

    active = checkpoint.active(role)
    process_recovery = active is not None
    if active is None:
        active = checkpoint.begin_call(role, prompt)
    else:
        active = checkpoint.prepare_process_resume(role)

    result: ResultMessage | None = None
    reconnects: list[ReconnectEvent] = []
    async with ResumableClaudeSession(
        pool,
        lambda token, session_id, resume_id, stderr: _audit_options(
            config,
            token,
            scratch_dir,
            stderr,
            session_id=session_id,
            resume_session_id=resume_id,
            output_schema=output_schema,
            allow_tools=allow_tools,
            max_output_tokens_per_response=max_output_tokens_per_response,
            max_turns=max_turns,
        ),
        session_id=checkpoint.session_id(role),
        reconnects=checkpoint.reconnects(role),
    ) as session:
        checkpoint.save_session(role, session.session_id, session.reconnect_events)
        try:
            await session.query(
                process_recovery_prompt(str(active["prompt"]))
                if process_recovery
                else prompt
            )
            async for message in session.receive_response():
                if isinstance(message, ResultMessage):
                    result = message
        finally:
            reconnects = session.reconnect_events
            checkpoint.save_session(role, session.session_id, reconnects)
    if result is None:
        raise RuntimeError(f"Judge call failed: {result and result.errors}")
    if terminal_no_verdict and _completed_without_verdict(result):
        verdict = {
            "decision_status": "no_decision",
            "stop_reason": result.stop_reason or result.subtype,
            "usage": dict(result.usage) if isinstance(result.usage, dict) else {},
        }
        checkpoint.finish_call(
            role,
            {
                "verdict": verdict,
                "reconnects": [dataclasses.asdict(event) for event in reconnects],
                "process_resume_count": int(active.get("process_resume_count", 0)),
            },
            session.session_id,
            reconnects,
        )
        return verdict, reconnects
    if result.is_error and not (
        terminal_no_verdict and isinstance(result.structured_output, dict)
    ):
        raise RuntimeError(f"Judge call failed: {result.errors}")
    if not isinstance(result.structured_output, dict):
        raise RuntimeError("Judge returned no structured verdict")
    verdict: dict[str, object] = dict(result.structured_output)
    if terminal_no_verdict:
        verdict["stop_reason"] = result.stop_reason or result.subtype
        verdict["usage"] = (
            dict(result.usage) if isinstance(result.usage, dict) else {}
        )
    checkpoint.finish_call(
        role,
        {
            "verdict": verdict,
            "reconnects": [dataclasses.asdict(event) for event in reconnects],
            "process_resume_count": int(active.get("process_resume_count", 0)),
        },
        session.session_id,
        reconnects,
    )
    return verdict, reconnects


def _selected_cut_multipliers(
    arm: ArmConfig, all_checkpoints: bool
) -> list[int]:
    if arm.mode != MODE_SEQUENTIAL:
        return []
    return (
        all_budget_cut_multipliers(arm.budget_units)
        if all_checkpoints
        else budget_cut_multipliers(arm.budget_units)
    )


def _audit_has_cut_schedule(
    output_dir: Path, arm: ArmConfig, all_checkpoints: bool
) -> bool:
    path = output_dir / SEED_AUDIT_FILENAME
    if not path.exists():
        return False
    if not all_checkpoints:
        return True
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    cuts = record.get("budget_cuts")
    if not isinstance(cuts, dict):
        return False
    return all(
        isinstance(cuts.get(f"{multiplier}x"), dict)
        for multiplier in all_budget_cut_multipliers(arm.budget_units)
    )


def _saved_reconnects(record: dict[str, object]) -> list[ReconnectEvent]:
    raw = record.get("session_reconnects", [])
    if not isinstance(raw, list):
        raise ValueError("Existing audit has malformed reconnect provenance")
    return [ReconnectEvent(**item) for item in raw if isinstance(item, dict)]


def _seed_existing_verdicts(
    output_dir: Path,
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    solution: str,
    cut_artifacts: dict[str, str | None],
) -> tuple[
    dict[str, tuple[dict[str, object], list[ReconnectEvent], str, str]],
    dict[str, object] | None,
]:
    """Reuse frozen sparse verdicts when extending to dense checkpoints."""
    path = output_dir / SEED_AUDIT_FILENAME
    if not path.exists():
        return {}, None
    existing: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "problem_id": problem.problem_id,
        "arm": arm.name,
        "seed": seed,
        "solver_model": config.model,
    }
    if any(existing.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing audit identity does not match this attempt")

    existing_model = existing.get("audit_model")
    if not isinstance(existing_model, str):
        raise ValueError("Existing audit has no judge-model provenance")
    cache: dict[
        str, tuple[dict[str, object], list[ReconnectEvent], str, str]
    ] = {}

    def add(text: str, record: dict[str, object], source: str) -> None:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if record.get("solution_sha256") != digest:
            return
        score = record.get("audit_score")
        note = record.get("note")
        if not isinstance(score, int) or not isinstance(note, str):
            raise ValueError("Existing audit verdict is malformed")
        record_model = record.get("audit_model", existing_model)
        if not isinstance(record_model, str):
            raise ValueError("Existing cut audit has malformed judge provenance")
        value = (
            {"score": score, "note": note},
            _saved_reconnects(record),
            source,
            record_model,
        )
        previous = cache.get(digest)
        if previous is not None and previous[0] != value[0]:
            raise ValueError("Identical proof has inconsistent existing verdicts")
        if previous is None:
            cache[digest] = value

    add(solution, existing, "full")
    raw_cuts = existing.get("budget_cuts", {})
    if not isinstance(raw_cuts, dict):
        raise ValueError("Existing audit has malformed budget cuts")
    for label, text in cut_artifacts.items():
        raw_record = raw_cuts.get(label)
        if text is not None and isinstance(raw_record, dict):
            add(text, raw_record, f"cut_{label}")
    return cache, existing


async def audit_seed(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    pool: TokenPool,
    reference_solution: str,
    *,
    output_dir_override: Path | None = None,
    record_extra: dict[str, object] | None = None,
    all_checkpoints: bool = False,
) -> None:
    """Grade one completed attempt (full solution + any budget-cut snapshots).

    Sequential arms carry solution_<m>x.md snapshots for the saturation curve;
    each unique standalone proof is judged once, and byte-identical snapshots
    reuse that verdict. A missing snapshot (no complete write-up within that
    budget) is scored invalid with an explanatory note, no judge call spent.
    """
    if arm.mode == MODE_PARALLEL and output_dir_override is None:
        await audit_parallel_bank(
            config, arm, problem, seed, pool, reference_solution
        )
        return
    if arm.mode == MODE_UNIFORM_STRATEGY and output_dir_override is None:
        await audit_uniform_strategy_bank(
            config, arm, problem, seed, pool, reference_solution
        )
        return
    output_dir = output_dir_override or seed_output_dir(
        config, arm, problem.problem_id, seed
    )
    if all_checkpoints:
        if arm.mode != MODE_SEQUENTIAL:
            raise ValueError("--all-checkpoints is valid only for sequential arms")
        materialize_budget_cut_snapshots(
            config,
            arm,
            output_dir,
            all_budget_cut_multipliers(arm.budget_units),
        )
    solution = seed_solution_text(output_dir)
    cut_multipliers = _selected_cut_multipliers(arm, all_checkpoints)
    cut_artifacts = {
        f"{multiplier}x": (
            cut_solution_path(output_dir, multiplier).read_text(encoding="utf-8")
            if cut_solution_path(output_dir, multiplier).exists()
            else None
        )
        for multiplier in cut_multipliers
    }
    if not solution.strip():
        cuts = {
            f"{multiplier}x": {
                "audit_score": AUDIT_SCORE_INVALID,
                "note": "No complete write-up was emitted within this budget cut.",
                "session_reconnect_count": 0,
                "session_reconnects": [],
            }
            for multiplier in cut_multipliers
        }
        empty_record: dict[str, object] = {
            "problem_id": problem.problem_id,
            "arm": arm.name,
            "seed": seed,
            "solver_model": config.model,
            "audit_model": config.audit_model,
            "audit_score": AUDIT_SCORE_INVALID,
            "note": "No complete write-up was emitted within the attempt budget.",
            "session_reconnect_count": 0,
            "session_reconnects": [],
            "provider_session_ids": {},
            "process_resume_count": 0,
            "budget_cuts": cuts,
        }
        if record_extra:
            empty_record.update(record_extra)
        write_seed_audit(
            output_dir,
            empty_record,
        )
        log.info(
            "%s/%s seed %d: score 0 (no gradeable write-up)",
            arm.name,
            problem.problem_id,
            seed,
        )
        return
    checkpoint = AttemptCheckpoint(
        {
            "stage": "audit",
            "solver_model": config.model,
            "audit_model": config.audit_model,
            "effort": config.effort,
            "arm": dataclasses.asdict(arm),
            "problem_id": problem.problem_id,
            "problem_statement": problem.statement,
            "seed": seed,
            "solution": solution,
            "budget_cut_artifacts": cut_artifacts,
            "all_checkpoints": all_checkpoints,
            "output_dir": output_dir.relative_to(RESULTS_ROOT).as_posix(),
            "record_extra": record_extra or {},
            "audit_max_turns": config.audit_max_turns,
            "protocol_fingerprint": protocol_fingerprint(
                agent_settings_path(config.audit_model)
            ),
            **(
                {"agent_runtime_policy": agent_runtime_policy(config.audit_model)}
                if uses_vllm(config.audit_model)
                else {}
            ),
        }
    )
    try:
        scratch_paths: dict[str, Path] = {}
        full_digest = hashlib.sha256(solution.encode("utf-8")).hexdigest()
        if all_checkpoints:
            verdict_cache, existing_audit = _seed_existing_verdicts(
                output_dir,
                config,
                arm,
                problem,
                seed,
                solution,
                cut_artifacts,
            )
        else:
            verdict_cache, existing_audit = {}, None
        full_cached = verdict_cache.get(full_digest)
        if full_cached is None:
            full_scratch = checkpoint.scratch_dir("full")
            scratch_paths["full"] = full_scratch
            verdict, full_reconnects = await _judge(
                config,
                audit_prompt(problem, reference_solution, solution),
                pool,
                str(full_scratch),
                checkpoint,
                "full",
            )
            full_audit_model = config.audit_model
            verdict_cache[full_digest] = (
                verdict,
                full_reconnects,
                "full",
                full_audit_model,
            )
        else:
            verdict, full_reconnects, _, full_audit_model = full_cached
            log.info(
                "%s/%s seed %d: reusing frozen full-proof audit",
                arm.name,
                problem.problem_id,
                seed,
            )

        cuts: dict[str, dict[str, object]] = {}
        for multiplier in cut_multipliers:
            cut_text = cut_artifacts[f"{multiplier}x"]
            if cut_text is None:
                cuts[f"{multiplier}x"] = {
                    "audit_score": AUDIT_SCORE_INVALID,
                    "note": "No complete write-up was emitted within this budget cut.",
                    "session_reconnect_count": 0,
                    "session_reconnects": [],
                }
                continue
            role = f"cut_{multiplier}x"
            cut_digest = hashlib.sha256(cut_text.encode("utf-8")).hexdigest()
            cached = verdict_cache.get(cut_digest)
            if cached is None:
                cut_scratch = checkpoint.scratch_dir(role)
                scratch_paths[role] = cut_scratch
                cut_verdict, cut_reconnects = await _judge(
                    config,
                    audit_prompt(problem, reference_solution, cut_text),
                    pool,
                    str(cut_scratch),
                    checkpoint,
                    role,
                )
                cut_audit_model = config.audit_model
                verdict_cache[cut_digest] = (
                    cut_verdict,
                    cut_reconnects,
                    role,
                    cut_audit_model,
                )
                reused_from: str | None = None
            else:
                (
                    cut_verdict,
                    cut_reconnects,
                    reused_from,
                    cut_audit_model,
                ) = cached
                log.info(
                    "%s/%s seed %d: reusing %s audit for identical %s proof",
                    arm.name,
                    problem.problem_id,
                    seed,
                    reused_from,
                    role,
                )
            cuts[f"{multiplier}x"] = {
                "audit_score": cut_verdict["score"],
                "note": cut_verdict["note"],
                "session_reconnect_count": len(cut_reconnects),
                "session_reconnects": [
                    dataclasses.asdict(event) for event in cut_reconnects
                ],
                "solution_sha256": cut_digest,
                "audit_model": cut_audit_model,
                **(
                    {"audit_reused_from": reused_from}
                    if reused_from is not None
                    else {}
                ),
            }

        checkpoint.prepare_completion(
            (output_dir / SEED_AUDIT_FILENAME).relative_to(RESULTS_ROOT).as_posix()
        )
        archive_audit_scratches(
            output_dir,
            scratch_paths,
            preserve_existing=all_checkpoints and existing_audit is not None,
        )
        provider_session_ids: dict[str, str] = {}
        raw_session_ids = (
            existing_audit.get("provider_session_ids")
            if existing_audit is not None
            else None
        )
        if isinstance(raw_session_ids, dict):
            provider_session_ids.update(
                {
                    key: value
                    for key, value in raw_session_ids.items()
                    if isinstance(key, str) and isinstance(value, str)
                }
            )
        provider_session_ids.update(checkpoint.session_ids())
        raw_resume_count = (
            existing_audit.get("process_resume_count")
            if existing_audit is not None
            else None
        )
        existing_resume_count = (
            raw_resume_count
            if isinstance(raw_resume_count, int) and not isinstance(raw_resume_count, bool)
            else 0
        )
        audit_record: dict[str, object] = {
            "problem_id": problem.problem_id,
            "arm": arm.name,
            "seed": seed,
            "solver_model": config.model,
            "audit_model": full_audit_model,
            "audit_score": verdict["score"],
            "note": verdict["note"],
            "solution_sha256": full_digest,
            "session_reconnect_count": len(full_reconnects),
            "session_reconnects": [
                dataclasses.asdict(event) for event in full_reconnects
            ],
            "provider_session_ids": provider_session_ids,
            "process_resume_count": existing_resume_count + sum(
                int(record.get("process_resume_count", 0))
                for record in checkpoint.data().get("calls", {}).values()
            ),
            "budget_cuts": cuts,
        }
        if record_extra:
            audit_record.update(record_extra)
        write_seed_audit(output_dir, audit_record)
        log.info(
            "%s/%s seed %d: score %s (%d cut snapshots graded)",
            arm.name,
            problem.problem_id,
            seed,
            verdict["score"],
            len(cuts),
        )
        checkpoint.complete()
    finally:
        checkpoint.close()


def _prefix_summary(
    records: list[dict[str, Any]],
    sizes: tuple[int, ...],
    *,
    key_template: str = "{size}x",
) -> dict[str, dict[str, object]]:
    """Best-score/pass-count summaries for prespecified candidate prefixes."""
    summaries: dict[str, dict[str, object]] = {}
    for size in sizes:
        prefix = records[:size]
        scores = [int(record["audit_score"]) for record in prefix]
        summaries[key_template.format(size=size)] = {
            "audit_score": max(scores),
            "candidate_pass_count": sum(score >= 5 for score in scores),
            "candidate_count": size,
            "note": f"Best of the first {size} prespecified candidate(s).",
        }
    return summaries


def _bank_audit_record(
    path: Path, expected: dict[str, object], label: str
) -> dict[str, Any]:
    """Load one candidate verdict and reject stale/cross-bank audit files."""
    record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"expected": value, "actual": record.get(key)}
        for key, value in expected.items()
        if record.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label} audit identity mismatch: {mismatches}")
    return record


async def audit_parallel_bank(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    pool: TokenPool,
    reference_solution: str,
) -> None:
    """Audit all eight fresh candidates in one Parallel-8 bank."""
    bank_dir = seed_output_dir(config, arm, problem.problem_id, seed)
    if not parallel_bank_done(bank_dir):
        raise FileNotFoundError(
            f"Parallel bank is incomplete or uses a retired protocol: {bank_dir}"
        )

    tasks = []
    for run in range(1, 9):
        run_dir = bank_run_output_dir(bank_dir, run)
        if not seed_done(run_dir):
            raise FileNotFoundError(f"Parallel run_{run:02d} is incomplete: {run_dir}")
        if seed_audited(run_dir):
            continue
        tasks.append(
            lambda r=run, d=run_dir: audit_seed(
                config,
                arm,
                problem,
                seed,
                pool,
                reference_solution,
                output_dir_override=d,
                record_extra={"parallel_bank_seed": seed, "parallel_run": r},
            )
        )
    await run_all(tasks, min(config.max_concurrency, 8))

    runs: list[dict[str, Any]] = []
    for run in range(1, 9):
        run_dir = bank_run_output_dir(bank_dir, run)
        expected = {
            "problem_id": problem.problem_id,
            "arm": arm.name,
            "seed": seed,
            "solver_model": config.model,
            "audit_model": config.audit_model,
            "parallel_bank_seed": seed,
            "parallel_run": run,
        }
        record = _bank_audit_record(
            run_dir / SEED_AUDIT_FILENAME,
            expected,
            f"Parallel run_{run:02d}",
        )
        runs.append(
            {
                "run": run,
                "audit_score": int(record["audit_score"]),
                "note": str(record["note"]),
            }
        )
    scores = [int(record["audit_score"]) for record in runs]
    pass_count = sum(score >= 5 for score in scores)
    first_success = next(
        (run for run, score in enumerate(scores, start=1) if score >= 5), None
    )
    pass_at_k = {
        str(k): (
            1.0
            if 8 - pass_count < k
            else 1.0 - comb(8 - pass_count, k) / comb(8, k)
        )
        for k in (1, 2, 4, 8)
    }
    write_seed_audit(
        bank_dir,
        {
            "problem_id": problem.problem_id,
            "arm": arm.name,
            "seed": seed,
            "solver_model": config.model,
            "audit_model": config.audit_model,
            "parallel_bank_protocol": PARALLEL_BANK_PROTOCOL,
            "audit_score": max(scores),
            "note": (
                f"Parallel candidate coverage: {pass_count}/8 proofs scored at "
                "least 5."
            ),
            "candidate_pass_count": pass_count,
            "candidate_count": 8,
            "first_success_run": first_success,
            "pass_at_k": pass_at_k,
            "runs": runs,
            "budget_cuts": {},
        },
    )
    log.info(
        "%s/%s bank seed %d: candidate coverage %d/8 (best score %d)",
        arm.name,
        problem.problem_id,
        seed,
        pass_count,
        max(scores),
    )


async def audit_uniform_strategy_bank(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    pool: TokenPool,
    reference_solution: str,
) -> None:
    """Audit all eight executor candidates and compile one bank verdict."""
    bank_dir = seed_output_dir(config, arm, problem.problem_id, seed)
    strategy_record = json.loads(
        (bank_dir / UNIFORM_STRATEGIES_FILENAME).read_text(encoding="utf-8")
    )
    assignments = [int(value) for value in strategy_record["run_strategy_indices"]]
    strategies = strategy_record["strategies"]
    if not isinstance(strategies, list):
        raise ValueError("Uniform Strategy bank strategies must be a list")
    if not strategies:
        bank_meta = json.loads((bank_dir / META_FILENAME).read_text(encoding="utf-8"))
        failure = str(bank_meta.get("planner_failure", "no eligible strategy set"))
        write_seed_audit(
            bank_dir,
            {
                "problem_id": problem.problem_id,
                "arm": arm.name,
                "seed": seed,
                "solver_model": config.model,
                "audit_model": config.audit_model,
                "audit_score": 0,
                "note": f"Uniform Strategy planner failure: {failure}",
                "candidate_pass_count": 0,
                "candidate_count": 0,
                "strategy_count": 0,
                "run_strategy_indices": [],
                "runs": [],
                "budget_cuts": {},
            },
        )
        log.info(
            "%s/%s seed %d: planner failure (no candidates)",
            arm.name,
            problem.problem_id,
            seed,
        )
        return
    if len(assignments) != config.uniform_strategy_branches or any(
        index < 1 or index > len(strategies) for index in assignments
    ):
        raise ValueError("Uniform Strategy bank has invalid run assignments")
    tasks = []
    for run, strategy_index in enumerate(assignments, start=1):
        run_dir = bank_run_output_dir(bank_dir, run)
        if seed_audited(run_dir):
            continue
        tasks.append(
            lambda r=run, i=strategy_index, d=run_dir: audit_seed(
                config,
                arm,
                problem,
                seed,
                pool,
                reference_solution,
                output_dir_override=d,
                record_extra={
                    "uniform_strategy_bank_seed": seed,
                    "uniform_strategy_run": r,
                    "uniform_strategy_index": i,
                },
            )
        )
    await run_all(tasks, min(config.max_concurrency, len(assignments)))

    runs = []
    for run, strategy_index in enumerate(assignments, start=1):
        run_dir = bank_run_output_dir(bank_dir, run)
        record = _bank_audit_record(
            run_dir / SEED_AUDIT_FILENAME,
            {
                "problem_id": problem.problem_id,
                "arm": arm.name,
                "seed": seed,
                "solver_model": config.model,
                "audit_model": config.audit_model,
                "uniform_strategy_bank_seed": seed,
                "uniform_strategy_run": run,
                "uniform_strategy_index": strategy_index,
            },
            f"Uniform Strategy run_{run:02d}",
        )
        runs.append(
            {
                "run": run,
                "strategy_index": strategy_index,
                "audit_score": int(record["audit_score"]),
                "note": str(record["note"]),
            }
        )
    best_score = max(int(record["audit_score"]) for record in runs)
    pass_count = sum(int(record["audit_score"]) >= 5 for record in runs)
    write_seed_audit(
        bank_dir,
        {
            "problem_id": problem.problem_id,
            "arm": arm.name,
            "seed": seed,
            "solver_model": config.model,
            "audit_model": config.audit_model,
            "audit_score": best_score,
            "note": (
                f"Audited candidate coverage: {pass_count}/"
                f"{len(runs)} executor proofs scored at least 5."
            ),
            "candidate_pass_count": pass_count,
            "candidate_count": len(runs),
            "strategy_count": len(strategies),
            "run_strategy_indices": assignments,
            "runs": runs,
            "candidate_prefixes": _prefix_summary(
                runs, (1, 2, 4, 8), key_template="first_{size}_runs"
            ),
            "budget_cuts": {},
        },
    )
    log.info(
        "%s/%s seed %d: candidate coverage %d/%d (best score %d)",
        arm.name,
        problem.problem_id,
        seed,
        pass_count,
        len(runs),
        best_score,
    )


async def main() -> None:
    """Grade every completed-but-unaudited attempt of one arm, then compile."""
    parser = argparse.ArgumentParser(description="Audit one experiment arm.")
    parser.add_argument("--arm", required=True, help="Arm name from config.json")
    parser.add_argument(
        "--problems", default=None, help="Comma-separated problem ids (default: all)"
    )
    parser.add_argument(
        "--domain", default=None, help="Only problems in this domain (default: all)"
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed subset to audit (default: the arm's seeds)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Solver model override — whose results tree to grade (default: config.json)",
    )
    parser.add_argument(
        "--audit-model",
        default=None,
        help="Judge model override (default: config.json); must differ from the solver",
    )
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="For sequential arms, audit every integer 1x,...,8x checkpoint",
    )
    args = parser.parse_args()

    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    config = override_models(load_config(CONFIG_PATH), args.model, args.audit_model)
    log.info("Solver model: %s; judge model: %s", config.model, config.audit_model)
    if args.arm not in config.arms:
        raise SystemExit(
            f"Unknown arm '{args.arm}'; config defines {sorted(config.arms)}"
        )
    arm = config.arms[args.arm]
    if arm.mode in {MODE_UNIFORM_STRATEGY_ONLY, MODE_UNIFORM_COMPRESS}:
        log.info("Arm %s has no proof artifacts; strategy audit follows", arm.name)
        return
    if arm.mode in {MODE_SELECTION, MODE_SELECTION_NO_PROBLEM}:
        path, count = compile_arm_audit(config, arm)
        log.info("Compiled %d deterministic selection verdicts -> %s", count, path)
        return
    if args.all_checkpoints and arm.mode != MODE_SEQUENTIAL:
        raise SystemExit("--all-checkpoints is valid only for sequential arms")
    all_problems = load_problems()
    problems = select_problems(all_problems, args.problems, args.domain)
    audit_references = load_audit_references()
    for problem in all_problems:
        reference = audit_references.get(problem.problem_id)
        if reference is None:
            raise SystemExit(f"No reference solution for {problem.problem_id}")
        reference_statement, _ = reference
        if reference_statement != problem.statement.strip():
            raise SystemExit(
                f"Problem statement mismatch in hard_solutions for "
                f"{problem.problem_id}"
            )
    seeds = select_seeds(arm, args.seeds)

    def generation_done(output_dir: Path) -> bool:
        if arm.mode == MODE_PARALLEL:
            return parallel_bank_done(output_dir)
        if arm.mode == MODE_UNIFORM_STRATEGY:
            return uniform_strategy_bank_done(output_dir)
        return seed_done(output_dir)

    def audit_done(output_dir: Path) -> bool:
        return (
            parallel_bank_audited(output_dir)
            if arm.mode == MODE_PARALLEL
            else _audit_has_cut_schedule(output_dir, arm, args.all_checkpoints)
        )

    generated = [
        (problem, seed)
        for problem in problems
        for seed in seeds
        if generation_done(seed_output_dir(config, arm, problem.problem_id, seed))
    ]
    ungenerated = len(problems) * len(seeds) - len(generated)
    if ungenerated:
        log.warning(
            "%d attempts have no generation output yet and are skipped", ungenerated
        )
    pending = [
        (problem, seed)
        for problem, seed in generated
        if not audit_done(seed_output_dir(config, arm, problem.problem_id, seed))
    ]
    log.info(
        "Arm %s: %d attempts to audit, %d already audited",
        arm.name,
        len(pending),
        len(generated) - len(pending),
    )

    pool = TokenPool.from_env(token_env_name(config.audit_model))
    tasks = [
        lambda p=problem, s=seed: audit_seed(
            config,
            arm,
            p,
            s,
            pool,
            audit_references[p.problem_id][1],
            all_checkpoints=args.all_checkpoints,
        )
        for problem, seed in pending
    ]
    # Parallel and Uniform Strategy banks audit their candidates internally.
    # One bank controller at a time preserves the global concurrency cap.
    outer_limit = (
        1
        if arm.mode in {MODE_PARALLEL, MODE_UNIFORM_STRATEGY}
        else config.max_concurrency
    )
    await run_all(tasks, outer_limit)

    path, count = compile_arm_audit(config, arm)
    log.info("Compiled %d verdicts -> %s", count, path)
    failed = [
        (problem.problem_id, seed)
        for problem, seed in pending
        if not audit_done(seed_output_dir(config, arm, problem.problem_id, seed))
    ]
    if failed:
        raise SystemExit(
            f"{len(failed)} correctness-audit task(s) failed; rerun the same command"
        )


if __name__ == "__main__":
    anyio.run(main)
