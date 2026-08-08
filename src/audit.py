"""Audit entrypoint: grade completed attempts of one arm with a judge model.

Usage:
    python -m src.audit --arm baseline
    python -m src.audit --arm hint --problems usamo-2026-3

Per the paper's grading protocol: the judge is a frontier model OTHER than the
solution's author (enforced in config), is given only the problem statement
and the standalone solution.md (the hint is not included), grades only the
'## Final Solution' section, and scores 7 (complete and rigorous), 6/5
(complete in essence, small obviously-fixable gap), or 0 (anything else — no
other partial credit) with a written note (why valid, or what is
missing/wrong). Each attempt's verdict is audit.json in its seed dir
(resumable marker); the per-arm compiled file is
results/<model>/<arm>/audit.jsonl, one line per (problem, seed).
"""

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path

import anyio

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ResultMessage,
)

from src.checkpoint import AttemptCheckpoint, protocol_fingerprint
from src.concurrency import run_all
from src.config import load_config, override_models
from src.constants import (
    AGENT_SETTINGS_PATH,
    ALLOWED_TOOLS,
    AUDIT_SCORE_INVALID,
    AUDIT_SCORES,
    CLI_PATH_ENV,
    CONFIG_PATH,
    DISALLOWED_TOOLS,
    LOG_FORMAT,
    LOG_LEVEL,
    META_FILENAME,
    MODE_SEQUENTIAL,
    MODE_UNIFORM_STRATEGY,
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
    isolated_session_env,
    process_recovery_prompt,
    provider_model_name,
    token_env_name,
)
from src.storage import (
    archive_audit_scratches,
    budget_cut_multipliers,
    compile_arm_audit,
    cut_solution_path,
    load_problems,
    seed_audited,
    seed_done,
    seed_output_dir,
    seed_solution_text,
    uniform_branch_output_dir,
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
        env=isolated_session_env(config.audit_model, oauth_token, scratch_dir),
        allowed_tools=list(ALLOWED_TOOLS),
        disallowed_tools=list(DISALLOWED_TOOLS),
        settings=str(AGENT_SETTINGS_PATH),
        extra_args={"setting-sources": ""},
        permission_mode=PERMISSION_MODE,
        max_turns=config.audit_max_turns,
        cwd=scratch_dir,
        output_format={"type": "json_schema", "schema": AUDIT_OUTPUT_SCHEMA},
        session_id=session_id,
        resume=resume_session_id,
    )


async def _judge(
    config: ExperimentConfig,
    prompt: str,
    pool: TokenPool,
    scratch_dir: str,
    checkpoint: AttemptCheckpoint,
    role: str,
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
    if result is None or result.is_error:
        raise RuntimeError(f"Judge call failed: {result and result.errors}")
    if not isinstance(result.structured_output, dict):
        raise RuntimeError("Judge returned no structured verdict")
    verdict: dict[str, object] = result.structured_output
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


async def audit_seed(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    pool: TokenPool,
    *,
    output_dir_override: Path | None = None,
    record_extra: dict[str, object] | None = None,
) -> None:
    """Grade one completed attempt (full solution + any budget-cut snapshots).

    Sequential arms carry solution_<m>x.md snapshots for the saturation curve;
    each is judged as its own standalone proof, so every curve point has an
    audit_score + note. A missing snapshot (no complete write-up within that
    budget) is scored invalid with an explanatory note, no judge call spent.
    """
    if arm.mode == MODE_UNIFORM_STRATEGY and output_dir_override is None:
        await audit_uniform_strategy_bank(config, arm, problem, seed, pool)
        return
    output_dir = output_dir_override or seed_output_dir(
        config, arm, problem.problem_id, seed
    )
    solution = seed_solution_text(output_dir)
    cut_multipliers = (
        budget_cut_multipliers(arm.budget_units) if arm.mode == MODE_SEQUENTIAL else []
    )
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
            "output_dir": output_dir.relative_to(RESULTS_ROOT).as_posix(),
            "record_extra": record_extra or {},
            "audit_max_turns": config.audit_max_turns,
            "protocol_fingerprint": protocol_fingerprint(),
        }
    )
    try:
        scratch_paths: dict[str, Path] = {}
        full_scratch = checkpoint.scratch_dir("full")
        scratch_paths["full"] = full_scratch
        verdict, full_reconnects = await _judge(
            config,
            audit_prompt(problem, solution),
            pool,
            str(full_scratch),
            checkpoint,
            "full",
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
            cut_scratch = checkpoint.scratch_dir(role)
            scratch_paths[role] = cut_scratch
            cut_verdict, cut_reconnects = await _judge(
                config,
                audit_prompt(problem, cut_text),
                pool,
                str(cut_scratch),
                checkpoint,
                role,
            )
            cuts[f"{multiplier}x"] = {
                "audit_score": cut_verdict["score"],
                "note": cut_verdict["note"],
                "session_reconnect_count": len(cut_reconnects),
                "session_reconnects": [
                    dataclasses.asdict(event) for event in cut_reconnects
                ],
            }

        checkpoint.prepare_completion(
            (output_dir / SEED_AUDIT_FILENAME).relative_to(RESULTS_ROOT).as_posix()
        )
        archive_audit_scratches(output_dir, scratch_paths)
        audit_record: dict[str, object] = {
            "problem_id": problem.problem_id,
            "arm": arm.name,
            "seed": seed,
            "solver_model": config.model,
            "audit_model": config.audit_model,
            "audit_score": verdict["score"],
            "note": verdict["note"],
            "session_reconnect_count": len(full_reconnects),
            "session_reconnects": [
                dataclasses.asdict(event) for event in full_reconnects
            ],
            "provider_session_ids": checkpoint.session_ids(),
            "process_resume_count": sum(
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


async def audit_uniform_strategy_bank(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    pool: TokenPool,
) -> None:
    """Audit all eight executor candidates and compile one bank verdict."""
    bank_dir = seed_output_dir(config, arm, problem.problem_id, seed)
    strategy_record = json.loads(
        (bank_dir / UNIFORM_STRATEGIES_FILENAME).read_text(encoding="utf-8")
    )
    assignments = [int(value) for value in strategy_record["branch_strategy_indices"]]
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
                "branch_strategy_indices": [],
                "branches": [],
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
        raise ValueError("Uniform Strategy bank has invalid branch assignments")
    tasks = []
    for branch, strategy_index in enumerate(assignments, start=1):
        branch_dir = uniform_branch_output_dir(bank_dir, branch)
        if seed_audited(branch_dir):
            continue
        tasks.append(
            lambda b=branch, i=strategy_index, d=branch_dir: audit_seed(
                config,
                arm,
                problem,
                seed,
                pool,
                output_dir_override=d,
                record_extra={
                    "uniform_strategy_bank_seed": seed,
                    "uniform_strategy_branch": b,
                    "uniform_strategy_index": i,
                },
            )
        )
    await run_all(tasks, min(config.max_concurrency, len(assignments)))

    branches = []
    for branch, strategy_index in enumerate(assignments, start=1):
        branch_dir = uniform_branch_output_dir(bank_dir, branch)
        record = json.loads(
            (branch_dir / SEED_AUDIT_FILENAME).read_text(encoding="utf-8")
        )
        branches.append(
            {
                "branch": branch,
                "strategy_index": strategy_index,
                "audit_score": int(record["audit_score"]),
                "note": str(record["note"]),
            }
        )
    best_score = max(int(record["audit_score"]) for record in branches)
    pass_count = sum(int(record["audit_score"]) >= 5 for record in branches)
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
                f"{len(branches)} executor proofs scored at least 5."
            ),
            "candidate_pass_count": pass_count,
            "candidate_count": len(branches),
            "strategy_count": len(strategies),
            "branch_strategy_indices": assignments,
            "branches": branches,
            "budget_cuts": {},
        },
    )
    log.info(
        "%s/%s seed %d: candidate coverage %d/%d (best score %d)",
        arm.name,
        problem.problem_id,
        seed,
        pass_count,
        len(branches),
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
    args = parser.parse_args()

    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    config = override_models(load_config(CONFIG_PATH), args.model, args.audit_model)
    log.info("Solver model: %s; judge model: %s", config.model, config.audit_model)
    if args.arm not in config.arms:
        raise SystemExit(
            f"Unknown arm '{args.arm}'; config defines {sorted(config.arms)}"
        )
    arm = config.arms[args.arm]
    problems = select_problems(load_problems(), args.problems, args.domain)
    seeds = select_seeds(arm, args.seeds)

    generated = [
        (problem, seed)
        for problem in problems
        for seed in seeds
        if seed_done(seed_output_dir(config, arm, problem.problem_id, seed))
    ]
    ungenerated = len(problems) * len(seeds) - len(generated)
    if ungenerated:
        log.warning(
            "%d attempts have no generation output yet and are skipped", ungenerated
        )
    pending = [
        (problem, seed)
        for problem, seed in generated
        if not seed_audited(seed_output_dir(config, arm, problem.problem_id, seed))
    ]
    log.info(
        "Arm %s: %d attempts to audit, %d already audited",
        arm.name,
        len(pending),
        len(generated) - len(pending),
    )

    pool = TokenPool.from_env(token_env_name(config.audit_model))
    tasks = [
        lambda p=problem, s=seed: audit_seed(config, arm, p, s, pool)
        for problem, seed in pending
    ]
    # A Uniform Strategy bank audits up to eight executor candidates in
    # parallel. One bank controller at a time preserves the global cap.
    outer_limit = 1 if arm.mode == MODE_UNIFORM_STRATEGY else config.max_concurrency
    await run_all(tasks, outer_limit)

    path, count = compile_arm_audit(config, arm)
    log.info("Compiled %d verdicts -> %s", count, path)


if __name__ == "__main__":
    anyio.run(main)
