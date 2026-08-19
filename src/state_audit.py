"""Annotate observable U/P/S route states for sequential proof checkpoints.

Correctness remains owned by ``audit.json``.  This stage reuses those immutable
scores: passing checkpoints become S without another model call; missing
solution text remains unobserved; only nonempty score-below-5 artifacts receive
a reference-guided three-step outline annotation.  The final state is
derived by code, not selected by the annotator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import anyio

from src.audit import _judge
from src.checkpoint import AttemptCheckpoint
from src.concurrency import run_all
from src.config import load_config, override_models
from src.constants import (
    CONFIG_PATH,
    LOG_FORMAT,
    LOG_LEVEL,
    MODE_SEQUENTIAL,
    RESULTS_ROOT,
    SEED_AUDIT_FILENAME,
    SEED_STATE_AUDIT_FILENAME,
)
from src.models import ArmConfig, ExperimentConfig, Problem
from src.prompts import state_audit_prompt
from src.run import select_problems, select_seeds
from src.solver import token_env_name
from src.storage import (
    budget_cut_multipliers,
    compile_arm_state_audit,
    cut_solution_path,
    load_problems,
    load_state_audit_references,
    seed_done,
    seed_output_dir,
    seed_solution_text,
    write_seed_state_audit,
)
from src.token_pool import TokenPool

log = logging.getLogger("state_audit")

STATE_AUDIT_ARMS = {"baseline-sequential", "hint-sequential"}

STATE_AUDIT_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "present": {"type": "boolean"},
                    "reason": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 600,
                    },
                },
                "required": ["present", "reason"],
                "additionalProperties": False,
            },
            "minItems": 3,
            "maxItems": 3,
        },
    },
    "required": ["steps"],
    "additionalProperties": False,
}


def _outline(problem: Problem) -> str:
    outline = (problem.hint_h3 or "").strip()
    steps = [line for line in outline.splitlines() if line.strip()]
    if len(steps) != 3:
        raise ValueError(
            f"{problem.problem_id}: state audit requires exactly three route steps"
        )
    return outline


def derive_state(verdict: dict[str, object]) -> str:
    """Map the three step-presence decisions mechanically to P or U."""
    steps = verdict.get("steps")
    if not isinstance(steps, list) or len(steps) != 3:
        raise ValueError("Malformed state-audit verdict")
    for step in steps:
        if (
            not isinstance(step, dict)
            or not isinstance(step.get("present"), bool)
            or not isinstance(step.get("reason"), str)
            or not str(step["reason"]).strip()
        ):
            raise ValueError("Malformed state-audit step")
    return "P" if all(bool(step["present"]) for step in steps) else "U"


def _missing_checkpoint() -> dict[str, object]:
    return {
        "state": None,
        "steps": [],
        "note": "No complete solution text was emitted at this budget; state is unobserved.",
    }


def _solution_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _solved_checkpoint(text: str) -> dict[str, object]:
    return {
        "state": "S",
        "steps": [],
        "note": "Correctness audit passed; state assigned mechanically as solved.",
        "solution_sha256": _solution_sha256(text),
    }


def _proof_artifacts(
    arm: ArmConfig,
    output_dir: Path,
    proof_audit: dict[str, Any],
) -> list[tuple[str, str | None, int]]:
    """Read checkpoint texts and their already-completed correctness scores."""
    artifacts: list[tuple[str, str | None, int]] = []
    raw_cuts = proof_audit.get("budget_cuts", {})
    if not isinstance(raw_cuts, dict):
        raise ValueError("Proof audit has malformed budget_cuts")
    for multiplier in budget_cut_multipliers(arm.budget_units):
        label = f"{multiplier}x"
        cut = raw_cuts.get(label)
        if not isinstance(cut, dict) or not isinstance(cut.get("audit_score"), int):
            raise ValueError(f"Proof audit is missing {label}")
        path = cut_solution_path(output_dir, multiplier)
        text = path.read_text(encoding="utf-8") if path.exists() else None
        score = int(cut["audit_score"])
        artifacts.append((label, text, score))
    full_text = seed_solution_text(output_dir)
    full_score = proof_audit.get("audit_score")
    if not isinstance(full_score, int):
        raise ValueError("Proof audit is missing its full score")
    full_label = f"{arm.budget_units}x"
    artifacts.append((full_label, full_text, full_score))
    return artifacts


async def state_audit_seed(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    pool: TokenPool,
    reference_solution: str,
) -> None:
    """Annotate every checkpoint of one already correctness-audited trajectory."""
    output_dir = seed_output_dir(config, arm, problem.problem_id, seed)
    proof_path = output_dir / SEED_AUDIT_FILENAME
    proof_audit: dict[str, Any] = json.loads(proof_path.read_text(encoding="utf-8"))
    outline = _outline(problem)
    state_path = output_dir / SEED_STATE_AUDIT_FILENAME
    artifacts = _proof_artifacts(arm, output_dir, proof_audit)
    if state_path.exists():
        return
    checkpoint = AttemptCheckpoint(
        {
            "stage": "state_audit",
            "solver_model": config.model,
            "state_audit_model": config.audit_model,
            "arm": arm.name,
            "problem_id": problem.problem_id,
            "seed": seed,
        }
    )
    try:
        records: dict[str, dict[str, object]] = {}
        verdict_cache: dict[str, dict[str, object]] = {}
        for label, text, proof_score in artifacts:
            if text is None or not text.strip():
                if proof_score >= 5:
                    raise ValueError(
                        f"{problem.problem_id} {label}: passing proof audit has "
                        "no solution text"
                    )
                records[label] = _missing_checkpoint()
                continue
            if proof_score >= 5:
                records[label] = _solved_checkpoint(text)
                continue

            cached = verdict_cache.get(text)
            if cached is None:
                role = f"state_{label}"
                scratch = checkpoint.scratch_dir(role)
                verdict, _ = await _judge(
                    config,
                    state_audit_prompt(
                        problem, outline, reference_solution, text
                    ),
                    pool,
                    str(scratch),
                    checkpoint,
                    role,
                    output_schema=STATE_AUDIT_OUTPUT_SCHEMA,
                    allow_tools=False,
                )
                verdict_cache[text] = verdict
            else:
                verdict = cached

            state = derive_state(verdict)
            raw_steps = verdict["steps"]
            if not isinstance(raw_steps, list):
                raise TypeError("State-audit steps are not a list")
            record: dict[str, object] = {
                "state": state,
                "steps": list(raw_steps),
                "note": (
                    "All three frozen outline steps are explicitly recognized."
                    if state == "P"
                    else "At least one frozen outline step is not recognized."
                ),
                "solution_sha256": _solution_sha256(text),
            }
            records[label] = record

        final_label = f"{arm.budget_units}x"
        final_checkpoint = records.pop(final_label)
        final_record: dict[str, object] = {
            "problem_id": problem.problem_id,
            "arm": arm.name,
            "seed": seed,
            "solver_model": config.model,
            "audit_model": config.audit_model,
            **final_checkpoint,
            "budget_cuts": records,
        }
        checkpoint.prepare_completion(
            state_path.relative_to(RESULTS_ROOT).as_posix()
        )
        write_seed_state_audit(output_dir, final_record)
        checkpoint.complete()
        log.info("%s/%s seed %d: state audit complete", arm.name, problem.problem_id, seed)
    finally:
        checkpoint.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description="State-audit one sequential arm.")
    parser.add_argument("--arm", required=True, help="Sequential arm name")
    parser.add_argument("--problems", default=None, help="Comma-separated problem ids")
    parser.add_argument("--domain", default=None, help="Only problems in this domain")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed subset")
    parser.add_argument("--model", default=None, help="Solver model override")
    parser.add_argument("--audit-model", default=None, help="State annotator override")
    args = parser.parse_args()

    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    config = override_models(load_config(CONFIG_PATH), args.model, args.audit_model)
    if args.arm not in config.arms:
        raise SystemExit(f"Unknown arm '{args.arm}'; config defines {sorted(config.arms)}")
    arm = config.arms[args.arm]
    if arm.mode != MODE_SEQUENTIAL:
        raise SystemExit("State audit is defined only for sequential arms")
    if arm.name not in STATE_AUDIT_ARMS:
        raise SystemExit(
            f"State audit is restricted to {sorted(STATE_AUDIT_ARMS)}"
        )
    all_problems = load_problems()
    problems = select_problems(all_problems, args.problems, args.domain)
    state_references = load_state_audit_references()
    for problem in all_problems:
        reference = state_references.get(problem.problem_id)
        if reference is None:
            raise SystemExit(
                f"No matching reference solution for {problem.problem_id}"
            )
        reference_statement, _ = reference
        if reference_statement != problem.statement.strip():
            raise SystemExit(
                f"Problem statement mismatch in hard_solutions for "
                f"{problem.problem_id}"
            )
    seeds = select_seeds(arm, args.seeds)

    generated = [
        (problem, seed)
        for problem in problems
        for seed in seeds
        if seed_done(seed_output_dir(config, arm, problem.problem_id, seed))
    ]
    ungenerated = len(problems) * len(seeds) - len(generated)
    if ungenerated:
        log.warning("%d attempts have no generation output and are skipped", ungenerated)

    proof_audited: list[tuple[Problem, int]] = []
    for problem, seed in generated:
        output_dir = seed_output_dir(config, arm, problem.problem_id, seed)
        proof_path = output_dir / SEED_AUDIT_FILENAME
        if not proof_path.exists():
            continue
        proof_audited.append((problem, seed))

    missing_proof_audits = len(generated) - len(proof_audited)
    if missing_proof_audits:
        log.warning(
            "%d generated attempts lack proof audits and are skipped; rerun the "
            "public audit command",
            missing_proof_audits,
        )

    pending: list[tuple[Problem, int]] = []
    for problem, seed in proof_audited:
        output_dir = seed_output_dir(config, arm, problem.problem_id, seed)
        if not (output_dir / SEED_STATE_AUDIT_FILENAME).exists():
            pending.append((problem, seed))

    log.info(
        "Arm %s: %d attempts to state-audit, %d already current",
        arm.name,
        len(pending),
        len(proof_audited) - len(pending),
    )
    if pending:
        pool = TokenPool.from_env(token_env_name(config.audit_model))

        async def worker(problem: Problem, seed: int) -> None:
            await state_audit_seed(
                config,
                arm,
                problem,
                seed,
                pool,
                state_references[problem.problem_id][1],
            )

        tasks = [
            lambda p=problem, s=seed: worker(p, s) for problem, seed in pending
        ]
        await run_all(tasks, config.max_concurrency)

    state_path, state_count = compile_arm_state_audit(config, arm)
    log.info("Compiled %d state records -> %s", state_count, state_path)


if __name__ == "__main__":
    anyio.run(main)
