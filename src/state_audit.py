"""Annotate observable U/P/S route states for sequential proof checkpoints.

Correctness remains owned by ``audit.json``.  This stage reuses those immutable
scores: passing checkpoints become S without another model call; missing
solution text remains unobserved; only nonempty score-below-5 artifacts receive
a reference-guided three-step outline annotation.  The final state is
derived by code, not selected by the annotator.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import anyio

from src.audit import _judge
from src.checkpoint import AttemptCheckpoint, protocol_fingerprint
from src.concurrency import run_all
from src.config import load_config, override_models
from src.constants import (
    CONFIG_PATH,
    LOG_FORMAT,
    LOG_LEVEL,
    MODE_SEQUENTIAL,
    STATE_RESULTS_ROOT,
    SEED_AUDIT_FILENAME,
    SEED_STATE_AUDIT_FILENAME,
    STATE_AUDIT_PROMPT_FILE,
)
from src.models import ArmConfig, ExperimentConfig, Problem
from src.prompts import state_audit_prompt
from src.run import select_problems, select_seeds
from src.solver import agent_runtime_policy, agent_settings_path, token_env_name, uses_vllm
from src.storage import (
    budget_cut_multipliers,
    compile_arm_state_audit,
    cut_solution_path,
    load_problems,
    load_state_audit_references,
    seed_done,
    seed_output_dir,
    state_output_dir,
    seed_solution_text,
    write_seed_state_audit,
)
from src.token_pool import TokenPool

log = logging.getLogger("state_audit")

STATE_AUDIT_PROTOCOL = "frozen_three_step_outline_recognition_v1"
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


def _state_protocol_fingerprint(config: ExperimentConfig) -> str:
    return protocol_fingerprint(
        agent_settings_path(config.audit_model), (STATE_AUDIT_PROMPT_FILE,)
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


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


def _missing_checkpoint(proof_score: int) -> dict[str, object]:
    return {
        "state": None,
        "steps": [],
        "proof_score": proof_score,
        "note": "No complete solution text was emitted at this budget; state is unobserved.",
    }


def _solved_checkpoint(proof_score: int) -> dict[str, object]:
    return {
        "state": "S",
        "steps": [],
        "proof_score": proof_score,
        "note": f"Correctness audit score {proof_score}; state assigned mechanically as solved.",
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


def _current_state_record(
    path: Path,
    *,
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    outline_sha256: str,
    reference_solution_sha256: str,
    proof_audit_sha256: str,
) -> bool:
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    expected = {
        "problem_id": problem.problem_id,
        "arm": arm.name,
        "seed": seed,
        "solver_model": config.model,
        "state_audit_model": config.audit_model,
        "state_audit_protocol": STATE_AUDIT_PROTOCOL,
        "outline_sha256": outline_sha256,
        "reference_solution_sha256": reference_solution_sha256,
        "proof_audit_sha256": proof_audit_sha256,
        "protocol_fingerprint": _state_protocol_fingerprint(config),
    }
    return all(record.get(key) == value for key, value in expected.items())


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
    state_dir = state_output_dir(config, arm, problem.problem_id, seed)
    proof_path = output_dir / SEED_AUDIT_FILENAME
    proof_bytes = proof_path.read_bytes()
    proof_audit: dict[str, Any] = json.loads(proof_bytes)
    outline = _outline(problem)
    outline_digest = _sha256_text(outline)
    reference_digest = _sha256_text(reference_solution)
    proof_digest = _sha256_bytes(proof_bytes)
    state_path = state_dir / SEED_STATE_AUDIT_FILENAME
    artifacts = _proof_artifacts(arm, output_dir, proof_audit)
    if _current_state_record(
        state_path,
        config=config,
        arm=arm,
        problem=problem,
        seed=seed,
        outline_sha256=outline_digest,
        reference_solution_sha256=reference_digest,
        proof_audit_sha256=proof_digest,
    ):
        return
    checkpoint = AttemptCheckpoint(
        {
            "stage": "state_audit",
            "solver_model": config.model,
            "state_audit_model": config.audit_model,
            "effort": config.effort,
            "arm": dataclasses.asdict(arm),
            "problem_id": problem.problem_id,
            "problem_statement": problem.statement,
            "seed": seed,
            "outline": outline,
            "outline_sha256": outline_digest,
            "reference_solution_sha256": reference_digest,
            "proof_audit_sha256": proof_digest,
            "state_audit_protocol": STATE_AUDIT_PROTOCOL,
            "audit_max_turns": config.audit_max_turns,
            "protocol_fingerprint": _state_protocol_fingerprint(config),
            **(
                {"agent_runtime_policy": agent_runtime_policy(config.audit_model)}
                if uses_vllm(config.audit_model)
                else {}
            ),
        }
    )
    try:
        records: dict[str, dict[str, object]] = {}
        verdict_cache: dict[str, tuple[dict[str, object], str]] = {}
        for label, text, proof_score in artifacts:
            if text is None or not text.strip():
                if proof_score >= 5:
                    raise ValueError(
                        f"{problem.problem_id} {label}: passing proof audit has "
                        "no solution text"
                    )
                records[label] = _missing_checkpoint(proof_score)
                continue
            digest = _sha256_text(text)
            if proof_score >= 5:
                records[label] = _solved_checkpoint(proof_score)
                continue

            cached = verdict_cache.get(digest)
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
                verdict_cache[digest] = (verdict, label)
                reused_from: str | None = None
            else:
                verdict, reused_from = cached

            state = derive_state(verdict)
            raw_steps = verdict["steps"]
            if not isinstance(raw_steps, list):
                raise TypeError("State-audit steps are not a list")
            record: dict[str, object] = {
                "state": state,
                "steps": list(raw_steps),
                "proof_score": proof_score,
                "note": (
                    "All three frozen outline steps are explicitly recognized."
                    if state == "P"
                    else "At least one frozen outline step is not recognized."
                ),
            }
            if reused_from is not None:
                record["state_audit_reused_from"] = reused_from
            records[label] = record

        fingerprint = _state_protocol_fingerprint(config)
        final_record: dict[str, object] = {
            "problem_id": problem.problem_id,
            "arm": arm.name,
            "seed": seed,
            "solver_model": config.model,
            "proof_audit_model": proof_audit.get("audit_model"),
            "state_audit_model": config.audit_model,
            "state_audit_protocol": STATE_AUDIT_PROTOCOL,
            "outline_sha256": outline_digest,
            "reference_solution_sha256": reference_digest,
            "proof_audit_sha256": proof_digest,
            "protocol_fingerprint": fingerprint,
            "checkpoints": records,
        }
        checkpoint.prepare_completion(
            state_path.relative_to(STATE_RESULTS_ROOT).as_posix()
        )
        write_seed_state_audit(state_dir, final_record)
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
        proof_path = output_dir / SEED_AUDIT_FILENAME
        outline_digest = _sha256_text(_outline(problem))
        reference_solution = state_references[problem.problem_id][1]
        if not _current_state_record(
            state_output_dir(config, arm, problem.problem_id, seed)
            / SEED_STATE_AUDIT_FILENAME,
            config=config,
            arm=arm,
            problem=problem,
            seed=seed,
            outline_sha256=outline_digest,
            reference_solution_sha256=_sha256_text(reference_solution),
            proof_audit_sha256=_sha256_bytes(proof_path.read_bytes()),
        ):
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
