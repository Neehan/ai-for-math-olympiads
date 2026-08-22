"""Annotate observable route-progress states for every audited proof artifact.

Correctness remains owned by ``audit.json``.  This stage reuses those immutable
scores: passing checkpoints become S without another model call; missing
solution text remains unobserved; only nonempty score-below-5 artifacts receive
a reference-guided three-step outline annotation.  For each trajectory, the
harness compares the number of recognized steps with the preceding observed
checkpoint: an increase is P and a flat or decreasing incomplete count is U.
Complete 3/3 recognition remains P while proof execution continues, and S is
carried forward after the first passing proof.  The first artifact is compared
with zero recognized steps.  The final state is derived by code, not selected
by the annotator.
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
    MODE_PARALLEL,
    MODE_SEQUENTIAL,
    MODE_UNIFORM_STRATEGY,
    RESULTS_ROOT,
    SEED_AUDIT_FILENAME,
    SEED_STATE_AUDIT_FILENAME,
)
from src.models import ArmConfig, ExperimentConfig, Problem
from src.prompts import state_audit_prompt
from src.run import select_problems, select_seeds
from src.solver import token_env_name
from src.storage import (
    all_budget_cut_multipliers,
    bank_run_output_dir,
    budget_cut_multipliers,
    compile_arm_state_audit,
    cut_solution_path,
    load_problems,
    load_state_audit_references,
    materialize_budget_cut_snapshots,
    parallel_bank_done,
    seed_done,
    seed_output_dir,
    seed_solution_text,
    write_seed_state_audit,
)
from src.token_pool import TokenPool

log = logging.getLogger("state_audit")

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


def recognized_step_count(verdict: dict[str, object]) -> int:
    """Validate a step-presence verdict and return its recognized-step count."""
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
    return sum(bool(step["present"]) for step in steps)


def derive_state(
    verdict: dict[str, object], previous_recognized_steps: int
) -> tuple[str, int]:
    """Map change in recognized-step count mechanically to P or U."""
    if not 0 <= previous_recognized_steps <= 3:
        raise ValueError("Previous recognized-step count must be between 0 and 3")
    current_recognized_steps = recognized_step_count(verdict)
    state = (
        "P"
        if current_recognized_steps == 3
        or current_recognized_steps > previous_recognized_steps
        else "U"
    )
    return state, current_recognized_steps


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


def _carried_solved_checkpoint(text: str | None) -> dict[str, object]:
    record: dict[str, object] = {
        "state": "S",
        "steps": [],
        "note": "A valid proof was observed earlier; solved state is carried forward.",
    }
    if text is not None and text.strip():
        record["solution_sha256"] = _solution_sha256(text)
    return record


def _proof_artifacts(
    arm: ArmConfig,
    output_dir: Path,
    proof_audit: dict[str, Any],
    *,
    all_checkpoints: bool = False,
) -> list[tuple[str, str | None, int]]:
    """Read checkpoint texts and their already-completed correctness scores."""
    artifacts: list[tuple[str, str | None, int]] = []
    raw_cuts = proof_audit.get("budget_cuts", {})
    if not isinstance(raw_cuts, dict):
        raise ValueError("Proof audit has malformed budget_cuts")
    cut_multipliers = (
        (
            all_budget_cut_multipliers(arm.budget_units)
            if all_checkpoints
            else budget_cut_multipliers(arm.budget_units)
        )
        if arm.mode == MODE_SEQUENTIAL
        else []
    )
    for multiplier in cut_multipliers:
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
    *,
    output_dir_override: Path | None = None,
    record_extra: dict[str, object] | None = None,
    all_checkpoints: bool = False,
) -> None:
    """Annotate one audited proof and, for sequential arms, its checkpoints."""
    output_dir = output_dir_override or seed_output_dir(
        config, arm, problem.problem_id, seed
    )
    proof_path = output_dir / SEED_AUDIT_FILENAME
    proof_audit: dict[str, Any] = json.loads(proof_path.read_text(encoding="utf-8"))
    outline = _outline(problem)
    state_path = output_dir / SEED_STATE_AUDIT_FILENAME
    if all_checkpoints:
        if arm.mode != MODE_SEQUENTIAL:
            raise ValueError("--all-checkpoints is valid only for sequential arms")
        materialize_budget_cut_snapshots(
            config,
            arm,
            output_dir,
            all_budget_cut_multipliers(arm.budget_units),
        )
    artifacts = _proof_artifacts(
        arm, output_dir, proof_audit, all_checkpoints=all_checkpoints
    )
    expected_cut_labels = {
        label for label, _, _ in artifacts if label != f"{arm.budget_units}x"
    }
    existing_state: dict[str, Any] | None = None
    if state_path.exists():
        loaded_state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_state, dict):
            raise ValueError("Existing state audit is malformed")
        existing_state = loaded_state
        existing_cuts = loaded_state.get("budget_cuts", {})
        complete = isinstance(existing_cuts, dict) and expected_cut_labels.issubset(
            existing_cuts
        )
        if complete:
            return
    if state_path.exists() and not all_checkpoints:
        return
    checkpoint = AttemptCheckpoint(
        {
            "stage": "state_audit",
            "solver_model": config.model,
            "state_audit_model": config.audit_model,
            "arm": arm.name,
            "problem_id": problem.problem_id,
            "seed": seed,
            "output_dir": output_dir.relative_to(RESULTS_ROOT).as_posix(),
            "record_extra": record_extra or {},
            "all_checkpoints": all_checkpoints,
        }
    )
    try:
        records: dict[str, dict[str, object]] = {}
        verdict_cache: dict[str, dict[str, object]] = {}
        if existing_state is not None:
            old_records = dict(existing_state.get("budget_cuts", {}))
            old_records[f"{arm.budget_units}x"] = existing_state
            for old_record in old_records.values():
                if not isinstance(old_record, dict):
                    continue
                digest = old_record.get("solution_sha256")
                steps = old_record.get("steps")
                if (
                    isinstance(digest, str)
                    and isinstance(steps, list)
                    and len(steps) == 3
                ):
                    verdict_cache[digest] = {"steps": steps}
        previous_recognized_steps = 0
        solved_seen = False
        for label, text, proof_score in artifacts:
            if (text is None or not text.strip()) and proof_score >= 5:
                raise ValueError(
                    f"{problem.problem_id} {label}: passing proof audit has "
                    "no solution text"
                )
            if solved_seen:
                records[label] = _carried_solved_checkpoint(text)
                continue
            if text is None or not text.strip():
                records[label] = _missing_checkpoint()
                continue
            if proof_score >= 5:
                records[label] = _solved_checkpoint(text)
                solved_seen = True
                continue

            digest = _solution_sha256(text)
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
                verdict_cache[digest] = verdict
            else:
                verdict = cached

            state, current_recognized_steps = derive_state(
                verdict, previous_recognized_steps
            )
            raw_steps = verdict["steps"]
            if not isinstance(raw_steps, list):
                raise TypeError("State-audit steps are not a list")
            record: dict[str, object] = {
                "state": state,
                "steps": list(raw_steps),
                "note": (
                    "Recognized outline-step count increased from "
                    f"{previous_recognized_steps}/3 to "
                    f"{current_recognized_steps}/3."
                    if current_recognized_steps > previous_recognized_steps
                    else "All three outline steps remain recognized; "
                    "the trajectory is in proof execution."
                    if current_recognized_steps == 3
                    else "Recognized outline-step count did not increase "
                    f"({previous_recognized_steps}/3 to "
                    f"{current_recognized_steps}/3)."
                ),
                "solution_sha256": _solution_sha256(text),
            }
            records[label] = record
            previous_recognized_steps = current_recognized_steps

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
        if record_extra:
            final_record.update(record_extra)
        checkpoint.prepare_completion(
            state_path.relative_to(RESULTS_ROOT).as_posix()
        )
        write_seed_state_audit(output_dir, final_record)
        checkpoint.complete()
        log.info("%s/%s seed %d: state audit complete", arm.name, problem.problem_id, seed)
    finally:
        checkpoint.close()


def _bank_targets(
    arm: ArmConfig, bank_dir: Path, seed: int
) -> list[tuple[Path, dict[str, object]]]:
    """Resolve the executor proofs represented by one completed bank audit."""
    proof_audit = json.loads(
        (bank_dir / SEED_AUDIT_FILENAME).read_text(encoding="utf-8")
    )
    raw_runs = proof_audit.get("runs", [])
    if not isinstance(raw_runs, list):
        raise ValueError("Bank proof audit has malformed runs")
    targets: list[tuple[Path, dict[str, object]]] = []
    for raw in raw_runs:
        if not isinstance(raw, dict) or not isinstance(raw.get("run"), int):
            raise ValueError("Bank proof audit has malformed run entry")
        run = int(raw["run"])
        extra: dict[str, object]
        if arm.mode == MODE_PARALLEL:
            extra = {"parallel_bank_seed": seed, "parallel_run": run}
        elif arm.mode == MODE_UNIFORM_STRATEGY:
            strategy_index = raw.get("strategy_index")
            if not isinstance(strategy_index, int):
                raise ValueError("Uniform bank run is missing strategy_index")
            extra = {
                "uniform_strategy_bank_seed": seed,
                "uniform_strategy_run": run,
                "uniform_strategy_index": strategy_index,
            }
        else:
            raise ValueError(f"Unsupported bank mode: {arm.mode}")
        run_dir = bank_run_output_dir(bank_dir, run)
        if not (run_dir / SEED_AUDIT_FILENAME).exists():
            raise FileNotFoundError(f"Bank executor lacks proof audit: {run_dir}")
        targets.append((run_dir, extra))
    return targets


async def main() -> None:
    parser = argparse.ArgumentParser(description="State-audit one experiment arm.")
    parser.add_argument("--arm", required=True, help="Arm name")
    parser.add_argument("--problems", default=None, help="Comma-separated problem ids")
    parser.add_argument("--domain", default=None, help="Only problems in this domain")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed subset")
    parser.add_argument("--model", default=None, help="Solver model override")
    parser.add_argument("--audit-model", default=None, help="State annotator override")
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="For sequential arms, annotate every integer 1x,...,8x checkpoint",
    )
    args = parser.parse_args()

    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    config = override_models(load_config(CONFIG_PATH), args.model, args.audit_model)
    if args.arm not in config.arms:
        raise SystemExit(f"Unknown arm '{args.arm}'; config defines {sorted(config.arms)}")
    arm = config.arms[args.arm]
    if args.all_checkpoints and arm.mode != MODE_SEQUENTIAL:
        raise SystemExit("--all-checkpoints is valid only for sequential arms")
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
        if (
            parallel_bank_done(seed_output_dir(config, arm, problem.problem_id, seed))
            if arm.mode == MODE_PARALLEL
            else seed_done(seed_output_dir(config, arm, problem.problem_id, seed))
        )
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

    targets: list[tuple[Problem, int, Path, dict[str, object] | None]] = []
    for problem, seed in proof_audited:
        output_dir = seed_output_dir(config, arm, problem.problem_id, seed)
        if arm.mode in {MODE_PARALLEL, MODE_UNIFORM_STRATEGY}:
            targets.extend(
                (problem, seed, run_dir, extra)
                for run_dir, extra in _bank_targets(arm, output_dir, seed)
            )
        else:
            targets.append((problem, seed, output_dir, None))
    def state_done(output_dir: Path) -> bool:
        path = output_dir / SEED_STATE_AUDIT_FILENAME
        if not path.exists():
            return False
        if not args.all_checkpoints:
            return True
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        cuts = record.get("budget_cuts")
        return isinstance(cuts, dict) and all(
            f"{multiplier}x" in cuts
            for multiplier in all_budget_cut_multipliers(arm.budget_units)
        )

    pending = [target for target in targets if not state_done(target[2])]

    log.info(
        "Arm %s: %d proof artifacts to state-audit, %d already current",
        arm.name,
        len(pending),
        len(targets) - len(pending),
    )
    if pending:
        pool = TokenPool.from_env(token_env_name(config.audit_model))

        async def worker(
            problem: Problem,
            seed: int,
            output_dir: Path,
            extra: dict[str, object] | None,
        ) -> None:
            await state_audit_seed(
                config,
                arm,
                problem,
                seed,
                pool,
                state_references[problem.problem_id][1],
                output_dir_override=output_dir,
                record_extra=extra,
                all_checkpoints=args.all_checkpoints,
            )

        tasks = [
            lambda p=problem, s=seed, d=output_dir, e=extra: worker(p, s, d, e)
            for problem, seed, output_dir, extra in pending
        ]
        await run_all(tasks, config.max_concurrency)

    state_path, state_count = compile_arm_state_audit(config, arm)
    log.info("Compiled %d state records -> %s", state_count, state_path)
    failed = [
        (problem.problem_id, seed, output_dir.name)
        for problem, seed, output_dir, _ in pending
        if not state_done(output_dir)
    ]
    if failed:
        raise SystemExit(
            f"{len(failed)} state-audit task(s) failed; rerun the same command"
        )


if __name__ == "__main__":
    anyio.run(main)
