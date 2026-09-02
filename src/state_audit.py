"""Annotate observable route-progress states for every audited proof artifact.

Correctness remains owned by ``audit.json``.  For proof trajectories, this stage
reuses those immutable scores: a passing checkpoint or complete 3/3 strategy recognition enters S;
missing solution text remains unobserved; only nonempty score-below-5 artifacts
receive a reference-guided three-step outline annotation.  Before acquisition,
an increased but incomplete recognized-step count is P and a flat or decreasing
incomplete count is U.  S is carried forward after first complete strategy
acquisition.  The first artifact is compared with zero recognized steps.  The
final state is derived by code, not selected by the annotator. Raw planner
proposals use a separate binary frozen-oracle-strategy match audit.
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
from src.config import load_config, override_max_concurrency, override_models
from src.constants import (
    CONFIG_PATH,
    LOG_FORMAT,
    LOG_LEVEL,
    MODE_PARALLEL,
    MODE_SEQUENTIAL,
    MODE_UNIFORM_COMPRESS,
    MODE_UNIFORM_STRATEGY,
    MODE_UNIFORM_STRATEGY_ONLY,
    RESULTS_ROOT,
    SEED_AUDIT_FILENAME,
    SEED_STATE_AUDIT_FILENAME,
    UNIFORM_STRATEGIES_FILENAME,
)
from src.models import ArmConfig, ExperimentConfig, Problem
from src.prompts import state_audit_prompt, strategy_state_audit_prompt
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
    uniform_strategy_bank_done,
    uniform_strategy_only_done,
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

STRATEGY_MATCH_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "oracle_strategy_match": {"type": "boolean"},
        "reason": {"type": "string", "minLength": 1, "maxLength": 600},
    },
    "required": ["oracle_strategy_match", "reason"],
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
    """Map recognized-step count mechanically to U, P, or acquired S."""
    if not 0 <= previous_recognized_steps <= 3:
        raise ValueError("Previous recognized-step count must be between 0 and 3")
    current_recognized_steps = recognized_step_count(verdict)
    state = (
        "S"
        if current_recognized_steps == 3
        else "P"
        if current_recognized_steps > previous_recognized_steps
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


def _proof_acquired_checkpoint(text: str, audit_model: str) -> dict[str, object]:
    return {
        "state": "S",
        "steps": [],
        "note": (
            "Correctness audit passed; complete strategy acquisition assigned "
            "mechanically."
        ),
        "solution_sha256": _solution_sha256(text),
        "audit_model": audit_model,
    }


def _carried_acquired_checkpoint(
    text: str | None, audit_model: str
) -> dict[str, object]:
    record: dict[str, object] = {
        "state": "S",
        "steps": [],
        "note": (
            "A complete strategy was observed earlier; acquired state is carried "
            "forward."
        ),
        "audit_model": audit_model,
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
) -> list[tuple[str, str | None, int, str]]:
    """Read checkpoint texts and their already-completed correctness scores."""
    artifacts: list[tuple[str, str | None, int, str]] = []
    full_audit_model = proof_audit.get("audit_model")
    if not isinstance(full_audit_model, str):
        raise ValueError("Proof audit has no judge-model provenance")
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
        cut_audit_model = cut.get("audit_model", full_audit_model)
        if not isinstance(cut_audit_model, str):
            raise ValueError(f"Proof audit {label} has malformed judge provenance")
        artifacts.append((label, text, score, cut_audit_model))
    full_text = seed_solution_text(output_dir)
    full_score = proof_audit.get("audit_score")
    if not isinstance(full_score, int):
        raise ValueError("Proof audit is missing its full score")
    full_label = f"{arm.budget_units}x"
    artifacts.append((full_label, full_text, full_score, full_audit_model))
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
        label for label, _, _, _ in artifacts if label != f"{arm.budget_units}x"
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
            "problem_statement": problem.statement,
            "outline": outline,
            "reference_solution": reference_solution,
            "seed": seed,
            "output_dir": output_dir.relative_to(RESULTS_ROOT).as_posix(),
            "record_extra": record_extra or {},
            "all_checkpoints": all_checkpoints,
        }
    )
    try:
        records: dict[str, dict[str, object]] = {}
        verdict_cache: dict[str, tuple[dict[str, object], str]] = {}
        if existing_state is not None:
            existing_model = existing_state.get("audit_model")
            if not isinstance(existing_model, str):
                raise ValueError("Existing state audit has no annotator provenance")
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
                    record_model = old_record.get("audit_model", existing_model)
                    if not isinstance(record_model, str):
                        raise ValueError(
                            "Existing state checkpoint has malformed annotator provenance"
                        )
                    verdict_cache[digest] = ({"steps": steps}, record_model)
        previous_recognized_steps = 0
        acquired_seen = False
        acquired_audit_model: str | None = None
        for label, text, proof_score, proof_audit_model in artifacts:
            if (text is None or not text.strip()) and proof_score >= 5:
                raise ValueError(
                    f"{problem.problem_id} {label}: passing proof audit has "
                    "no solution text"
                )
            if acquired_seen:
                assert acquired_audit_model is not None
                records[label] = _carried_acquired_checkpoint(
                    text, acquired_audit_model
                )
                continue
            if text is None or not text.strip():
                records[label] = _missing_checkpoint()
                continue
            if proof_score >= 5:
                records[label] = _proof_acquired_checkpoint(text, proof_audit_model)
                acquired_seen = True
                acquired_audit_model = proof_audit_model
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
                audit_model = config.audit_model
                verdict_cache[digest] = (verdict, audit_model)
            else:
                verdict, audit_model = cached

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
                    "All three outline steps are recognized; complete strategy "
                    "acquired."
                    if state == "S"
                    else "Recognized outline-step count increased from "
                    f"{previous_recognized_steps}/3 to "
                    f"{current_recognized_steps}/3."
                    if current_recognized_steps > previous_recognized_steps
                    else "Recognized outline-step count did not increase "
                    f"({previous_recognized_steps}/3 to "
                    f"{current_recognized_steps}/3)."
                ),
                "solution_sha256": _solution_sha256(text),
                "audit_model": audit_model,
            }
            records[label] = record
            previous_recognized_steps = current_recognized_steps
            if state == "S":
                acquired_seen = True
                acquired_audit_model = audit_model

        final_label = f"{arm.budget_units}x"
        final_checkpoint = records.pop(final_label)
        final_record: dict[str, object] = {
            "problem_id": problem.problem_id,
            "arm": arm.name,
            "seed": seed,
            "solver_model": config.model,
            "audit_model": final_checkpoint.get("audit_model", config.audit_model),
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


async def state_audit_strategy_artifact(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    pool: TokenPool,
    reference_solution: str,
) -> None:
    """Annotate each raw planner proposal for strategy acquisition."""
    output_dir = seed_output_dir(config, arm, problem.problem_id, seed)
    state_path = output_dir / SEED_STATE_AUDIT_FILENAME
    if state_path.exists():
        return
    if arm.mode != MODE_UNIFORM_STRATEGY_ONLY:
        raise ValueError(f"Unsupported strategy-artifact mode: {arm.mode}")
    strategy_artifact = json.loads(
        (output_dir / UNIFORM_STRATEGIES_FILENAME).read_text(encoding="utf-8")
    )
    raw_strategies = strategy_artifact.get("strategies")
    if not isinstance(raw_strategies, list) or not all(
        isinstance(strategy, str) and strategy.strip()
        for strategy in raw_strategies
    ):
        raise ValueError(f"Malformed planner-only strategy bank: {output_dir}")
    entries = [
        {
            "candidate_id": f"strategy_{index}",
            "strategy": str(strategy).strip(),
        }
        for index, strategy in enumerate(raw_strategies, start=1)
    ]
    strategies = [str(entry["strategy"]) for entry in entries]
    oracle_strategy = (problem.hint_h2 or "").strip()
    if not oracle_strategy:
        raise ValueError(f"{problem.problem_id}: no frozen oracle strategy")
    checkpoint = AttemptCheckpoint(
        {
            "stage": "oracle_strategy_match_audit_v1",
            "solver_model": config.model,
            "state_audit_model": config.audit_model,
            "arm": arm.name,
            "problem_id": problem.problem_id,
            "problem_statement": problem.statement,
            "oracle_strategy": oracle_strategy,
            "reference_solution": reference_solution,
            "seed": seed,
            "strategies": strategies,
        }
    )
    try:
        records: list[dict[str, object]] = []
        verdict_cache: dict[str, dict[str, object]] = {}
        for index, entry in enumerate(entries, start=1):
            strategy = str(entry["strategy"])
            digest = _solution_sha256(strategy)
            verdict = verdict_cache.get(digest)
            if verdict is None:
                role = f"strategy_{index}"
                verdict, _ = await _judge(
                    config,
                    strategy_state_audit_prompt(
                        problem, oracle_strategy, reference_solution, strategy
                    ),
                    pool,
                    str(checkpoint.scratch_dir(role)),
                    checkpoint,
                    role,
                    output_schema=STRATEGY_MATCH_OUTPUT_SCHEMA,
                    allow_tools=False,
                )
                verdict_cache[digest] = verdict
            match = verdict.get("oracle_strategy_match")
            reason = verdict.get("reason")
            if (
                not isinstance(match, bool)
                or not isinstance(reason, str)
                or not reason.strip()
                or any(
                    ord(character) < 32 and character not in "\n\t\r"
                    for character in reason
                )
            ):
                raise ValueError("Malformed oracle-strategy-match verdict")
            record: dict[str, object] = {
                "strategy_index": index,
                "candidate_id": str(entry["candidate_id"]),
                "oracle_strategy_match": match,
                "reason": reason.strip(),
                "strategy_sha256": digest,
                "audit_model": config.audit_model,
            }
            records.append(record)
        acquired_count = sum(
            bool(record["oracle_strategy_match"]) for record in records
        )
        final_record: dict[str, object] = {
            "problem_id": problem.problem_id,
            "arm": arm.name,
            "seed": seed,
            "solver_model": config.model,
            "audit_model": config.audit_model,
            "state": "S" if acquired_count else "U",
            "steps": [],
            "note": (
                f"{acquired_count}/{len(records)} candidate strategies match the "
                "frozen oracle strategy."
            ),
            "strategies": records,
            "budget_cuts": {},
        }
        checkpoint.prepare_completion(
            state_path.relative_to(RESULTS_ROOT).as_posix()
        )
        write_seed_state_audit(output_dir, final_record)
        checkpoint.complete()
        log.info(
            "%s/%s seed %d: %d/%d strategies acquired",
            arm.name,
            problem.problem_id,
            seed,
            acquired_count,
            len(records),
        )
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
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Operational session-concurrency override (default: config.json)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    config = override_models(load_config(CONFIG_PATH), args.model, args.audit_model)
    config = override_max_concurrency(config, args.max_concurrency)
    if args.arm not in config.arms:
        raise SystemExit(f"Unknown arm '{args.arm}'; config defines {sorted(config.arms)}")
    arm = config.arms[args.arm]
    if arm.mode == MODE_UNIFORM_COMPRESS:
        raise SystemExit(
            "baseline-uniform-compress has no state-audit stage; audit the "
            "matching baseline-uniform-strategy-only raw proposals instead"
        )
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

    if arm.mode == MODE_UNIFORM_STRATEGY_ONLY:
        def strategy_artifact_done(problem: Problem, seed: int) -> bool:
            output_dir = seed_output_dir(config, arm, problem.problem_id, seed)
            return uniform_strategy_only_done(output_dir)

        generated_strategy_artifacts = [
            (problem, seed)
            for problem in problems
            for seed in seeds
            if strategy_artifact_done(problem, seed)
        ]
        missing = len(problems) * len(seeds) - len(generated_strategy_artifacts)
        if missing:
            log.warning("%d strategy artifacts are missing and are skipped", missing)
        pending_strategy_artifacts = [
            (problem, seed)
            for problem, seed in generated_strategy_artifacts
            if not (
                seed_output_dir(config, arm, problem.problem_id, seed)
                / SEED_STATE_AUDIT_FILENAME
            ).exists()
        ]
        log.info(
            "Arm %s: %d strategy artifacts to state-audit, %d already current",
            arm.name,
            len(pending_strategy_artifacts),
            len(generated_strategy_artifacts) - len(pending_strategy_artifacts),
        )
        if pending_strategy_artifacts:
            pool = TokenPool.from_env(token_env_name(config.audit_model))
            tasks = [
                lambda p=problem, s=seed: state_audit_strategy_artifact(
                    config,
                    arm,
                    p,
                    s,
                    pool,
                    state_references[p.problem_id][1],
                )
                for problem, seed in pending_strategy_artifacts
            ]
            await run_all(tasks, config.max_concurrency)
        state_path, state_count = compile_arm_state_audit(config, arm)
        log.info("Compiled %d state records -> %s", state_count, state_path)
        failed = [
            (problem.problem_id, seed)
            for problem, seed in pending_strategy_artifacts
            if not (
                seed_output_dir(config, arm, problem.problem_id, seed)
                / SEED_STATE_AUDIT_FILENAME
            ).exists()
        ]
        if failed:
            raise SystemExit(
                f"{len(failed)} strategy-state audit task(s) failed; rerun"
            )
        return

    generated = [
        (problem, seed)
        for problem in problems
        for seed in seeds
        if (
            parallel_bank_done(seed_output_dir(config, arm, problem.problem_id, seed))
            if arm.mode == MODE_PARALLEL
            else uniform_strategy_bank_done(
                seed_output_dir(config, arm, problem.problem_id, seed)
            )
            if arm.mode == MODE_UNIFORM_STRATEGY
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
