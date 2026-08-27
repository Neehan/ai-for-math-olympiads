"""Compression and controlled selection experiments over Uniform-C proposals."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from src.checkpoint import AttemptCheckpoint
from src.constants import (
    COMPRESSED_STRATEGIES_FILENAME,
    DEFAULT_UNIFORM_COMPRESS_MODEL,
    RESULTS_ROOT,
    SELECTION_FILENAME,
    UNIFORM_COMPRESS_EXAMPLE_IDS,
    UNIFORM_COMPRESS_SAMPLE_SEED,
    UNIFORM_STRATEGIES_FILENAME,
)
from src.models import ArmConfig, ExperimentConfig, Problem
from src.prompts import (
    selection_no_problem_prompt,
    selection_prompt,
    uniform_compress_prompt,
)
from src.storage import seed_output_dir, write_auxiliary_result
from src.token_pool import TokenPool

COMPRESS_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"strategy": {"type": "string", "minLength": 1, "maxLength": 400}},
    "required": ["strategy"],
    "additionalProperties": False,
}

SELECTION_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "ranking": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1, "maximum": 4},
            "minItems": 4,
            "maxItems": 4,
            "uniqueItems": True,
        },
        "reason": {"type": "string", "minLength": 1, "maxLength": 1200},
    },
    "required": ["ranking", "reason"],
    "additionalProperties": False,
}


def _stable_rng(*parts: object) -> random.Random:
    material = "\0".join(str(part) for part in parts).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def sampled_strategy_indices(source_model: str, problem_id: str, count: int) -> list[int]:
    """Choose three raw proposal indices reproducibly, without outcome filtering."""
    if count < 3:
        raise ValueError(
            f"{problem_id}: compression requires at least three generated strategies; "
            f"found {count}"
        )
    indices = list(range(count))
    _stable_rng(UNIFORM_COMPRESS_SAMPLE_SEED, source_model, problem_id).shuffle(indices)
    return indices[:3]


def compression_examples(
    all_problems: list[Problem], target_problem_id: str
) -> list[tuple[str, str]]:
    """Select five fixed oracle examples, replacing a target collision by #6."""
    by_id = {problem.problem_id: problem for problem in all_problems}
    missing = [pid for pid in UNIFORM_COMPRESS_EXAMPLE_IDS if pid not in by_id]
    if missing:
        raise ValueError(
            "Compression example pool is unavailable for this dataset: " + ", ".join(missing)
        )
    chosen = [pid for pid in UNIFORM_COMPRESS_EXAMPLE_IDS if pid != target_problem_id][:5]
    if len(chosen) != 5:
        raise AssertionError("Frozen six-example pool could not supply five examples")
    examples: list[tuple[str, str]] = []
    for problem_id in chosen:
        problem = by_id[problem_id]
        if problem.hint_h2 is None:
            raise ValueError(f"Compression example {problem_id} has no oracle sketch")
        examples.append((problem.statement, problem.hint_h2))
    return examples


def _source_strategy_path(
    config: ExperimentConfig, problem_id: str, seed: int
) -> Path:
    return (
        RESULTS_ROOT
        / config.model_dirname
        / "baseline-uniform-strategy-only"
        / problem_id
        / f"seed_{seed}"
        / UNIFORM_STRATEGIES_FILENAME
    )


def _load_raw_strategies(
    config: ExperimentConfig, problem_id: str, seed: int
) -> list[str]:
    path = _source_strategy_path(config, problem_id, seed)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing planner-only bank {path}; run baseline-uniform-strategy-only first"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    strategies = raw.get("strategies")
    if not isinstance(strategies, list) or not all(
        isinstance(strategy, str) and strategy.strip() for strategy in strategies
    ):
        raise ValueError(f"Malformed strategy bank: {path}")
    return [strategy.strip() for strategy in strategies]


def _worker_config(config: ExperimentConfig, worker_model: str) -> ExperimentConfig:
    return dataclasses.replace(config, audit_model=worker_model)


async def compress_uniform_strategies(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    pool: TokenPool,
    all_problems: list[Problem],
    worker_model: str = DEFAULT_UNIFORM_COMPRESS_MODEL,
) -> None:
    """Sample and independently compress three raw proposals into <=25 words."""
    from src.audit import _judge

    if problem.hint_h2 is None or len(problem.hint_h2.split()) > 25:
        raise ValueError(
            f"{problem.problem_id}: compression requires a frozen <=25-word oracle sketch"
        )
    raw_strategies = _load_raw_strategies(config, problem.problem_id, seed)
    indices = sampled_strategy_indices(config.model, problem.problem_id, len(raw_strategies))
    examples = compression_examples(all_problems, problem.problem_id)
    output_dir = seed_output_dir(config, arm, problem.problem_id, seed)
    checkpoint = AttemptCheckpoint(
        {
            "stage": "uniform_compress",
            "source_model": config.model,
            "worker_model": worker_model,
            "problem_id": problem.problem_id,
            "problem_statement": problem.statement,
            "oracle_strategy": problem.hint_h2,
            "seed": seed,
            "sample_seed": UNIFORM_COMPRESS_SAMPLE_SEED,
            "sampled_indices": indices,
            "raw_strategies": [raw_strategies[index] for index in indices],
            "examples": examples,
            "example_ids": [
                pid
                for pid in UNIFORM_COMPRESS_EXAMPLE_IDS
                if pid != problem.problem_id
            ][:5],
        }
    )
    try:
        worker_config = _worker_config(config, worker_model)
        compressed: list[dict[str, object]] = []
        for candidate_number, raw_index in enumerate(indices, start=1):
            role = f"candidate_{candidate_number}"
            verdict, _ = await _judge(
                worker_config,
                uniform_compress_prompt(problem, raw_strategies[raw_index], examples),
                pool,
                str(checkpoint.scratch_dir(role)),
                checkpoint,
                role,
                output_schema=COMPRESS_OUTPUT_SCHEMA,
                allow_tools=False,
            )
            strategy = str(verdict["strategy"]).strip()
            word_count = len(strategy.split())
            if not strategy:
                raise ValueError(
                    f"{problem.problem_id} candidate {candidate_number}: compressor "
                    "returned an empty strategy"
                )
            if word_count > 25:
                raise ValueError(
                    f"{problem.problem_id} candidate {candidate_number}: compressor "
                    f"returned {word_count} words (maximum 25)"
                )
            compressed.append(
                {
                    "candidate_id": f"generated_{candidate_number}",
                    "raw_strategy_index": raw_index + 1,
                    "raw_strategy": raw_strategies[raw_index],
                    "strategy": strategy,
                    "word_count": word_count,
                }
            )
        artifact: dict[str, object] = {
            "problem_id": problem.problem_id,
            "source_model": config.model,
            "oracle_strategy": problem.hint_h2,
            "generated_strategies": compressed,
        }
        meta: dict[str, object] = {
            "problem_id": problem.problem_id,
            "arm": arm.name,
            "mode": arm.mode,
            "model": config.model,
            "worker_model": worker_model,
            "seed": seed,
            "sample_seed": UNIFORM_COMPRESS_SAMPLE_SEED,
            "sampled_raw_strategy_indices": [index + 1 for index in indices],
            "strategy_count": 3,
            "gradeable_solution_emitted": False,
        }
        checkpoint.prepare_completion(
            (output_dir / "meta.json").relative_to(RESULTS_ROOT).as_posix()
        )
        write_auxiliary_result(
            output_dir, COMPRESSED_STRATEGIES_FILENAME, artifact, meta
        )
        checkpoint.complete()
    finally:
        checkpoint.close()


def _selection_candidates(
    problem: Problem, record: dict[str, Any], seed: int, source_model: str
) -> tuple[list[dict[str, object]], int]:
    oracle = record.get("oracle_strategy")
    if problem.hint_h2 is None or oracle != problem.hint_h2.strip():
        raise ValueError(
            f"{problem.problem_id}: frozen selection oracle does not match hard hint"
        )
    generated = record.get("generated_strategies")
    if not isinstance(generated, list) or len(generated) != 3:
        raise ValueError(f"{problem.problem_id}: selection needs exactly 3 proposals")
    candidates: list[dict[str, object]] = [
        {
            "candidate_id": "oracle",
            "strategy": oracle,
            "strategy_acquired": True,
            "acquisition_basis": "oracle",
            "provenance": "oracle",
        }
    ]
    for item in generated:
        if not isinstance(item, dict):
            raise TypeError("Malformed generated selection candidate")
        if not isinstance(item.get("strategy_acquired"), bool):
            raise TypeError("Selection candidate lacks strategy_acquired label")
        acquisition_basis = item.get("acquisition_basis")
        if acquisition_basis not in {
            "reference_steps",
            "human_alternative",
            "none",
        }:
            raise TypeError("Selection candidate lacks valid acquisition_basis")
        strategy_acquired = item["strategy_acquired"]
        if strategy_acquired != (acquisition_basis != "none"):
            raise ValueError(
                "Selection candidate has inconsistent strategy_acquired and "
                "acquisition_basis"
            )
        adjudication_note = item.get("adjudication_note")
        if acquisition_basis == "human_alternative":
            if not isinstance(adjudication_note, str) or not adjudication_note.strip():
                raise ValueError(
                    "Human-alternative selection candidate needs adjudication_note"
                )
        elif adjudication_note is not None:
            raise ValueError(
                "Only human-alternative candidates may have adjudication_note"
            )
        candidates.append(
            {
                "candidate_id": str(item["candidate_id"]),
                "strategy": str(item["strategy"]),
                "strategy_acquired": strategy_acquired,
                "acquisition_basis": acquisition_basis,
                **(
                    {"adjudication_note": adjudication_note.strip()}
                    if isinstance(adjudication_note, str)
                    else {}
                ),
                "provenance": "generated",
            }
        )
    _stable_rng("selection-order-v1", source_model, problem.problem_id, seed).shuffle(
        candidates
    )
    oracle_position = next(
        index for index, candidate in enumerate(candidates, start=1)
        if candidate["candidate_id"] == "oracle"
    )
    return candidates, oracle_position


async def run_selection(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    pool: TokenPool,
    frozen_record: dict[str, Any],
    worker_model: str,
    *,
    include_problem: bool,
) -> None:
    """Rank a frozen randomized four-strategy set and score it deterministically."""
    from src.audit import _judge

    candidates, oracle_position = _selection_candidates(
        problem, frozen_record, seed, config.model
    )
    selection_output_tokens = config.selection_budget_tokens(arm)
    candidate_texts = [str(candidate["strategy"]) for candidate in candidates]
    prompt = (
        selection_prompt(problem, candidate_texts)
        if include_problem
        else selection_no_problem_prompt(candidate_texts)
    )
    output_dir = seed_output_dir(config, arm, problem.problem_id, seed)
    checkpoint = AttemptCheckpoint(
        {
            "stage": arm.mode,
            "source_model": config.model,
            "worker_model": worker_model,
            "problem_id": problem.problem_id,
            **(
                {"problem_statement": problem.statement}
                if include_problem
                else {}
            ),
            "seed": seed,
            "include_problem": include_problem,
            "budget_output_tokens": selection_output_tokens,
            "candidate_order": candidates,
        }
    )
    try:
        verdict, _ = await _judge(
            _worker_config(config, worker_model),
            prompt,
            pool,
            str(checkpoint.scratch_dir("selection")),
            checkpoint,
            "selection",
            output_schema=SELECTION_OUTPUT_SCHEMA,
            allow_tools=False,
            max_output_tokens_per_response=selection_output_tokens,
            max_turns=1,
            terminal_no_verdict=True,
        )
        ranking_value = verdict.get("ranking")
        valid_ranking = (
            isinstance(ranking_value, list)
            and all(type(position) is int for position in ranking_value)
            and sorted(ranking_value) == [1, 2, 3, 4]
        )
        decision_status = (
            "selected"
            if valid_ranking
            else str(verdict.get("decision_status", "no_decision"))
        )
        ranking: list[int] = (
            [int(position) for position in ranking_value]
            if valid_ranking and isinstance(ranking_value, list)
            else []
        )
        ranked_candidates = [candidates[int(position) - 1] for position in ranking]
        oracle_rank = ranking.index(oracle_position) + 1 if ranking else None
        top = ranked_candidates[0] if ranked_candidates else None
        strategy_acquired_count = sum(
            bool(candidate["strategy_acquired"]) for candidate in candidates
        )
        usage_value = verdict.get("usage")
        usage: dict[str, Any] = (
            dict(usage_value) if isinstance(usage_value, dict) else {}
        )
        output_tokens_spent = (
            int(usage.get("output_tokens", 0))
            if type(usage.get("output_tokens", 0)) is int
            else 0
        )
        artifact: dict[str, object] = {
            "problem_id": problem.problem_id,
            "source_model": config.model,
            "worker_model": worker_model,
            "seed": seed,
            "include_problem": include_problem,
            "decision_status": decision_status,
            "candidates": candidates,
            "ranking_positions": ranking,
            "ranked_candidate_ids": [
                candidate["candidate_id"] for candidate in ranked_candidates
            ],
            "reason": str(
                verdict.get(
                    "reason",
                    f"No valid ranking returned ({verdict.get('stop_reason', 'unknown')}).",
                )
            ),
            "stop_reason": verdict.get("stop_reason"),
        }
        audit: dict[str, object] = {
            "problem_id": problem.problem_id,
            "arm": arm.name,
            "seed": seed,
            "source_model": config.model,
            "worker_model": worker_model,
            "decision_status": decision_status,
            "oracle_rank": oracle_rank,
            "oracle_top1": oracle_rank == 1,
            "top_candidate_id": top["candidate_id"] if top is not None else None,
            "strategy_acquired_candidate_count": strategy_acquired_count,
            "random_strategy_acquired_top1_probability": (
                strategy_acquired_count / len(candidates)
            ),
        }
        if include_problem:
            audit.update(
                {
                    "strategy_acquired_top1": bool(
                        top is not None and top["strategy_acquired"]
                    ),
                }
            )
        else:
            audit["style_leakage_oracle_top1"] = oracle_rank == 1
        meta: dict[str, object] = {
            "problem_id": problem.problem_id,
            "arm": arm.name,
            "mode": arm.mode,
            "model": config.model,
            "worker_model": worker_model,
            "seed": seed,
            "candidate_count": 4,
            "budget_output_tokens": selection_output_tokens,
            "output_tokens_spent": output_tokens_spent,
            "oracle_position": oracle_position,
            "decision_status": decision_status,
            "gradeable_solution_emitted": False,
        }
        checkpoint.prepare_completion(
            (output_dir / "meta.json").relative_to(RESULTS_ROOT).as_posix()
        )
        write_auxiliary_result(
            output_dir, SELECTION_FILENAME, artifact, meta, audit=audit
        )
        checkpoint.complete()
    finally:
        checkpoint.close()
