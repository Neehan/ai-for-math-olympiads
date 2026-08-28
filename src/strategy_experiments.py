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
    UNIFORM_COMPRESS_EXAMPLE_IDS,
    UNIFORM_COMPRESS_SAMPLE_SEED,
    UNIFORM_STRATEGIES_FILENAME,
)
from src.models import ArmConfig, ExperimentConfig, Problem
from src.prompts import uniform_compress_prompt
from src.storage import seed_output_dir, write_auxiliary_result
from src.token_pool import TokenPool

COMPRESS_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"strategy": {"type": "string", "minLength": 1, "maxLength": 400}},
    "required": ["strategy"],
    "additionalProperties": False,
}


def _stable_rng(*parts: object) -> random.Random:
    material = "\0".join(str(part) for part in parts).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def sampled_strategy_indices(
    source_model: str, problem_id: str, seed: int, count: int
) -> list[int]:
    """Choose three raw proposal indices reproducibly, without outcome filtering."""
    if count < 3:
        raise ValueError(
            f"{problem_id}: compression requires at least three generated strategies; "
            f"found {count}"
        )
    indices = list(range(count))
    _stable_rng(
        UNIFORM_COMPRESS_SAMPLE_SEED, source_model, problem_id, seed
    ).shuffle(indices)
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


def compression_source_status(
    config: ExperimentConfig, problem_id: str, seed: int
) -> tuple[bool, str | None]:
    """Return whether a planner bank can enter compression, with a skip reason."""
    path = _source_strategy_path(config, problem_id, seed)
    if not path.is_file():
        return False, "missing planner-only bank"
    strategies = _load_raw_strategies(config, problem_id, seed)
    if len(strategies) < 3:
        return False, f"only {len(strategies)} proposal(s)"
    return True, None


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
    indices = sampled_strategy_indices(
        config.model, problem.problem_id, seed, len(raw_strategies)
    )
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

        async def canonicalize(input_strategy: str, role_prefix: str) -> tuple[str, int]:
            base_prompt = uniform_compress_prompt(
                problem, input_strategy, examples
            )
            strategy = ""
            word_count = 0
            for attempt in range(1, 4):
                role = (
                    role_prefix
                    if attempt == 1
                    else f"{role_prefix}_retry_{attempt - 1}"
                )
                prompt = base_prompt
                if attempt > 1:
                    prompt += (
                        "\n\nYour previous draft violated the required format. Return "
                        "one proof sketch containing 18--25 whitespace-delimited "
                        "words. Preserve the same mathematical mechanisms, specificity, "
                        "omissions, and errors."
                    )
                verdict, _ = await _judge(
                    worker_config,
                    prompt,
                    pool,
                    str(checkpoint.scratch_dir(role)),
                    checkpoint,
                    role,
                    output_schema=COMPRESS_OUTPUT_SCHEMA,
                    allow_tools=False,
                )
                strategy = str(verdict["strategy"]).strip()
                word_count = len(strategy.split())
                if strategy and 18 <= word_count <= 25:
                    break
            if not strategy or not 18 <= word_count <= 25:
                raise ValueError(
                    f"{problem.problem_id}/{role_prefix}: canonicalizer failed the "
                    f"18--25-word limit after 3 attempts ({word_count} words)"
                )
            return strategy, word_count

        compressed: list[dict[str, object]] = []
        for candidate_number, raw_index in enumerate(indices, start=1):
            strategy, word_count = await canonicalize(
                raw_strategies[raw_index], f"candidate_{candidate_number}"
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
            "oracle_strategy_sha256": hashlib.sha256(
                problem.hint_h2.encode("utf-8")
            ).hexdigest(),
            "oracle_word_count": len(problem.hint_h2.split()),
            "strategy_count": 3,
            "candidate_count": 4,
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
    if record.get("proposal_seed") != 1:
        raise ValueError(
            f"{problem.problem_id}: selection candidates must come from proposal seed 1"
        )
    oracle = record.get("oracle_strategy")
    if problem.hint_h2 is None or oracle != problem.hint_h2.strip():
        raise ValueError(
            f"{problem.problem_id}: selection oracle does not match frozen hard hint"
        )
    generated = record.get("generated_strategies")
    if not isinstance(generated, list) or len(generated) != 3:
        raise ValueError(f"{problem.problem_id}: selection needs exactly 3 proposals")
    candidates: list[dict[str, object]] = [
        {
            "candidate_id": "oracle",
            "strategy": oracle,
            "oracle_strategy_match": True,
            "provenance": "oracle",
        }
    ]
    for item in generated:
        if not isinstance(item, dict):
            raise TypeError("Malformed generated selection candidate")
        oracle_strategy_match = item.get("oracle_strategy_match")
        if not isinstance(oracle_strategy_match, bool):
            raise TypeError("Selection candidate lacks oracle_strategy_match label")
        candidates.append(
            {
                "candidate_id": str(item["candidate_id"]),
                "strategy": str(item["strategy"]),
                "oracle_strategy_match": oracle_strategy_match,
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
