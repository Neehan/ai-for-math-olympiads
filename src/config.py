"""Load and validate config.json into typed experiment configuration."""

import dataclasses
import json
from pathlib import Path

from src.constants import (
    HINT_KINDS,
    HINT_NONE,
    MODE_PARALLEL,
    MODE_SEQUENTIAL,
    MODE_SINGLE,
    MODE_UNIFORM_STRATEGY,
    MODE_UNIFORM_STRATEGY_ONLY,
    MODE_UNIFORM_COMPRESS,
    MODE_SELECTION,
    MODE_SELECTION_NO_PROBLEM,
    MODES,
)
from src.models import ArmConfig, ExperimentConfig

_VALID_EFFORTS: frozenset[str] = frozenset({"low", "medium", "high", "max"})

_REQUIRED_TOP_KEYS: frozenset[str] = frozenset(
    {
        "model",
        "audit_model",
        "effort",
        "unit_output_tokens",
        "wrap_up_reserve_tokens",
        "uniform_strategy_plan_tokens",
        "uniform_strategy_plan_wrap_up_reserve_tokens",
        "uniform_strategy_branches",
        "max_turns_per_phase",
        "audit_max_turns",
        "max_concurrency",
        "arms",
    }
)
_REQUIRED_ARM_KEYS: frozenset[str] = frozenset(
    {"hint", "mode", "budget_units", "seeds"}
)


def _validate_keys(
    raw: dict[str, object], required: frozenset[str], where: str
) -> None:
    """Fail fast on missing or unknown keys."""
    keys = set(raw)
    if keys != required:
        missing = sorted(required - keys)
        unknown = sorted(keys - required)
        raise ValueError(f"{where}: missing keys {missing}, unknown keys {unknown}")


def _parse_arm(name: str, raw: dict[str, object]) -> ArmConfig:
    """Parse and validate one arm entry."""
    _validate_keys(raw, _REQUIRED_ARM_KEYS, f"arm '{name}'")
    hint = str(raw["hint"])
    mode = str(raw["mode"])
    budget_units = int(str(raw["budget_units"]))
    seeds = [int(str(s)) for s in list(raw["seeds"])]  # type: ignore[call-overload]
    if hint not in HINT_KINDS:
        raise ValueError(f"arm '{name}': hint '{hint}' not in {sorted(HINT_KINDS)}")
    if mode not in MODES:
        raise ValueError(f"arm '{name}': mode '{mode}' not in {sorted(MODES)}")
    if budget_units < 1:
        raise ValueError(f"arm '{name}': budget_units must be >= 1")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"arm '{name}': seeds must be non-empty and unique")
    return ArmConfig(
        name=name, hint=hint, mode=mode, budget_units=budget_units, seeds=seeds
    )


def load_config(path: Path) -> ExperimentConfig:
    """Load config.json, validating every field. Fails loud on any mismatch."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    _validate_keys(raw, _REQUIRED_TOP_KEYS, str(path))
    arms_raw = raw["arms"]
    if not isinstance(arms_raw, dict) or not arms_raw:
        raise ValueError(f"{path}: 'arms' must be a non-empty object")
    arms = {name: _parse_arm(name, spec) for name, spec in arms_raw.items()}
    config = ExperimentConfig(
        model=str(raw["model"]),
        audit_model=str(raw["audit_model"]),
        effort=str(raw["effort"]),
        unit_output_tokens=int(raw["unit_output_tokens"]),
        wrap_up_reserve_tokens=int(raw["wrap_up_reserve_tokens"]),
        uniform_strategy_plan_tokens=int(raw["uniform_strategy_plan_tokens"]),
        uniform_strategy_plan_wrap_up_reserve_tokens=int(
            raw["uniform_strategy_plan_wrap_up_reserve_tokens"]
        ),
        uniform_strategy_branches=int(raw["uniform_strategy_branches"]),
        max_turns_per_phase=int(raw["max_turns_per_phase"]),
        audit_max_turns=int(raw["audit_max_turns"]),
        max_concurrency=int(raw["max_concurrency"]),
        arms=arms,
    )
    if config.effort not in _VALID_EFFORTS:
        raise ValueError(f"{path}: effort must be one of {sorted(_VALID_EFFORTS)}")
    if config.unit_output_tokens < 1 or config.max_turns_per_phase < 1:
        raise ValueError(f"{path}: token and turn budgets must be positive")
    if config.audit_max_turns < 1:
        raise ValueError(f"{path}: audit turn guard must be positive")
    if not 0 < config.wrap_up_reserve_tokens < config.unit_output_tokens:
        raise ValueError(
            f"{path}: wrap_up_reserve_tokens must be positive and below "
            f"unit_output_tokens"
        )
    if not 0 < config.uniform_strategy_plan_tokens < config.unit_output_tokens:
        raise ValueError(
            f"{path}: uniform_strategy_plan_tokens must be positive and below "
            f"unit_output_tokens"
        )
    if not (
        0
        < config.uniform_strategy_plan_wrap_up_reserve_tokens
        < config.uniform_strategy_plan_tokens
    ):
        raise ValueError(
            f"{path}: uniform_strategy_plan_wrap_up_reserve_tokens must be "
            "positive and below uniform_strategy_plan_tokens"
        )
    if config.uniform_strategy_branches < 1:
        raise ValueError(f"{path}: uniform_strategy_branches must be positive")
    for arm in config.arms.values():
        if arm.mode != MODE_UNIFORM_STRATEGY:
            continue
        total_budget = config.budget_tokens(arm)
        executor_pool = total_budget - config.uniform_strategy_plan_tokens
        if executor_pool <= 0 or executor_pool % config.uniform_strategy_branches:
            raise ValueError(
                f"{path}: Uniform Strategy arm '{arm.name}' must leave a positive "
                "executor budget divisible by uniform_strategy_branches"
            )
        executor_budget = executor_pool // config.uniform_strategy_branches
        if arm.hint != HINT_NONE or arm.budget_units != 8 or arm.seeds != [1]:
            raise ValueError(
                f"{path}: Uniform Strategy arm '{arm.name}' must use hint='none', "
                "budget_units=8, and seeds [1]"
            )
        if config.wrap_up_reserve_tokens >= executor_budget:
            raise ValueError(
                f"{path}: wrap_up_reserve_tokens must be below each Uniform "
                f"Strategy executor budget ({executor_budget})"
            )
    auxiliary_specs = {
        "baseline-uniform-strategy-only": (
            MODE_UNIFORM_STRATEGY_ONLY,
            [1],
        ),
        "baseline-uniform-compress": (MODE_UNIFORM_COMPRESS, [1]),
        "selection": (MODE_SELECTION, [1, 2, 3]),
        "selection-no-problem": (MODE_SELECTION_NO_PROBLEM, [1, 2, 3]),
    }
    for name, (mode, seeds) in auxiliary_specs.items():
        arm = config.arms.get(name)
        if arm is None:
            raise ValueError(f"{path}: {name} arm is required")
        if (
            arm.hint != HINT_NONE
            or arm.mode != mode
            or arm.budget_units != 1
            or arm.seeds != seeds
        ):
            raise ValueError(
                f"{path}: {name} must use hint='none', mode={mode!r}, "
                f"budget_units=1, and seeds {seeds}"
            )
    baseline = config.arms.get("baseline")
    parallel = config.arms.get("baseline-parallel")
    if baseline is None or parallel is None:
        raise ValueError(f"{path}: baseline and baseline-parallel arms are required")
    if (
        baseline.hint != HINT_NONE
        or baseline.mode != MODE_SINGLE
        or baseline.budget_units != 1
        or baseline.seeds != [1, 2, 3]
    ):
        raise ValueError(
            f"{path}: baseline must be a no-hint single 1x arm with seeds [1, 2, 3]"
        )
    if (
        parallel.hint != HINT_NONE
        or parallel.mode != MODE_PARALLEL
        or parallel.budget_units != 8
        or parallel.seeds != [1, 2, 3]
    ):
        raise ValueError(
            f"{path}: baseline-parallel must use fresh 8x no-hint banks "
            "with seeds [1, 2, 3]"
        )
    baseline_2x = config.arms.get("baseline-sequential-2x")
    if baseline_2x is None:
        raise ValueError(f"{path}: baseline-sequential-2x arm is required")
    if (
        baseline_2x.hint != HINT_NONE
        or baseline_2x.mode != MODE_SEQUENTIAL
        or baseline_2x.budget_units != 2
        or baseline_2x.seeds != list(range(1, 13))
    ):
        raise ValueError(
            f"{path}: baseline-sequential-2x must use no hint, sequential "
            "mode, two total budget units, and seeds 1 through 12"
        )
    baseline_4x = config.arms.get("baseline-sequential-4x")
    if baseline_4x is None:
        raise ValueError(f"{path}: baseline-sequential-4x arm is required")
    if (
        baseline_4x.hint != HINT_NONE
        or baseline_4x.mode != MODE_SEQUENTIAL
        or baseline_4x.budget_units != 4
        or baseline_4x.seeds != list(range(1, 7))
    ):
        raise ValueError(
            f"{path}: baseline-sequential-4x must use no hint, sequential "
            "mode, four total budget units, and seeds 1 through 6"
        )
    hint = config.arms.get("hint")
    if hint is None:
        raise ValueError(f"{path}: hint arm is required")
    if hint.mode != MODE_SINGLE or hint.budget_units != 1 or hint.seeds != [1, 2, 3]:
        raise ValueError(
            f"{path}: hint must be a single 1x arm with seeds [1, 2, 3]"
        )
    late_baseline = config.arms.get("late-baseline-sequential")
    if late_baseline is None:
        raise ValueError(f"{path}: late-baseline-sequential arm is required")
    if (
        late_baseline.hint != "none"
        or late_baseline.mode != "sequential"
        or late_baseline.budget_units != 4
        or late_baseline.seeds != [1, 2, 3]
    ):
        raise ValueError(
            f"{path}: late-baseline-sequential must use no hint, sequential "
            "mode, four total budget units, and seeds [1, 2, 3]"
        )
    late_hint = config.arms.get("late-hint-sequential")
    if late_hint is None:
        raise ValueError(f"{path}: late-hint-sequential arm is required")
    if (
        late_hint.hint != "h2"
        or late_hint.mode != "sequential"
        or late_hint.budget_units != 4
        or late_hint.seeds != [1, 2, 3]
    ):
        raise ValueError(
            f"{path}: late-hint-sequential must use the h2 oracle strategy, "
            "sequential mode, four total budget units, and seeds [1, 2, 3]"
        )
    if config.max_concurrency < 1:
        raise ValueError(f"{path}: max_concurrency must be >= 1")
    _check_judge_differs(config, str(path))
    return config


def _check_judge_differs(config: ExperimentConfig, where: str) -> None:
    """A solution may never be graded by its author."""
    if config.audit_model == config.model:
        raise ValueError(f"{where}: audit_model must differ from model")


def override_models(
    config: ExperimentConfig, model: str | None, audit_model: str | None
) -> ExperimentConfig:
    """Apply CLI --model/--audit-model overrides on top of config.json.

    Re-validates the judge-differs invariant on the effective pair, so an
    override that collides with the config default fails loud with the fix
    (pass the other flag too).
    """
    if model is None and audit_model is None:
        return config
    effective = dataclasses.replace(
        config,
        model=model if model is not None else config.model,
        audit_model=audit_model if audit_model is not None else config.audit_model,
    )
    _check_judge_differs(effective, "--model/--audit-model")
    return effective


def override_max_concurrency(
    config: ExperimentConfig, max_concurrency: int | None
) -> ExperimentConfig:
    """Apply the operational CLI concurrency override.

    Concurrency controls scheduling only and is deliberately absent from run
    and audit checkpoint identities.
    """
    if max_concurrency is None:
        return config
    if max_concurrency < 1:
        raise ValueError("--max-concurrency must be >= 1")
    return dataclasses.replace(config, max_concurrency=max_concurrency)
