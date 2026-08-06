"""Load and validate config.json into typed experiment configuration."""

import dataclasses
import json
from pathlib import Path

from src.constants import HINT_KINDS, HINT_NONE, MODE_IDEASEARCH, MODE_SINGLE, MODES
from src.models import ArmConfig, ExperimentConfig

_VALID_EFFORTS: frozenset[str] = frozenset({"low", "medium", "high", "max"})

_REQUIRED_TOP_KEYS: frozenset[str] = frozenset(
    {
        "model",
        "audit_model",
        "effort",
        "unit_output_tokens",
        "wrap_up_reserve_tokens",
        "ideasearch_plan_tokens",
        "ideasearch_plan_wrap_up_reserve_tokens",
        "max_turns_per_phase",
        "sequential_max_rounds",
        "audit_max_turns",
        "max_concurrency",
        "arms",
    }
)
_REQUIRED_ARM_KEYS: frozenset[str] = frozenset(
    {"hint", "mode", "budget_units", "seeds"}
)


def _validate_keys(raw: dict[str, object], required: frozenset[str], where: str) -> None:
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
    return ArmConfig(name=name, hint=hint, mode=mode, budget_units=budget_units, seeds=seeds)


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
        ideasearch_plan_tokens=int(raw["ideasearch_plan_tokens"]),
        ideasearch_plan_wrap_up_reserve_tokens=int(
            raw["ideasearch_plan_wrap_up_reserve_tokens"]
        ),
        max_turns_per_phase=int(raw["max_turns_per_phase"]),
        sequential_max_rounds=int(raw["sequential_max_rounds"]),
        audit_max_turns=int(raw["audit_max_turns"]),
        max_concurrency=int(raw["max_concurrency"]),
        arms=arms,
    )
    if config.effort not in _VALID_EFFORTS:
        raise ValueError(f"{path}: effort must be one of {sorted(_VALID_EFFORTS)}")
    if config.unit_output_tokens < 1 or config.max_turns_per_phase < 1:
        raise ValueError(f"{path}: token and turn budgets must be positive")
    if config.sequential_max_rounds < 1 or config.audit_max_turns < 1:
        raise ValueError(f"{path}: round and audit turn guards must be positive")
    if not 0 < config.wrap_up_reserve_tokens < config.unit_output_tokens:
        raise ValueError(
            f"{path}: wrap_up_reserve_tokens must be positive and below "
            f"unit_output_tokens"
        )
    if not 0 < config.ideasearch_plan_tokens < config.unit_output_tokens:
        raise ValueError(
            f"{path}: ideasearch_plan_tokens must be positive and below "
            f"unit_output_tokens"
        )
    if not (
        0
        < config.ideasearch_plan_wrap_up_reserve_tokens
        < config.ideasearch_plan_tokens
    ):
        raise ValueError(
            f"{path}: ideasearch_plan_wrap_up_reserve_tokens must be positive "
            f"and below ideasearch_plan_tokens"
        )
    ideasearch_proof_tokens = (
        config.unit_output_tokens - config.ideasearch_plan_tokens
    )
    if config.wrap_up_reserve_tokens >= ideasearch_proof_tokens:
        raise ValueError(
            f"{path}: wrap_up_reserve_tokens must be below the IdeaSearch proof "
            f"budget ({ideasearch_proof_tokens})"
        )
    for arm in config.arms.values():
        if arm.mode != MODE_IDEASEARCH:
            continue
        if (
            arm.hint != HINT_NONE
            or arm.budget_units != 1
            or arm.seeds != list(range(1, 9))
        ):
            raise ValueError(
                f"{path}: IdeaSearch arm '{arm.name}' must use hint='none', "
                f"budget_units=1, and seeds [1, ..., 8]"
            )
    baseline = config.arms.get("baseline")
    parallel = config.arms.get("baseline-parallel")
    if baseline is None or parallel is None:
        raise ValueError(f"{path}: baseline and baseline-parallel arms are required")
    if any(
        arm.hint != HINT_NONE
        or arm.mode != MODE_SINGLE
        or arm.budget_units != 1
        for arm in (baseline, parallel)
    ):
        raise ValueError(
            f"{path}: baseline and baseline-parallel must be no-hint single 1x arms"
        )
    if (
        set(baseline.seeds) & set(parallel.seeds)
        or sorted(baseline.seeds + parallel.seeds) != list(range(1, 9))
    ):
        raise ValueError(
            f"{path}: baseline plus baseline-parallel must define each seed 1..8 "
            "exactly once"
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
