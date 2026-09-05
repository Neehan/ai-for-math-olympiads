#!/usr/bin/env python3
"""Reproduce the paper's proposal--execution allocation curves exactly.

Predictions use the finite-bank estimator in Equation 7 of the paper.  The
implementation sums over binary acquisition patterns with their exact
without-replacement multiplicities; this is algebraically identical to
enumerating every ordered permutation of raw Parallel observations, but is
much faster.  No Monte Carlo sampling is used.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PASSING_SCORE = 5


@dataclass(frozen=True)
class Profile:
    key: str
    label: str
    model: str
    results_roots: tuple[str, ...]
    n_arms: int
    max_blocks: int
    observed_arms: tuple[str, ...]


PAPER_PROFILES = (
    Profile(
        "muse-n1",
        "Muse Spark 1.2; N=1, K<=8",
        "muse-spark-1.2-contributor",
        ("results", "results-imobench"),
        1,
        8,
        ("baseline-sequential",),
    ),
    Profile(
        "opus-n1",
        "Claude Opus 4.8; N=1, K<=8",
        "claude-opus-4-8",
        ("results",),
        1,
        8,
        ("baseline-sequential",),
    ),
    Profile(
        "gpt54-n1",
        "GPT-5.4; N=1, K<=8",
        "litellm-gpt-5.4",
        ("results", "results-imobench"),
        1,
        8,
        ("baseline-sequential",),
    ),
    Profile(
        "gpt55-n1",
        "GPT-5.5; N=1, K<=8",
        "litellm-gpt-5.5",
        ("results",),
        1,
        8,
        ("baseline-sequential",),
    ),
    Profile(
        "muse-n2",
        "Muse Spark 1.2; N=2, K<=4",
        "muse-spark-1.2-contributor",
        ("results", "results-imobench"),
        2,
        4,
        ("baseline-sequential", "late-baseline-sequential"),
    ),
    Profile(
        "gpt54-n2",
        "GPT-5.4; N=2, K<=4",
        "litellm-gpt-5.4",
        ("results", "results-imobench"),
        2,
        4,
        ("baseline-sequential", "late-baseline-sequential"),
    ),
)


@dataclass
class RootData:
    proposals: dict[str, list[bool]]
    executions: dict[str, list[dict[int, bool]]]
    observed: dict[str, dict[tuple[str, int], dict[int, bool]]]


@dataclass(frozen=True)
class Report:
    profile: Profile
    trials: int
    x: tuple[int, ...]
    observed: tuple[int, ...]
    predicted: tuple[float, ...]
    mae: float
    rmse: float


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        records.append(value)
    return records


def _passing(value: object, *, threshold: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= threshold


def _proof_curve(
    record: Mapping[str, object],
    *,
    final_block: int,
    threshold: int,
) -> dict[int, bool]:
    raw_cuts = record.get("budget_cuts", {})
    if not isinstance(raw_cuts, dict):
        raise ValueError("budget_cuts must be an object")
    curve: dict[int, bool] = {}
    for label, raw_cut in raw_cuts.items():
        if not isinstance(label, str) or not label.endswith("x"):
            continue
        if not isinstance(raw_cut, dict):
            raise ValueError(f"budget cut {label!r} must be an object")
        curve[int(label[:-1])] = _passing(raw_cut.get("audit_score"), threshold=threshold)
    curve[final_block] = _passing(record.get("audit_score"), threshold=threshold)
    missing = [block for block in range(1, final_block + 1) if block not in curve]
    if missing:
        raise ValueError(
            f"Missing proof cuts {missing} for {record.get('arm')}/"
            f"{record.get('problem_id')} seed {record.get('seed')}"
        )
    return curve


def _load_root(
    root: Path,
    model: str,
    observed_arms: Sequence[str],
    *,
    threshold: int,
) -> RootData:
    model_root = root / model
    proposals: dict[str, list[bool]] = defaultdict(list)
    proposal_keys: set[tuple[str, int, int]] = set()
    for record in _read_jsonl(model_root / "baseline-parallel/state_audit.jsonl"):
        problem = str(record["problem_id"])
        key = (problem, int(record["seed"]), int(record["parallel_run"]))
        if key in proposal_keys:
            raise ValueError(f"Duplicate Parallel observation in {root}: {key}")
        proposal_keys.add(key)
        proposals[problem].append(record.get("state") == "S")

    executions: dict[str, list[dict[int, bool]]] = defaultdict(list)
    execution_keys: set[tuple[str, int]] = set()
    for record in _read_jsonl(model_root / "hint-sequential/audit.jsonl"):
        problem = str(record["problem_id"])
        key = (problem, int(record["seed"]))
        if key in execution_keys:
            raise ValueError(f"Duplicate oracle observation in {root}: {key}")
        execution_keys.add(key)
        executions[problem].append(
            _proof_curve(record, final_block=8, threshold=threshold)
        )

    observed: dict[str, dict[tuple[str, int], dict[int, bool]]] = {}
    for arm in observed_arms:
        final_block = 8 if arm == "baseline-sequential" else 4
        arm_records: dict[tuple[str, int], dict[int, bool]] = {}
        for record in _read_jsonl(model_root / arm / "audit.jsonl"):
            key = (str(record["problem_id"]), int(record["seed"]))
            if key in arm_records:
                raise ValueError(f"Duplicate observed trajectory in {root}/{arm}: {key}")
            arm_records[key] = _proof_curve(
                record,
                final_block=final_block,
                threshold=threshold,
            )
        observed[arm] = arm_records

    return RootData(dict(proposals), dict(executions), observed)


def _permutation_count(total: int, selected: int) -> int:
    if selected < 0 or selected > total:
        return 0
    return math.factorial(total) // math.factorial(total - selected)


def _arm_success(
    proposal_pattern: Sequence[bool],
    execution: Mapping[int, bool],
) -> bool:
    blocks = len(proposal_pattern)
    for position, acquired in enumerate(proposal_pattern, 1):
        if acquired:
            return execution[blocks - position + 1]
    return False


def exact_allocation_probability(
    proposals: Sequence[bool],
    executions: Sequence[Mapping[int, bool]],
    *,
    n_arms: int,
    blocks_per_arm: int,
) -> float:
    """Evaluate Equation 7 exactly from finite binary intervention banks."""

    selected = n_arms * blocks_per_arm
    m_p = len(proposals)
    m_e = len(executions)
    if m_p < selected:
        raise ValueError(f"Need {selected} Parallel observations, found {m_p}")
    if m_e < n_arms:
        raise ValueError(f"Need {n_arms} oracle observations, found {m_e}")

    acquired_count = sum(proposals)
    proposal_denominator = _permutation_count(m_p, selected)
    oracle_permutations = tuple(itertools.permutations(range(m_e), n_arms))
    oracle_denominator = len(oracle_permutations)
    probability = 0.0

    # A status pattern with u acquisitions represents (c)_u (m-c)_(L-u)
    # ordered raw permutations.  Grouping identical patterns avoids iterating
    # over as many as 24!/(24-8)! branch identities.
    for pattern in itertools.product((False, True), repeat=selected):
        successes = sum(pattern)
        pattern_count = (
            _permutation_count(acquired_count, successes)
            * _permutation_count(m_p - acquired_count, selected - successes)
        )
        if pattern_count == 0:
            continue
        successful_oracle_assignments = 0
        for oracle_indices in oracle_permutations:
            allocation_succeeds = False
            for arm in range(n_arms):
                start = arm * blocks_per_arm
                stop = start + blocks_per_arm
                if _arm_success(
                    pattern[start:stop], executions[oracle_indices[arm]]
                ):
                    allocation_succeeds = True
                    break
            successful_oracle_assignments += allocation_succeeds
        probability += (
            pattern_count
            / proposal_denominator
            * successful_oracle_assignments
            / oracle_denominator
        )
    return probability


def brute_force_allocation_probability(
    proposals: Sequence[bool],
    executions: Sequence[Mapping[int, bool]],
    *,
    n_arms: int,
    blocks_per_arm: int,
) -> float:
    """Literal Equation 7 enumeration, used only for small-case verification."""

    selected = n_arms * blocks_per_arm
    outcomes = 0
    count = 0
    for proposal_indices in itertools.permutations(range(len(proposals)), selected):
        for oracle_indices in itertools.permutations(range(len(executions)), n_arms):
            allocation_succeeds = False
            for arm in range(n_arms):
                start = arm * blocks_per_arm
                stop = start + blocks_per_arm
                pattern = [proposals[index] for index in proposal_indices[start:stop]]
                if _arm_success(pattern, executions[oracle_indices[arm]]):
                    allocation_succeeds = True
                    break
            outcomes += allocation_succeeds
            count += 1
    if count == 0:
        raise ValueError("No valid permutations")
    return outcomes / count


def _matched_observations(
    data: RootData,
    arms: Sequence[str],
    *,
    max_blocks: int,
) -> tuple[dict[str, int], list[int]]:
    common = set(data.observed[arms[0]])
    for arm in arms[1:]:
        common &= set(data.observed[arm])
    if not common:
        raise ValueError(f"No matched observed trials across {arms}")

    weights: dict[str, int] = defaultdict(int)
    observed = [0] * max_blocks
    for key in sorted(common):
        problem, _seed = key
        weights[problem] += 1
        for block in range(1, max_blocks + 1):
            observed[block - 1] += any(
                data.observed[arm][key][block] for arm in arms
            )
    return dict(weights), observed


def build_report(profile: Profile, *, threshold: int = PASSING_SCORE) -> Report:
    predicted = [0.0] * profile.max_blocks
    observed = [0] * profile.max_blocks
    trials = 0
    for root_name in profile.results_roots:
        data = _load_root(
            REPO_ROOT / root_name,
            profile.model,
            profile.observed_arms,
            threshold=threshold,
        )
        weights, root_observed = _matched_observations(
            data,
            profile.observed_arms,
            max_blocks=profile.max_blocks,
        )
        for block, count in enumerate(root_observed):
            observed[block] += count
        trials += sum(weights.values())
        for problem, weight in weights.items():
            try:
                proposals = data.proposals[problem]
                executions = data.executions[problem]
            except KeyError as error:
                raise ValueError(
                    f"Missing prediction inputs for {root_name}/{profile.model}/{problem}"
                ) from error
            for block in range(1, profile.max_blocks + 1):
                predicted[block - 1] += weight * exact_allocation_probability(
                    proposals,
                    executions,
                    n_arms=profile.n_arms,
                    blocks_per_arm=block,
                )

    errors = [prediction - outcome for prediction, outcome in zip(predicted, observed)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    return Report(
        profile=profile,
        trials=trials,
        x=tuple(profile.n_arms * block for block in range(1, profile.max_blocks + 1)),
        observed=tuple(observed),
        predicted=tuple(predicted),
        mae=mae,
        rmse=rmse,
    )


def _tex_coordinates(x: Iterable[int], y: Iterable[float]) -> str:
    return "coordinates {" + " ".join(
        f"({budget},{value:.3f})" for budget, value in zip(x, y)
    ) + "};"


def _report_as_dict(report: Report) -> dict[str, object]:
    return {
        "profile": report.profile.key,
        "label": report.profile.label,
        "trials": report.trials,
        "x": list(report.x),
        "observed": list(report.observed),
        "predicted": [round(value, 6) for value in report.predicted],
        "mae": round(report.mae, 6),
        "rmse": round(report.rmse, 6),
        "final_observed": report.observed[-1],
        "final_predicted": round(report.predicted[-1], 6),
    }


def run_self_check() -> None:
    cases = (
        ([True, False, False], [{1: True}, {1: True}], 2, 1),
        (
            [True, False, True, False],
            [{1: False, 2: True}, {1: True, 2: True}],
            2,
            2,
        ),
        (
            [False, True, False, True, False],
            [{1: True, 2: True}, {1: False, 2: True}, {1: True, 2: False}],
            1,
            2,
        ),
    )
    for proposals, executions, n_arms, blocks in cases:
        compressed = exact_allocation_probability(
            proposals,
            executions,
            n_arms=n_arms,
            blocks_per_arm=blocks,
        )
        brute_force = brute_force_allocation_probability(
            proposals,
            executions,
            n_arms=n_arms,
            blocks_per_arm=blocks,
        )
        if not math.isclose(compressed, brute_force, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError((compressed, brute_force))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        action="append",
        choices=[profile.key for profile in PAPER_PROFILES],
        help="Paper profile to report; repeat as needed (default: all).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human/TeX report.",
    )
    parser.add_argument(
        "--passing-score",
        type=int,
        default=PASSING_SCORE,
        help=f"Minimum proof-audit score (default: {PASSING_SCORE}).",
    )
    args = parser.parse_args()

    run_self_check()
    selected = set(args.profile or ())
    profiles = [
        profile for profile in PAPER_PROFILES if not selected or profile.key in selected
    ]
    reports = [build_report(profile, threshold=args.passing_score) for profile in profiles]
    if args.json:
        print(json.dumps([_report_as_dict(report) for report in reports], indent=2))
        return

    print("Exact finite-bank self-checks: passed")
    for report in reports:
        print(f"\n{report.profile.key}: {report.profile.label}")
        print(f"  trials:    {report.trials}")
        print(f"  observed:  {list(report.observed)}")
        print(f"  predicted: {[round(value, 3) for value in report.predicted]}")
        print(f"  MAE/RMSE:  {report.mae:.2f} / {report.rmse:.2f}")
        print(f"  TeX:       {_tex_coordinates(report.x, report.predicted)}")


if __name__ == "__main__":
    main()
