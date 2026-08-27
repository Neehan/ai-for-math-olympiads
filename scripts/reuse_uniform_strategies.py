"""Materialize planner-only artifacts from completed Uniform-C planner banks."""

import argparse
import json
import shutil
from pathlib import Path


def reuse(
    results_root: Path, model: str, problems: set[str] | None = None
) -> int:
    model_dir = model.replace("/", "-")
    source_root = results_root / model_dir / "baseline-uniform-strategy"
    target_root = results_root / model_dir / "baseline-uniform-strategy-only"
    copied = 0
    for source in sorted(source_root.glob("*/seed_1")):
        if problems is not None and source.parent.name not in problems:
            continue
        strategies_path = source / "strategies.json"
        source_meta_path = source / "meta.json"
        if not strategies_path.is_file() or not source_meta_path.is_file():
            continue
        artifact = json.loads(strategies_path.read_text(encoding="utf-8"))
        strategies = artifact.get("strategies")
        if not isinstance(strategies, list):
            continue
        if not all(isinstance(item, str) and item.strip() for item in strategies):
            raise ValueError(f"Malformed strategy bank: {strategies_path}")
        source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
        if not strategies and not isinstance(source_meta.get("planner_failure"), str):
            continue
        target = target_root / source.parent.name / source.name
        target_meta_path = target / "meta.json"
        target_strategies_path = target / "strategies.json"
        if target_meta_path.is_file() and target_strategies_path.is_file():
            try:
                target_meta = json.loads(target_meta_path.read_text(encoding="utf-8"))
                target_artifact = json.loads(
                    target_strategies_path.read_text(encoding="utf-8")
                )
                target_strategies = target_artifact.get("strategies")
                if (
                    target_meta.get("mode") == "uniform_strategy_only"
                    and isinstance(target_strategies, list)
                    and len(target_strategies) == target_meta.get("strategy_count")
                    and (
                        bool(target_strategies)
                        or isinstance(target_meta.get("planner_failure"), str)
                    )
                    and (target / "solution.md").is_file()
                ):
                    continue
            except (json.JSONDecodeError, OSError):
                pass
        target.mkdir(parents=True, exist_ok=True)
        # A rewritten proposal set invalidates any derived annotations from a
        # malformed/partial earlier target. Valid frozen targets return above.
        for stale_name in ("audit.json", "state_audit.json"):
            (target / stale_name).unlink(missing_ok=True)
        target_strategies_path.write_text(
            json.dumps(
                {"strategies": strategies, "run_strategy_indices": []},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        for name in ("logs.jsonl.zst", "plan_scratch"):
            src = source / name
            dst = target / name
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            elif src.is_file():
                shutil.copy2(src, dst)
        (target / "solution.md").write_text(
            "Planner-only Uniform-C strategy bank; no proof artifact.\n",
            encoding="utf-8",
        )
        meta = {
            "problem_id": source_meta.get("problem_id", source.parent.name),
            "arm": "baseline-uniform-strategy-only",
            "mode": "uniform_strategy_only",
            "hint": "none",
            "model": model,
            "effort": source_meta.get("effort"),
            "seed": 1,
            "budget_output_tokens": source_meta.get(
                "uniform_strategy_plan_budget_output_tokens", 80000
            ),
            "output_tokens_spent": source_meta.get(
                "uniform_strategy_plan_output_tokens_spent", 0
            ),
            "uniform_strategy_plan_budget_output_tokens": source_meta.get(
                "uniform_strategy_plan_budget_output_tokens", 80000
            ),
            "uniform_strategy_plan_output_tokens_spent": source_meta.get(
                "uniform_strategy_plan_output_tokens_spent", 0
            ),
            "strategy_count": len(strategies),
            "provider_session_ids": source_meta.get("provider_session_ids", {}),
            "gradeable_solution_emitted": False,
            "reused_from": (
                f"baseline-uniform-strategy/{source.parent.name}/seed_1"
            ),
        }
        if not strategies:
            meta["planner_failure"] = source_meta["planner_failure"]
        target_meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--problems", default=None)
    args = parser.parse_args()
    problems = (
        {problem.strip() for problem in args.problems.split(",") if problem.strip()}
        if args.problems
        else None
    )
    count = reuse(args.results_root, args.model, problems)
    if count:
        print(f"reused {count} existing Uniform-C planner bank(s)")


if __name__ == "__main__":
    main()
