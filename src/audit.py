"""Audit entrypoint: grade completed attempts of one arm with a judge model.

Usage:
    python -m src.audit --arm baseline
    python -m src.audit --arm hint --problems usamo-2026-3

Per the paper's grading protocol: the judge is a frontier model OTHER than the
solution's author (enforced in config), sees only the problem statement and
the standalone solution.md (hint stripped, blind to arm), and grades on the
near-binary 7/0 standard with a written note (why valid, or what is
missing/wrong). Each attempt's verdict is audit.json in its seed dir
(resumable marker); the per-arm compiled file is
results/<model>/<arm>/audit.jsonl, one line per (problem, seed).
"""

import argparse
import logging

import anyio

from claude_agent_sdk import (
    ClaudeAgentOptions,
    RateLimitEvent,
    ResultMessage,
    query,
)

from src.concurrency import run_all
from src.config import load_config
from src.constants import (
    AGENT_SETTINGS_PATH,
    ALLOWED_TOOLS,
    AUDIT_MAX_TURNS,
    AUDIT_SCORE_INVALID,
    AUDIT_SCORE_VALID,
    AUDIT_SCRATCH_SUBDIR,
    CONFIG_PATH,
    DISALLOWED_TOOLS,
    LOG_FORMAT,
    LOG_LEVEL,
    OAUTH_TOKEN_ENV,
    PERMISSION_MODE,
)
from src.models import ArmConfig, ExperimentConfig, Problem, RateLimitExhausted
from src.prompts import audit_prompt
from src.run import select_problems
from src.solver import run_resumable
from src.storage import (
    archive_audit_scratch,
    budget_cut_multipliers,
    compile_arm_audit,
    cut_solution_path,
    fresh_scratch_dir,
    load_problems,
    seed_audited,
    seed_done,
    seed_output_dir,
    seed_solution_text,
    write_seed_audit,
)
from src.token_pool import TokenPool

log = logging.getLogger("audit")

# Structured output contract for the judge; enforced by the API, so a verdict
# always parses. Score is strictly near-binary: 7 (valid) or 0 (not).
AUDIT_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": [AUDIT_SCORE_INVALID, AUDIT_SCORE_VALID]},
        "note": {"type": "string", "minLength": 1},
    },
    "required": ["score", "note"],
    "additionalProperties": False,
}


def _audit_options(
    config: ExperimentConfig, oauth_token: str, scratch_dir: str
) -> ClaudeAgentOptions:
    """Judge session options: scratch tools to CHECK (audit, not solve),
    structured 7/0 output, opaque scratch cwd (never the repo root).

    Same tool policy as the solver — the judge may verify a computation but,
    per prompts/audit.md, a passing check never substitutes for written proof.
    setting_sources exclusion needs extra_args (a falsy [] is dropped).
    """
    return ClaudeAgentOptions(
        model=config.audit_model,
        effort=config.effort,  # type: ignore[arg-type]
        env={OAUTH_TOKEN_ENV: oauth_token},
        allowed_tools=list(ALLOWED_TOOLS),
        disallowed_tools=list(DISALLOWED_TOOLS),
        settings=str(AGENT_SETTINGS_PATH),
        extra_args={"setting-sources": ""},
        permission_mode=PERMISSION_MODE,
        max_turns=AUDIT_MAX_TURNS,
        cwd=scratch_dir,
        output_format={"type": "json_schema", "schema": AUDIT_OUTPUT_SCHEMA},
    )


async def _judge(
    config: ExperimentConfig, prompt: str, oauth_token: str, scratch_dir: str
) -> dict[str, object]:
    """Run one judge call and return its validated structured verdict."""
    result: ResultMessage | None = None
    rate_limit_reset: int | None = None
    options = _audit_options(config, oauth_token, scratch_dir)
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            result = message
        elif isinstance(message, RateLimitEvent):
            info = message.rate_limit_info
            if info.status == "rejected":
                rate_limit_reset = info.resets_at if info.resets_at is not None else 0
    if rate_limit_reset is not None:
        raise RateLimitExhausted(rate_limit_reset)
    if result is None or result.is_error:
        raise RuntimeError(f"Judge call failed: {result and result.errors}")
    if not isinstance(result.structured_output, dict):
        raise RuntimeError("Judge returned no structured verdict")
    verdict: dict[str, object] = result.structured_output
    return verdict


async def audit_seed(
    config: ExperimentConfig,
    arm: ArmConfig,
    problem: Problem,
    seed: int,
    oauth_token: str,
) -> None:
    """Grade one completed attempt (full solution + any budget-cut snapshots).

    Sequential arms carry solution_<m>x.md snapshots for the saturation curve;
    each is judged as its own standalone proof, so every curve point has an
    audit_score + note. A missing snapshot (no complete write-up within that
    budget) is scored invalid with an explanatory note, no judge call spent.
    """
    output_dir = seed_output_dir(config, arm, problem.problem_id, seed)
    solution = seed_solution_text(output_dir)
    scratch_path = fresh_scratch_dir()
    scratch = str(scratch_path)
    verdict = await _judge(config, audit_prompt(problem, solution), oauth_token, scratch)

    cuts: dict[str, dict[str, object]] = {}
    for multiplier in budget_cut_multipliers(arm.budget_units):
        cut_path = cut_solution_path(output_dir, multiplier)
        if not cut_path.exists():
            cuts[f"{multiplier}x"] = {
                "audit_score": AUDIT_SCORE_INVALID,
                "note": "No complete write-up was emitted within this budget cut.",
            }
            continue
        cut_text = cut_path.read_text(encoding="utf-8")
        cut_verdict = await _judge(
            config, audit_prompt(problem, cut_text), oauth_token, scratch
        )
        cuts[f"{multiplier}x"] = {
            "audit_score": cut_verdict["score"],
            "note": cut_verdict["note"],
        }

    archive_audit_scratch(output_dir, scratch_path)
    write_seed_audit(
        output_dir,
        {
            "problem_id": problem.problem_id,
            "arm": arm.name,
            "seed": seed,
            "solver_model": config.model,
            "audit_model": config.audit_model,
            "audit_score": verdict["score"],
            "note": verdict["note"],
            "budget_cuts": cuts,
        },
    )
    log.info(
        "%s/%s seed %d: score %s (%d cut snapshots graded)",
        arm.name,
        problem.problem_id,
        seed,
        verdict["score"],
        len(cuts),
    )


async def main() -> None:
    """Grade every completed-but-unaudited attempt of one arm, then compile."""
    parser = argparse.ArgumentParser(description="Audit one experiment arm.")
    parser.add_argument("--arm", required=True, help="Arm name from config.json")
    parser.add_argument(
        "--problems", default=None, help="Comma-separated problem ids (default: all)"
    )
    parser.add_argument(
        "--domain", default=None, help="Only problems in this domain (default: all)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    config = load_config(CONFIG_PATH)
    if args.arm not in config.arms:
        raise SystemExit(
            f"Unknown arm '{args.arm}'; config defines {sorted(config.arms)}"
        )
    arm = config.arms[args.arm]
    problems = select_problems(load_problems(), args.problems, args.domain)

    generated = [
        (problem, seed)
        for problem in problems
        for seed in arm.seeds
        if seed_done(seed_output_dir(config, arm, problem.problem_id, seed))
    ]
    ungenerated = len(problems) * len(arm.seeds) - len(generated)
    if ungenerated:
        log.warning(
            "%d attempts have no generation output yet and are skipped", ungenerated
        )
    pending = [
        (problem, seed)
        for problem, seed in generated
        if not seed_audited(seed_output_dir(config, arm, problem.problem_id, seed))
    ]
    log.info(
        "Arm %s: %d attempts to audit, %d already audited",
        arm.name,
        len(pending),
        len(generated) - len(pending),
    )

    pool = TokenPool.from_env()
    tasks = [
        lambda p=problem, s=seed: run_resumable(
            pool, lambda token: audit_seed(config, arm, p, s, token)
        )
        for problem, seed in pending
    ]
    await run_all(tasks, config.max_concurrency)

    path, count = compile_arm_audit(config, arm)
    log.info("Compiled %d verdicts -> %s", count, path)


if __name__ == "__main__":
    anyio.run(main)
