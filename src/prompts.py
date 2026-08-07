"""Load prompt templates from prompts/ and render {{placeholder}} substitutions.

Templates are plain markdown files the experimenter edits directly; rendering
is literal string replacement (no str.format), so LaTeX braces in problem
statements or templates can never break substitution.
"""

import re

from src.constants import (
    AUDIT_PROMPT_FILE,
    CRITIQUE_PROMPT_FILE,
    HINT_PROMPT_FILE,
    PROMPTS_DIR,
    REVISE_PROMPT_FILE,
    SYSTEM_PROMPT_FILE,
    TASK_PROMPT_FILE,
    UNIFORM_STRATEGY_EXECUTE_PROMPT_FILE,
    UNIFORM_STRATEGY_PLAN_PROMPT_FILE,
    UNIFORM_STRATEGY_PLAN_WRAP_UP_PROMPT_FILE,
    WRAP_UP_PROMPT_FILE,
)
from src.models import Problem


def _load(filename: str) -> str:
    """Read one template file from the prompts directory."""
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def _render(template: str, values: dict[str, str]) -> str:
    """Replace each {{key}} with its value; fail loud on unfilled placeholders.

    Placeholders are checked on the TEMPLATE, never on the rendered output —
    substituted content (LaTeX, model solutions) may legitimately contain
    '{{' and must not trip the check. Substitution is a single pass over the
    template, so substituted content is never itself re-substituted.
    """
    names = set(_PLACEHOLDER.findall(template))
    missing = names - set(values)
    if missing:
        raise ValueError(f"Unfilled placeholders in prompt template: {sorted(missing)}")
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], template)


def system_prompt() -> str:
    """The solver system prompt (verbatim from prompts/system.md)."""
    return _load(SYSTEM_PROMPT_FILE)


def task_prompt(
    problem: Problem, hint_text: str | None, scratch_dir: str, budget_tokens: int
) -> str:
    """Initial solve prompt: ONLY the statement (plus scratch dir, token
    budget, and optional hint) — no metadata that could identify the contest.

    hint_text is the raw hint text for the arm's tier (h1 placebo, h2 frozen
    one-sentence strategy hint, or h3 outline) or None for the no-hint arm;
    when present it is wrapped by prompts/hint.md.
    """
    hint_block = (
        ""
        if hint_text is None
        else _render(_load(HINT_PROMPT_FILE), {"hint": hint_text})
    )
    return _render(
        _load(TASK_PROMPT_FILE),
        {
            "budget_tokens": f"{budget_tokens:,}",
            "scratch_dir": scratch_dir,
            "hint_block": hint_block,
            "statement": problem.statement.strip(),
        },
    )


def critique_prompt() -> str:
    """Sequential-channel critique prompt (self-review, no rewrite)."""
    return _load(CRITIQUE_PROMPT_FILE)


def revise_prompt() -> str:
    """Sequential-channel revise prompt (act on the critique, re-emit proof)."""
    return _load(REVISE_PROMPT_FILE)


def wrap_up_prompt(tokens_left: int) -> str:
    """Final wrap-up prompt: stop working, write down the solution now."""
    return _render(_load(WRAP_UP_PROMPT_FILE), {"tokens_left": f"{tokens_left:,}"})


def uniform_strategy_plan_prompt(
    problem: Problem,
    scratch_dir: str,
    budget_tokens: int,
    wrap_up_reserve_tokens: int,
    max_strategies: int,
) -> str:
    """Ask one planner to enumerate a bounded set of distinct strategies."""
    exploration_tokens = budget_tokens - wrap_up_reserve_tokens
    return _render(
        _load(UNIFORM_STRATEGY_PLAN_PROMPT_FILE),
        {
            "budget_tokens": f"{budget_tokens:,}",
            "exploration_tokens": f"{exploration_tokens:,}",
            "wrap_up_reserve_tokens": f"{wrap_up_reserve_tokens:,}",
            "max_strategies": str(max_strategies),
            "scratch_dir": scratch_dir,
            "statement": problem.statement.strip(),
        },
    )


def uniform_strategy_plan_wrap_up_prompt(tokens_left: int, max_strategies: int) -> str:
    """Force the planner to emit its final parseable strategy set."""
    return _render(
        _load(UNIFORM_STRATEGY_PLAN_WRAP_UP_PROMPT_FILE),
        {
            "tokens_left": f"{tokens_left:,}",
            "max_strategies": str(max_strategies),
        },
    )


def uniform_strategy_execute_prompt(
    problem: Problem,
    proposed_strategy: str,
    scratch_dir: str,
    budget_tokens: int,
) -> str:
    """Give a fresh executor only the statement and its assigned strategy."""
    return _render(
        _load(UNIFORM_STRATEGY_EXECUTE_PROMPT_FILE),
        {
            "budget_tokens": f"{budget_tokens:,}",
            "scratch_dir": scratch_dir,
            "proposed_strategy": proposed_strategy.strip(),
            "statement": problem.statement.strip(),
        },
    )


def audit_prompt(problem: Problem, solution_text: str) -> str:
    """Judge prompt: statement + standalone solution (the hint is not included)."""
    return _render(
        _load(AUDIT_PROMPT_FILE),
        {"statement": problem.statement.strip(), "solution": solution_text.strip()},
    )
