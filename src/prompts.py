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
    LATE_CONTINUATION_PROMPT_FILE,
    PROMPTS_DIR,
    REVISE_PROMPT_FILE,
    STATE_AUDIT_PROMPT_FILE,
    STRATEGY_STATE_AUDIT_PROMPT_FILE,
    SYSTEM_PROMPT_FILE,
    TASK_PROMPT_FILE,
    UNIFORM_STRATEGY_PLAN_PROMPT_FILE,
    UNIFORM_STRATEGY_PLAN_WRAP_UP_PROMPT_FILE,
    UNIFORM_COMPRESS_PROMPT_FILE,
    SELECTION_PROMPT_FILE,
    SELECTION_WRAP_PROMPT_FILE,
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


def late_continuation_prompt(
    hint_text: str | None, scratch_dir: str, budget_tokens: int
) -> str:
    """Continue a retained native 3x session, optionally injecting h2."""
    hint_block = (
        ""
        if hint_text is None
        else _render(_load(HINT_PROMPT_FILE), {"hint": hint_text.strip()})
    )
    return _render(
        _load(LATE_CONTINUATION_PROMPT_FILE),
        {
            "budget_tokens": f"{budget_tokens:,}",
            "scratch_dir": scratch_dir,
            "hint_block": hint_block,
        },
    )


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
    """Use the exact strategy task wrapper for a planner strategy.

    Keeping one rendering path prevents executor-level wording, formatting, or
    authority cues from differing between Uniform Search and the privileged
    strategy arm. Strategy provenance remains recorded in bank metadata, not
    exposed through a different solver instruction.
    """
    return task_prompt(problem, proposed_strategy.strip(), scratch_dir, budget_tokens)


def audit_prompt(
    problem: Problem, reference_solution: str, solution_text: str
) -> str:
    """Reference-assisted correctness prompt for one standalone solution."""
    if not reference_solution.strip():
        raise ValueError(f"{problem.problem_id}: no verified reference solution")
    return _render(
        _load(AUDIT_PROMPT_FILE),
        {
            "statement": problem.statement.strip(),
            "reference_solution": reference_solution.strip(),
            "solution": solution_text.strip(),
        },
    )


def state_audit_prompt(
    problem: Problem,
    outline: str,
    reference_solution: str,
    solution_text: str,
) -> str:
    """Reference-guided outline annotation for one solution artifact."""
    return _render(
        _load(STATE_AUDIT_PROMPT_FILE),
        {
            "statement": problem.statement.strip(),
            "outline": outline.strip(),
            "reference_solution": reference_solution.strip(),
            "solution": solution_text.strip(),
        },
    )


def strategy_state_audit_prompt(
    problem: Problem,
    oracle_strategy: str,
    reference_solution: str,
    strategy_text: str,
) -> str:
    """Reference-guided oracle-route matching for a proposed strategy."""
    return _render(
        _load(STRATEGY_STATE_AUDIT_PROMPT_FILE),
        {
            "statement": problem.statement.strip(),
            "oracle_strategy": oracle_strategy.strip(),
            "reference_solution": reference_solution.strip(),
            "strategy": strategy_text.strip(),
        },
    )


def uniform_compress_prompt(
    problem: Problem,
    strategy_text: str,
    examples: list[tuple[str, str]],
) -> str:
    """Compress one generated strategy in the frozen oracle-sketch style."""
    rendered_examples = "\n\n".join(
        f"Example {index}\nProblem:\n{statement.strip()}\nSketch:\n{hint.strip()}"
        for index, (statement, hint) in enumerate(examples, start=1)
    )
    return _render(
        _load(UNIFORM_COMPRESS_PROMPT_FILE),
        {
            "examples": rendered_examples,
            "statement": problem.statement.strip(),
            "strategy": strategy_text.strip(),
        },
    )


def selection_prompt(
    problem: Problem,
    candidates: list[str],
    budget_tokens: int,
    reserve_tokens: int,
) -> str:
    """Rank four anonymous strategy sketches for the stated problem."""
    return _render(
        _load(SELECTION_PROMPT_FILE),
        {
            "statement": problem.statement.strip(),
            "budget_tokens": str(budget_tokens),
            "working_tokens": str(budget_tokens - reserve_tokens),
            "reserve_tokens": str(reserve_tokens),
            "candidates": "\n\n".join(
                f"Strategy {index}: {candidate.strip()}"
                for index, candidate in enumerate(candidates, start=1)
            ),
        },
    )


def selection_no_problem_prompt(
    candidates: list[str], budget_tokens: int, reserve_tokens: int
) -> str:
    """Render the identical selection task with only the statement withheld."""
    return _render(
        _load(SELECTION_PROMPT_FILE),
        {
            "statement": "[WITHHELD FOR THIS CONTROL]",
            "budget_tokens": str(budget_tokens),
            "working_tokens": str(budget_tokens - reserve_tokens),
            "reserve_tokens": str(reserve_tokens),
            "candidates": "\n\n".join(
                f"Strategy {index}: {candidate.strip()}"
                for index, candidate in enumerate(candidates, start=1)
            )
        },
    )


def selection_wrap_prompt(remaining_tokens: int) -> str:
    """Request the final structured ranking inside the reserved half-budget."""
    return _render(
        _load(SELECTION_WRAP_PROMPT_FILE),
        {"remaining_tokens": str(remaining_tokens)},
    )
