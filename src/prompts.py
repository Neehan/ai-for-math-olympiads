"""Load prompt templates from prompts/ and render {{placeholder}} substitutions.

Templates are plain markdown files the experimenter edits directly; rendering
is literal string replacement (no str.format), so LaTeX braces in problem
statements or templates can never break substitution.
"""

from src.constants import (
    AUDIT_PROMPT_FILE,
    CRITIQUE_PROMPT_FILE,
    HINT_PROMPT_FILE,
    PROMPTS_DIR,
    REVISE_PROMPT_FILE,
    SYSTEM_PROMPT_FILE,
    TASK_PROMPT_FILE,
)
from src.models import Problem


def _load(filename: str) -> str:
    """Read one template file from the prompts directory."""
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def _render(template: str, values: dict[str, str]) -> str:
    """Replace each {{key}} with its value; fail loud on leftover placeholders."""
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    if "{{" in rendered:
        leftover = rendered[rendered.index("{{") : rendered.index("{{") + 40]
        raise ValueError(f"Unfilled placeholder in prompt template: '{leftover}'")
    return rendered


def system_prompt() -> str:
    """The solver system prompt (verbatim from prompts/system.md)."""
    return _load(SYSTEM_PROMPT_FILE)


def task_prompt(
    problem: Problem, hint_text: str | None, scratch_dir: str, budget_tokens: int
) -> str:
    """Initial solve prompt: ONLY the statement (plus scratch dir, token
    budget, and optional hint) — no metadata that could identify the contest.

    hint_text is the raw hint text (H1 tags or H2 outline) or None for the
    no-hint arm; when present it is wrapped by prompts/hint.md.
    """
    hint_block = "" if hint_text is None else _render(_load(HINT_PROMPT_FILE), {"hint": hint_text})
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


def audit_prompt(problem: Problem, solution_text: str) -> str:
    """Judge prompt: statement + standalone solution, blind to hint and arm."""
    return _render(
        _load(AUDIT_PROMPT_FILE),
        {"statement": problem.statement.strip(), "solution": solution_text.strip()},
    )
