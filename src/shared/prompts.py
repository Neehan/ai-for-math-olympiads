"""System and task prompts for the solving agent (C0 baseline: no KB, no corpus)."""

from src.shared.models import Problem

SYSTEM_PROMPT: str = (
    "You are an expert mathematician competing in an international mathematics "
    "olympiad. You solve hard proof problems with full rigor.\n\n"
    "Rules:\n"
    "- Produce a complete, rigorous, self-contained solution.\n"
    "- If the task requires a proof, prove every claim; do not skip cases or "
    "assert unjustified leaps.\n"
    "- If the task asks for a value or characterization, state the final answer "
    "clearly and prove it is correct.\n"
    "- You have NO internet access: web search, web fetch, and network commands "
    "(curl, wget, pip install, git clone, etc.) are disabled and will be "
    "blocked. Do not attempt them; solve the problem entirely on your own.\n"
    "- You have a scratchpad: you may use the filesystem (Read/Write/Edit) and "
    "Bash (including running scripts) to explore, check small cases, or verify "
    "computations. Write ALL scratch files only inside the scratch directory "
    "given in the task; do not read or write anywhere else on the machine.\n"
    "- The scratchpad is ONLY for you to discover and check ideas. The final "
    "solution must be a self-contained, by-hand-verifiable proof. You may NOT "
    "cite a scratch computation as a proof step. Phrases such as 'verified "
    "symbolically', 'checked numerically', 'a computer-algebra check confirms', "
    "'one can verify', 'it can be shown', 'it is easy to see', and 'follows "
    "analogously' are FORBIDDEN in the final solution: every step must be "
    "justified by a complete written argument a human referee can follow. A "
    "numerical or symbolic check is evidence for YOU, never a substitute for "
    "proof.\n"
    "- Do not claim a problem is solved when it is not. If a step is true but "
    "you cannot prove it by hand, state it EXPLICITLY as an unproven claim and "
    "present your best partial progress; do NOT assert it or hide the gap "
    "behind a computation. An honest partial solution is worth more than a "
    "bluffed complete one.\n"
    "- Adversarially review your solution and make sure it is correct. Do not skip any cases or assert unjustified leaps.\n"
    "- End your final message with a section titled '## Final Solution' "
    "containing the complete write-up."
)


def task_prompt(problem: Problem, scratch_path: str, max_turns: int) -> str:
    """Build the initial task prompt for a problem (statement only, no hints).

    scratch_path is the agent's private working directory; the prompt states it
    explicitly so all file work is confined there and auditable. max_turns is
    stated so the model knows its budget and can pace itself.
    """
    return (
        f"Solve the following olympiad problem.\n\n"
        f"You have a budget of up to {max_turns} tool-use turns for this "
        f"attempt; the run stops automatically when that budget is exhausted, "
        f"so pace yourself and make sure your final write-up is emitted before "
        f"then.\n"
        f"Your private scratch directory is: {scratch_path}\n"
        f"Use it for any files, scripts, or intermediate work. Do not write "
        f"outside it. Your solution is graded from your final message, not from "
        f"files, so the write-up must be complete on its own. Any step you only "
        f"checked in the scratchpad must still be PROVEN in words in the final "
        f"write-up; a numerical or symbolic check is not a proof.\n\n"
        f"Domain: {problem.domain}\n"
        f"Task type: {problem.task}\n"
        f"Expected answer type: {problem.answer_type}\n\n"
        f"Problem:\n{problem.statement.strip()}\n\n"
        f"Work through it carefully, then give your complete solution ending "
        f"with a '## Final Solution' section."
    )


def ralph_refine_prompt(iteration: int, total_iterations: int, max_turns: int) -> str:
    """Build the Ralph refine prompt, stating where the model is in the loop.

    The model is told the current iteration, the total, and its per-iteration
    turn budget so it knows its limits and paces its self-improvement.
    """
    return (
        f"This is refinement iteration {iteration} of {total_iterations} for "
        f"this problem, with up to {max_turns} tool-use turns in this "
        f"iteration.\n\n"
        "Critically review your solution above as if you were a strict olympiad "
        "grader. Find any gap, unjustified step, missing case, computational "
        "error, or place where you claimed something without proof. If you find "
        "issues, fix them and produce an improved complete solution. If the "
        "solution is already fully rigorous and correct, say so explicitly and "
        "restate it. Always end with a '## Final Solution' section containing "
        "the current best write-up."
    )
