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
    "computations. Read and write files ONLY inside the scratch directory given "
    "in the task; do not read or write anywhere else on the machine. This keeps "
    "the run auditable: your entire working record lives in one directory a "
    "reviewer can inspect, and it guarantees you solved the problem from the "
    "statement alone. Access outside the scratch directory is blocked and "
    "logged, so do not attempt it.\n"
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


def self_refine_critique_prompt(max_turns: int) -> str:
    """Build the Self-Refine CRITIQUE prompt (feedback only, no rewrite).

    Self-Refine (Madaan et al. 2023) separates feedback from refinement: this
    phase produces a critique, and a separate revise phase acts on it. The model
    is asked to be an adversarial grader and to name the FIRST genuine gap, so
    the feedback is specific and actionable rather than a vague restatement. It
    keeps the same turn budget so it may re-derive or check a step in scratch to
    decide whether the step is actually justified.
    """
    return (
        f"You now switch roles: you are a strict, adversarial olympiad grader "
        f"reviewing the solution you just wrote (above). You have up to "
        f"{max_turns} tool-use turns for this review.\n\n"
        "Do NOT rewrite the solution in this message. Produce only a critique. "
        "Read your solution as a referee who is trying to find the FIRST place "
        "it genuinely fails: an unproven load-bearing step, a hand-waved claim "
        "(anything resting on 'verified numerically', 'one can check', 'it is "
        "easy to see', 'follows analogously'), a missing case, a wrong bound, or "
        "a misread of the problem. For each issue, quote the exact step and say "
        "precisely why it is not yet a proof. If, after honestly trying to break "
        "it, you find no genuine gap, state 'NO GENUINE GAP FOUND' and briefly "
        "say why the solution is already rigorous. Be concrete: a vague worry is "
        "not useful feedback. End with a section titled '## Critique' listing "
        "the issues in order of severity (most fatal first)."
    )


def self_refine_revise_prompt(max_turns: int) -> str:
    """Build the Self-Refine REVISE prompt (act on the critique, re-emit proof).

    Consumes the critique from the previous phase (same session, full context)
    and produces the final solution. If the critique found no gap, the model
    confirms and restates; otherwise it fixes what it can and — crucially — must
    stay honest about anything it still cannot prove rather than bluffing a
    complete proof over the gap the critique just exposed.
    """
    return (
        f"You now switch back to solver. Using your critique above, produce the "
        f"best possible solution. You have up to {max_turns} tool-use turns.\n\n"
        "Address each issue the critique raised. Fix every gap you can close "
        "with a complete, by-hand-verifiable argument. If the critique found no "
        "genuine gap, confirm that and restate the solution. If some issue "
        "cannot be fully resolved, do NOT paper over it: state the remaining "
        "step EXPLICITLY as an unproven claim and present your best honest "
        "partial progress — an honest partial solution outranks a bluffed "
        "complete one. End with a '## Final Solution' section containing the "
        "complete final write-up (this section is what will be graded)."
    )
