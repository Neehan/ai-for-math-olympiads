# Harnesses

Four solution-generation harnesses built on the Claude Agent SDK, sharing one
tool policy and solver. This is the **C0 baseline** condition of the paper's
condition ladder: no knowledge base, no crux corpus, no oracle hints — the agent
receives only the problem statement. Later conditions (C1 static KB, C2 corpus,
C3 oracle crux) layer onto the same `prompts.py` without touching the harnesses.

- `single_llm/` — one attempt per problem.
- `best_of_n/` — N independent attempts per problem (all stored; no selector,
  no proof verifier — pass@k and selection are decided later by the judge).
- `ralph_loop/` — one persistent session per problem: an initial solution, then
  self-critique/refinement iterations, each recorded.
- `self_refine/` — the cheap reflection baseline (Self-Refine, Madaan et al.
  2023): one persistent session running generate → self-critique → revise, a
  fixed **one** round (~2× the single-LLM budget, vs 8× for BoN/Ralph). Feedback
  is a standalone critique phase, separate from the revise phase; Ralph owns the
  many-round axis, this is the fixed one-round control. Each phase recorded.

`shared/` holds everything common: `constants.py` (single source of truth for
model, tool lists, run parameters, paths), `models.py`, `prompts.py`,
`tool_policy.py`, `bash_guard.py`, `solver.py`, `concurrency.py`,
`io_utils.py`, `logging_setup.py`.

## Model

`claude-opus-4-5` (May-2025 cutoff), set in `shared/constants.py`. The 2026
problems are outside its training window (the NOVEL set).

## Run limits (single source of truth in `constants.py`)

| Limit | Value | Meaning |
|---|---|---|
| `MAX_TURNS_PER_ATTEMPT` | 128 | Tool-use turns per attempt. Same for every system (equivalence). The model is told this budget so it paces itself. |
| `N_SAMPLES` | 8 | Independent Best-of-N samples per problem (matches Ralph's 8 rounds → equal 1024-turn budget). |
| `RALPH_ITERATIONS` | 8 | Ralph rounds per problem (1 solve + 7 refine). |
| `MAX_CONCURRENCY` | 9 | Simultaneous agent sessions across a run (infra knob; no effect on results). |

Worst-case turns per problem: Single = 128, BoN = 8×128 = 1024, Ralph = 8×128 =
1024 (BoN and Ralph matched — same budget, parallel vs sequential). There is **no hard dollar budget cap**: cost is recorded per attempt but
nothing halts on spend. When reporting, state what fraction of attempts hit the
128-turn cap (computable from the logs) — if high, the cap is shaping results.

## Tool policy (contamination control)

Allowed (pre-approved, run headless): Read, Write, Edit, MultiEdit, Bash, Grep,
Glob, TodoWrite. The agent's file work is confined to a per-problem scratch dir
under `.scratch/<harness>/<problem_id>` (git-ignored), stated in the prompt.

Enforcement (tested against this SDK version):
- `disallowed_tools` **removes** WebSearch, WebFetch, Task, Agent, ToolSearch,
  AskUserQuestion, SlashCommand, NotebookEdit. This is the mechanism that
  actually blocks built-ins.
- `bash_network_guard` (PreToolUse hook) blocks network Bash commands (curl,
  wget, git clone/pull/push/fetch, pip/npm/apt install, ssh, nc, urllib, …).
- `can_use_tool` denies anything outside the allowlist as a secondary layer.
  Note: it does **not** reliably gate SDK built-ins — `disallowed_tools` is the
  real guarantee at the tool layer.

The Bash guard is a pattern blocklist (evadable in principle; the model has no
incentive to). Every tool call — including blocked ones — is recorded in the
audit log, so a run can be proven network-free after the fact.

## Outputs (everything is reported)

- `results/<harness>/<problem_id>.md` — human-readable: metadata, a per-attempt
  tool-call summary (results truncated to 500 chars), and the solution text.
- `logs/<harness>/<problem_id>.jsonl` — the audit trail: one JSON line per
  attempt with **full untruncated** tool calls (name, input, result), cost,
  turns, duration, stop reason, and text.

Both are committed deliverables. `.scratch/` (agent working files) is
git-ignored.

## Running

Resumable: each harness skips problems whose result file already exists.

**Run one harness per git branch.** Two harnesses in the same working tree let
the later one Read/Grep the earlier one's `results/`, `logs/`, and `.scratch/` —
cross-harness contamination that voids the experiment. `run_one.sh` runs exactly
one harness and refuses more; isolate each in its own branch, then merge results
to main:

```bash
git checkout main && git checkout -b run/single_llm
./run_one.sh single_llm        # commit results, merge to main
git checkout main && git checkout -b run/best_of_n
./run_one.sh best_of_n         # fresh tree — no trace of single_llm
```

Auth is the Claude Code OAuth token (subscription login). Put it in a `.env`
file (git-ignored, auto-loaded by both the harnesses and `run_one.sh`):

```bash
cp .env.example .env
# then set CLAUDE_CODE_OAUTH_TOKEN=$(claude setup-token) in .env
```

```bash
./run_one.sh single_llm        # one of: single_llm | best_of_n | ralph_loop | self_refine
python -m src.single_llm.run   # a harness directly (also loads .env)
```

## Type checking

```bash
npx pyright        # standard mode; must pass clean (0 errors, 0 warnings)
```
