# Harness

One harness implementing the paper's arms (top-level README): every attempt is one (arm, problem, seed) run of the Claude Agent SDK, in one of two modes.

- **single** — one solve phase (arms `baseline`, `baseline-parallel`, `placebo-hint`, `hint`).
- **sequential** — solve → (critique → revise)× under one shared output-token budget, cut off mid-phase when the budget is spent (arms `baseline-sequential`, `hint-sequential`). This is budget-bounded Self-Refine (Madaan et al. 2023); the per-phase cumulative token counts in the logs let the analysis cut the trajectory at 2×/4× for the saturation curve.

## Configuration — `config.json`

Single source of experiment knobs: `model` (solver), `audit_model` (judge; must differ from the solver), `effort` (fixed `high` per the paper), `unit_output_tokens` (1× = 200k), `max_turns_per_phase` (runaway guard), `max_concurrency`, and the arm table (`hint`: none/h1/h2, `mode`, `budget_units`, `seeds`). Arm names are the slugs used everywhere — CLI, `results/` paths, and the top-level README arm table. `baseline` uses seeds 1–3 and `baseline-parallel` seeds 4–8, so together they form the 8 parallel-channel seeds without collision.

## Compute ladder (1×/2×/4×/8×)

- **Parallel:** the 8 independent seeds (`baseline` 1–3 + `baseline-parallel` 4–8) are each audited; pass@1/2/4/8 comes from the standard unbiased estimator (Chen et al. 2021) over the per-seed audit scores — never "first n seeds", which is ordering-dependent noise.
- **Sequential:** one 8× trajectory per seed, cut post hoc. Every phase's full write-up is preserved in the logs (phases are separate records; nothing is overwritten), and the harness snapshots `solution_1x.md` / `solution_2x.md` / `solution_4x.md` — the last COMPLETE non-critique write-up emitted before cumulative output tokens crossed each threshold, i.e. exactly what a hard-stopped run at that budget would have been graded on. `meta.json` records which phase each cut came from. The audit stage grades every snapshot as a standalone proof, so each curve point has its own score + note; a budget with no complete write-up scores 0 with an explanatory note. Caveat to state in the paper: the model knows its full 8× budget, so pacing at lower cuts approximates (rather than replays) a true smaller-budget run — the curve's 1× point should come from the real `baseline` arm.
- **Repeats:** the pre-registered k = 3 runs per cell are the arm's `seeds` (1, 2, 3); each (problem, seed) attempt is fully independent with its own `seed_<k>/` output tree.

## Compute budget enforcement

Compute = the attempt's total output-token budget. Two layers:

1. `task_budget` (API-side): the model is told its remaining token budget so it paces itself.
2. Harness cutoff: every assistant message's `usage.output_tokens` is accumulated (deduped by message id) in a per-attempt `BudgetTracker`; the moment the budget is exceeded the session is interrupted and the phase is marked `budget_exhausted`.

## Problem data (never committed)

Problems and hints are NOT in this repo — committing them would leak contest identity. They are fetched at runtime from the `notadib/math-contests-2026` dataset with stdlib `urllib`, straight into memory (no `hf_hub`, no disk cache):

- `hard_problems.jsonl` — statements. Only `problem_id`, `statement`, and `domain` (for `--domain` filtering) are kept; contest-identifying metadata is dropped at load and the prompt carries the statement alone.
- `hard_hints.jsonl` — technique tags per problem → the **H1** hint (comma-joined).
- `hard_outlines.jsonl` — audited solution outlines → the **H2** hint (numbered steps).

In Docker, the entrypoint prefetches all three BEFORE the egress firewall closes (HuggingFace stays blocked while agents run — an agent that could fetch the hints file would be contaminated); the loader consumes and deletes the temp copies before any agent spawns, so no trace remains.

## Prompts — `prompts/*.md`

All prompts are editable markdown files: `system.md`, `task.md`, `hint.md`, `critique.md`, `revise.md`, `audit.md`. Placeholders use `{{name}}` and are filled by literal replacement (LaTeX braces can never break rendering); an unfilled placeholder fails loud. A hint arm fails fast before spending tokens if a selected problem lacks its hint.

## Isolation (contamination control)

- **Docker is the boundary.** `./run.sh --arm B1` builds a throwaway container (`--rm`) that holds only the harness, problems, and prompts — no reference solutions, nothing else from the machine.
- **Network:** the entrypoint installs an egress firewall (loopback + established + TLS to Anthropic endpoints only, default DROP) and self-tests it before any token is spent: the run aborts unless a non-Anthropic host is unreachable and the API is reachable. Only the LLM connection leaves the container.
- **Tools:** `agent_settings.json` (passed via the SDK's `--settings`) denies `WebSearch`, `WebFetch`, and network/download Bash commands (`curl`, `wget`, `git`, `pip install`, …); `disallowed_tools` strips the multi-agent/web built-ins. `setting_sources=[]` keeps user/project settings out of the run.
- Every tool call is captured untruncated in the logs, so a run can be proven clean after the fact.

## Outputs — `results/<model>/<arm>/<problem_id>/seed_<k>/`

- `logs.jsonl.zst` — one JSON line per phase: prompt, full response text, every tool call (full input/result), per-phase and cumulative output tokens, turns, duration, cost, stop reason.
- `solution.md` — the graded write-up (last non-critique phase), standalone.
- `scratch/` — copy of the agent's scratch directory (the scripts and files it created; the prompt requires all file work to live there).
- `meta.json` — attempt metadata and totals; written last, so its presence is the completion marker for resumable runs.
- `audit.json` — the judge's verdict for this attempt (written by the audit stage).

## Audit (grading)

After generation, `python -m src.audit --arm <slug>` grades every completed attempt. Per the paper's protocol the judge (`audit_model`, config-enforced to differ from the solver) sees only the problem statement and the standalone `solution.md` — hint stripped, blind to arm — and returns a structured near-binary verdict (`audit_score` 7 or 0) plus a `note` saying why the solution is valid or exactly what is missing/wrong. The judge session has no tools and its verdict shape is enforced by a JSON schema, so it always parses. The judge prompt is `prompts/audit.md`, editable like the rest. Verdicts land as `audit.json` per seed (resumable marker) and are compiled into one file per arm: `results/<model>/<arm>/audit.jsonl`, one line per (problem, seed). Audits share the token pool and rate-limit rotation.

## Running

```bash
cp .env.example .env                     # set CLAUDE_CODE_OAUTH_TOKEN (one or more)
./run.sh --arm baseline                  # all problems, baseline arm, in Docker
./run.sh --arm hint --problems id1,id2
./run.sh --arm baseline --domain algebra # one whole domain
python -m src.run --arm baseline         # dev run outside Docker (no firewall)
python -m src.audit --arm baseline       # grade completed attempts (no sandbox needed)
```

Concurrency is async (anyio) under a capacity limiter: at most `max_concurrency` (config.json) agent sessions in flight at once.

Resumable: attempts whose `meta.json` exists are skipped; a rate-limited token cools down and work rotates to the next token, sleeping only when all are cooling. Run one arm per invocation; gated arms (`baseline-parallel`/`baseline-sequential` on `baseline` failures, `hint-sequential` on `hint` failures) take the failure list via `--problems`.

## Type checking

```bash
npx pyright        # standard mode; must pass clean
```
