# Harness

One harness implementing the paper's arms (top-level README): every attempt is one (arm, problem, seed) run of the Claude Agent SDK, in one of three modes.

- **single** — one solve phase (arms `baseline`, `baseline-parallel`, `placebo-hint`, `hint`, `outline`).
- **sequential** — solve → (critique → revise)× under one shared output-token budget, cut off mid-phase when the budget is spent (arms `baseline-sequential`, `hint-sequential`, `outline-sequential`). Two consecutive critiques containing the exact standalone verdict `NO GENUINE GAP FOUND` stop the trajectory as self-converged; one optimistic critique does not. This is budget-bounded Self-Refine (Madaan et al. 2023); the per-phase cumulative token counts in the logs let the analysis cut the trajectory at 2×/4× for the saturation curve.
- **ideasearch** — one fresh planner (≤20k) proposes a candidate strategy, then one fresh executor (≤180k) receives only the problem and its branch's plan and writes the proof. `baseline-ideasearch` has eight independent seeds/branches, an IdeaSearch-8 adaptation for proof generation.

## Configuration — `config.json`

Single source of experiment knobs: `model` (solver), `audit_model` (judge; must differ from the solver) — both overridable per invocation with `--model` / `--audit-model` (config values are the defaults; the solver≠judge check applies to the effective pair, and audit's `--model` selects whose results tree to grade), `effort` (fixed `high` per the paper), `unit_output_tokens` (1× = 200k), the IdeaSearch plan and wrap reserves, `max_turns_per_phase` / `audit_max_turns` (per-phase guards; the token budget is the sequential stop), `max_concurrency`, and the arm table (`hint`: none/h1/h2/h3, `mode`, `budget_units`, `seeds`). Models are Anthropic ids (`claude-opus-4-8`) or OpenRouter `vendor/model` ids (`openai/gpt-5.5`), which route through OpenRouter's Anthropic-compatible endpoint using `OPENROUTER_API_KEY*` keys from `.env` (same round-robin pool scheme); results paths use the model id with `/` replaced by `-`. Arm names are the slugs used everywhere — CLI, `results/` paths, and the top-level README arm table. `baseline` uses seeds 1–3 and `baseline-parallel` seeds 4–8, so together they form the 8 parallel-channel seeds without collision.

## Compute ladder (1×/2×/4×/8×)

- **Parallel:** the 8 independent seeds (`baseline` 1–3 + `baseline-parallel` 4–8) are each audited; pass@1/2/4/8 comes from the standard unbiased estimator (Chen et al. 2021) over the per-seed audit scores — never "first n seeds", which is ordering-dependent noise.
- **Sequential:** one 8× trajectory per seed, cut post hoc. Every phase's full write-up is preserved in the logs (phases are separate records; nothing is overwritten), and the harness snapshots `solution_1x.md` / `solution_2x.md` / `solution_4x.md` — the last COMPLETE non-critique write-up emitted before cumulative output tokens crossed each threshold, i.e. exactly what a hard-stopped run at that budget would have been graded on. `meta.json` records which phase each cut came from. The audit stage grades every snapshot as a standalone proof, so each curve point has its own score + note; a budget with no complete write-up scores 0 with an explanatory note. Caveat to state in the paper: the model knows its full 8× budget, so pacing at lower cuts approximates (rather than replays) a true smaller-budget run — the curve's 1× point should come from the real `baseline` arm.
- **Repeats:** the pre-registered k = 3 runs per cell are the arm's `seeds` (1, 2, 3); each (problem, seed) attempt is fully independent with its own `seed_<k>/` output tree.
- **IdeaSearch-8:** seeds 1–8 are independent branches, not repeats of one shared search tree. Planner and executor use separate SDK sessions and scratch directories; unused planner tokens do not transfer to the executor.

## Compute budget enforcement

Compute = the attempt's total output-token budget. Three layers:

1. `task_budget` (API-side): the model is told its remaining token budget so it paces itself.
2. Harness cutoff: streamed `message_delta` events carry each API message's real output-token count (partial messages enabled; the per-message `usage` on assistant events is only an initial snapshot) and feed a per-attempt `BudgetTracker` that interrupts the session the moment a cutoff is crossed; each phase's total is then trued-up from the ResultMessage's exact per-query usage. Single responses are capped at 64k via `CLAUDE_CODE_MAX_OUTPUT_TOKENS` — in Docker the CLI is pinned to npm `2.1.222` via `HARNESS_CLI_PATH`, because the SDK-bundled CLI ignores that cap on Opus and errors long thinking turns at 32k.
3. Wrap-up protocol: working phases run against the soft limit (budget − `wrap_up_reserve_tokens`, config). When it is reached, the harness injects a wrap-up prompt — "you have ~N tokens left, stop working, write down what you have" — and only that final phase may spend the reserve, up to the hard budget. Every graded artifact is therefore a deliberate write-up, not a mid-sentence truncation; the reserve is inside the budget and identical across all arms and cells (same protocol as an exam room's time call). API-error phases are never written as results.

## Problem data (never committed)

Problems and hints are NOT in this repo — committing them would leak contest identity. They are fetched at runtime from the `notadib/math-contests-2026` dataset with stdlib `urllib`, straight into memory (no `hf_hub`, no disk cache):

- `hard_problems.jsonl` — statements. Only `problem_id`, `statement`, and `domain` (for `--domain` filtering) are kept; contest-identifying metadata is dropped at load and the prompt carries the statement alone.
- `hard_hints.jsonl` — hint ladder source: `placebo` field → **h1** (not authored yet — placebo arms fail fast before spending a token), scalar `hint` field → **h2** (the frozen ≤25-word oracle strategy hint, inserted verbatim). The retired five-tag development file is archived on HuggingFace as `hard_hints-v1.jsonl` and is never fetched by the harness.
- `hard_outlines.jsonl` — audited strategy outlines → **h3** (numbered steps; the `outline` and `outline-sequential` arms).

In Docker, the entrypoint prefetches all three BEFORE the egress firewall closes (HuggingFace stays blocked while agents run — an agent that could fetch the hints file would be contaminated); the loader consumes and deletes the temp copies before any agent spawns, so no trace remains.

## Prompts — `prompts/*.md`

All prompts are editable markdown files, including the three `ideasearch_*.md` templates. Placeholders use `{{name}}` and are filled by literal replacement (LaTeX braces can never break rendering); an unfilled placeholder fails loud. A hint arm fails fast before spending tokens if a selected problem lacks its hint.

## Isolation (contamination control)

- **Docker is the boundary.** `./run.sh run --arm <slug>` builds a throwaway container (`--rm`) that holds only the harness, problems, and prompts — no reference solutions, nothing else from the machine.
- **Prior results never enter a generation container.** The run stage mounts a staging dir pre-seeded with only `meta.json` completion markers (resume still works), then merges only newly completed attempts containing both `solution.md` and the last-written `meta.json`. Marker-only resume inputs and partial writes are discarded rather than re-merged, preventing a long-running arm from resurrecting stale results archived while it was active. An agent can never read past arms' solutions or hint-carrying logs, even by exploring the filesystem. The audit stage mounts the real tree (the judge must read solutions).
- **Network:** the entrypoint installs an egress firewall (loopback + established + TLS to the LLM API endpoints only — Anthropic and `openrouter.ai` — default DROP) and self-tests it before any token is spent: the run aborts unless a non-allowlisted host is unreachable and both APIs are reachable. Only the LLM connection leaves the container.
- **Tools:** `agent_settings.json` (passed via the SDK's `--settings`) denies `WebSearch`, `WebFetch`, and network/download Bash commands (`curl`, `wget`, `git`, `pip install`, …); `disallowed_tools` strips the web, subagent/fleet, publishing, and background-scheduling built-ins. User/project settings are excluded by passing `--setting-sources` explicitly empty (the SDK silently drops a falsy `setting_sources=[]`).
- **Live transcripts:** each solver, planner, executor, and judge call has an isolated provider UUID and transcript in a host-persisted opaque workspace (for example `/c/w/a1b2c3d4`); concurrent seeds never share one, runtime state is excluded from archived scratch, and Docker mounts only the current stage/model/arm checkpoint namespace so other interventions remain invisible.
- Every tool call is captured untruncated in the logs, so a run can be proven clean after the fact.

## Outputs — `results/<model>/<arm>/<problem_id>/seed_<k>/`

- `logs.jsonl.zst` — one JSON line per phase: prompt, full response text, every tool call (full input/result), per-phase and cumulative output tokens, turns, duration, cost, stop reason, and any credential reconnects.
- `solution.md` — the graded artifact: the last COMPLETE proof-producing phase (normally the wrap-up), same convention as the budget cuts so the full-budget point can never score below a lower cut by truncation artifact. The judge grades ONLY its `## Final Solution` section (anything before the heading is working notes; no heading at all scores 0).
- `scratch/` — copy of the proof solver's scratch directory. The live path is short and opaque (`/c/w/a1b2c3d4`) so the prompt carries no arm, contest, or seed identity. IdeaSearch branches additionally archive the isolated planner workspace as `plan_scratch/`.
- `meta.json` — attempt metadata and totals, provider session UUIDs, process-recovery count, and sequential `termination_reason` (`self_converged` or `token_limit`); written last, so its presence is the completion marker.
- `audit.json` — the judge's verdict (full solution + per-cut scores/notes); `audit_scratch/` — any computations the judge ran while grading.

## Audit (grading)

After generation, `./run.sh audit --arm <slug>` grades every completed attempt — in the same firewalled container as generation, because a judge with tools and internet could fetch official solutions from public archives and its archived scratch would contaminate future runs. The judge (`audit_model`, config-enforced to differ from the solver) is given only the problem statement and the standalone `solution.md` (the hint is not included), grades only the `## Final Solution` section, and returns a structured verdict plus a `note` saying why the solution is valid or exactly what is missing/wrong (schema-enforced, always parses). The scale (`prompts/audit.md`, with calibration examples): **7** complete and rigorous; **6** complete in essence with exactly one small local obvious-fix gap; **5** complete in essence with two or three such gaps; **0** anything else — no other partial credit (a solution missing one of two required bounds scores 0). The judge HAS scratch tools, to audit rather than solve: it may recompute a bound or test a small case in its own opaque scratch dir (archived as `audit_scratch/` beside the attempt), but the prompt forbids filling gaps — a failing check is evidence of error, a passing check never substitutes for written proof. Sequential attempts' budget-cut snapshots are each judged as standalone proofs. The judge prompt is `prompts/audit.md`, editable like the rest. Verdicts land as `audit.json` per seed (resumable marker) and are compiled by scanning the arm's whole results tree into `results/<model>/<arm>/audit.jsonl` (a `--problems`-filtered re-audit can never truncate it). Audits share the token pool and rate-limit rotation.

## Running

```bash
cp .env.example .env   # set CLAUDE_CODE_OAUTH_TOKEN (or OPENROUTER_API_KEY* for vendor/model ids)

# generation (Docker), one arm per invocation
./run.sh run --arm baseline
./run.sh run --arm baseline-parallel
./run.sh run --arm baseline-ideasearch --problems <surviving-ids>
./run.sh run --arm hint
./run.sh run --arm outline
./run.sh run --arm placebo-hint
./run.sh run --arm baseline-sequential --problems <failed-ids>
./run.sh run --arm hint-sequential --problems <failed-ids>
./run.sh run --arm outline-sequential --problems <failed-ids>

# optional filters (combine freely)
./run.sh run --arm baseline --domain combinatorics   # one domain
./run.sh run --arm baseline --problems id1,id2       # explicit subset
./run.sh run --arm baseline --seeds 1                # pilot: seed subset (run stage only)

# model overrides (default: config.json; solver and judge must differ)
./run.sh run --arm baseline --model claude-fable-5 --audit-model claude-opus-4-8
./run.sh audit --arm baseline --model claude-fable-5 --audit-model openai/gpt-5.6-sol

# audit (same container, same filters); compiles audit.jsonl
./run.sh audit --arm baseline
./run.sh audit --arm hint --domain combinatorics
./run.sh audit --arm baseline --seeds 1

# dev only — NO firewall, never for canonical data
python -m src.run --arm baseline
python -m src.audit --arm baseline
```

Concurrency is async (anyio) under a capacity limiter: at most `max_concurrency` (config.json) agent sessions in flight at once. Available credentials are assigned round-robin; keys are not per-request concurrency limits, so one healthy key can serve all eight sessions. Duplicate values under multiple env names are ignored.

Resumable: attempts whose `meta.json` exists are skipped; unfinished single/sequential solves, both IdeaSearch roles, and each audit call resume from private host checkpoints. The checkpoint couples the provider UUID/transcript, scratch, committed phase ledger, controller position, and every `BudgetTracker` counter. A killed in-progress response is retained as discarded audit evidence and the resumed session is asked to emit one complete replacement; all provider-reported prefix and replacement tokens remain charged. Because a hard kill can occur after the provider emits tokens but before its next usage event, `meta.json` flags every process-recovered attempt for a sensitivity analysis rather than claiming unknowable suffix tokens are exact. Phase commits are fsynced before controller advancement, exact duplicate invocations are locked, and run checkpoints are deleted only after the staged result has merged durably into the host results tree. Live quota rejection still rotates credentials within the same UUID and exact in-memory accounting; if every credential is cooling, the attempt waits.

## Type checking

```bash
npx pyright        # standard mode; must pass clean
python -m unittest discover -s tests -v
```
