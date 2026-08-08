# Harness

One harness implementing the paper's arms (top-level README): every attempt is one (arm, problem, seed) run of the Claude Agent SDK, in one of three modes.

- **single** — one solve phase (arms `baseline`, `baseline-parallel`, `placebo-hint`, `hint`, `hint-parallel`, `outline`).
- **sequential** — solve → (critique → revise)× under one shared output-token budget, cut off mid-phase when the budget is spent (arms `baseline-sequential`, `hint-sequential`, `outline-sequential`). Two consecutive critiques containing the exact standalone verdict `NO GENUINE GAP FOUND` stop the trajectory as self-converged; one optimistic critique does not. This is budget-bounded Self-Refine (Madaan et al. 2023); the per-phase cumulative token counts in the logs let the analysis cut the trajectory at 2×/4× for the saturation curve.
- **uniform_strategy** — one shared planner (≤80k) enumerates up to eight distinct strategies, then eight fresh executors receive one assigned strategy each. `baseline-uniform-strategy` repeats this complete 8× bank for seeds 1–3.

## Configuration — `config.json`

Single source of experiment knobs: `model` (solver), `audit_model` (judge; must differ from the solver) — both overridable per invocation with `--model` / `--audit-model` (config values are the defaults; the solver≠judge check applies to the effective pair, and audit's `--model` selects whose results tree to grade), `effort` (fixed `high` per the paper), `unit_output_tokens` (1× = 200k), the Uniform Strategy planner/reserve/branch settings, `max_turns_per_phase` / `audit_max_turns` (per-phase guards; the token budget is the sequential stop), `max_concurrency`, and the arm table (`hint`: none/h1/h2/h3, `mode`, `budget_units`, `seeds`). Models are Anthropic ids (`claude-opus-4-8`), OpenRouter `vendor/model` ids (`openai/gpt-5.5`), or local Codex-subscription aliases (`litellm/gpt-5.5`). The last uses `LITELLM_BASE_URL*` as the round-robin/cooldown pool and `LITELLM_API_KEY` for the local proxy; setup is in `docs/codex-subscription-via-litellm.md`. Results paths use the model id with `/` replaced by `-`. Arm names are the slugs used everywhere — CLI, `results/` paths, and the top-level README arm table. `baseline` uses seeds 1–3 and `baseline-parallel` seeds 4–8, so together they form the 8 parallel-channel seeds without collision.

## Compute ladder (1×/2×/4×/8×)

- **Parallel:** the 8 independent seeds (`baseline` 1–3 + `baseline-parallel` 4–8) are each audited; pass@1/2/4/8 comes from the standard unbiased estimator (Chen et al. 2021) over the per-seed audit scores — never "first n seeds", which is ordering-dependent noise.
- **Sequential:** one 8× trajectory per seed, cut post hoc. Every phase's full write-up is preserved in the logs (phases are separate records; nothing is overwritten), and the harness snapshots `solution_1x.md` / `solution_2x.md` / `solution_4x.md` — the last COMPLETE non-critique write-up emitted before cumulative output tokens crossed each threshold, i.e. exactly what a hard-stopped run at that budget would have been graded on. `meta.json` records which phase each cut came from. The audit stage grades every snapshot as a standalone proof, so each curve point has its own score + note; a budget with no complete write-up scores 0 with an explanatory note. Caveat to state in the paper: the model knows its full 8× budget, so pacing at lower cuts approximates (rather than replays) a true smaller-budget run — the curve's 1× point should come from the real `baseline` arm.
- **Repeats:** baseline, hint, sequential, and Uniform Strategy Search use the arm's three independent seeds (1, 2, 3). A parallel result is one prespecified eight-attempt bank assembled from seeds 1–3 and 4–8; each attempt has its own `seed_<k>/` output tree.
- **Uniform Strategy Search-8:** each seed is one independent bank. The planner receives 5% from each executor allocation, so an 80k shared planner produces `m≤8` strategies and eight fresh executors receive 190k each: `80k + 8×190k = 1.6M = 8×`. Executors are allocated round-robin across strategies, differing by at most one when `m` does not divide eight. Unused tokens do not transfer. Bank audit reports oracle-audited candidate coverage (any valid proof), not selected-proof accuracy.

## Compute budget enforcement

Compute = the attempt's total output-token budget. Three layers:

1. `task_budget` (API-side): the model is told its remaining token budget so it paces itself.
2. Harness cutoff: streamed `message_delta` events carry each API message's real output-token count (partial messages enabled; the per-message `usage` on assistant events is only an initial snapshot) and feed a per-attempt `BudgetTracker` that interrupts after the response crossing the soft threshold; each phase's total is then trued-up from the ResultMessage's exact per-query usage. Working responses are capped at 64k via `CLAUDE_CODE_MAX_OUTPUT_TOKENS` — in Docker the CLI is pinned to npm `2.1.222` via `HARNESS_CLI_PATH`, because the SDK-bundled CLI ignores that cap on Opus and errors long thinking turns at 32k.
3. Strict wrap-up and eligibility: working phases stop around budget − 20k (180k for a 200k attempt; 170k for a 190k executor; 60k for the 80k planner). If still below the hard allocation, the same transcript is resumed for one tool-free turn whose response cap is exactly 20k; the harness, not the model, writes that response to `solution.md`. A phase ending beyond its hard allocation remains fully logged but is ineligible for grading or strategy extraction. Thus accepted artifacts respect the exact 200k/1.6M tiers even though provider billing may include one response-boundary overrun; allocated, realized, and overrun tokens are recorded separately. Given the 64k working-response cap, a complete Uniform bank's theoretical reported-token ceiling is below 1.996M, while no accepted strategies or proofs may use more than its 1.6M allocation.

## Problem data (never committed)

Problems and hints are NOT in this repo — committing them would leak contest identity. They are fetched at runtime from the `notadib/math-contests-2026` dataset with stdlib `urllib`, straight into memory (no `hf_hub`, no disk cache):

- `hard_problems.jsonl` — statements. Only `problem_id`, `statement`, and `domain` (for `--domain` filtering) are kept; contest-identifying metadata is dropped at load and the prompt carries the statement alone.
- `hard_hints.jsonl` — hint ladder source: `placebo` field → **h1** (not authored yet — placebo arms fail fast before spending a token), scalar `hint` field → **h2** (the frozen ≤25-word oracle strategy hint, inserted verbatim). The retired five-tag development file is archived on HuggingFace as `hard_hints-v1.jsonl` and is never fetched by the harness.
- `hard_outlines.jsonl` — audited strategy outlines → **h3** (numbered steps; the `outline` and `outline-sequential` arms).

In Docker, the entrypoint prefetches all three BEFORE the egress firewall closes (HuggingFace stays blocked while agents run — an agent that could fetch the hints file would be contaminated); the loader consumes and deletes the temp copies before any agent spawns, so no trace remains.

## Prompts — `prompts/*.md`

All prompts are editable markdown files, including the three `uniform_strategy_*.md` templates. Placeholders use `{{name}}` and are filled by literal replacement (LaTeX braces can never break rendering); an unfilled placeholder fails loud. A hint arm fails fast before spending tokens if a selected problem lacks its hint.

## Isolation (contamination control)

- **Docker is the boundary.** `./run.sh run --arm <slug>` builds a throwaway container (`--rm`) that holds only the harness, problems, and prompts — no reference solutions, nothing else from the machine.
- **Prior results never enter a generation container.** The run stage mounts a staging dir pre-seeded with only `meta.json` completion markers (resume still works), then merges only newly completed attempts containing both `solution.md` and the last-written `meta.json`. Marker-only resume inputs and partial writes are discarded rather than re-merged, preventing a long-running arm from resurrecting stale results archived while it was active. An agent can never read past arms' solutions or hint-carrying logs, even by exploring the filesystem. The audit stage mounts the real tree (the judge must read solutions).
- **Network:** the entrypoint installs a default-DROP egress firewall and allows only the active provider: Anthropic, OpenRouter, or the explicitly configured local LiteLLM sidecars. It self-tests the selected route and confirms that a non-allowlisted host is unreachable before any token is spent.
- **Tools:** `agent_settings.json` (passed via the SDK's `--settings`) denies `WebSearch`, `WebFetch`, and network/download Bash commands (`curl`, `wget`, `git`, `pip install`, …); `disallowed_tools` strips the web, subagent/fleet, publishing, and background-scheduling built-ins. User/project settings are excluded by passing `--setting-sources` explicitly empty (the SDK silently drops a falsy `setting_sources=[]`).
- **Live transcripts:** each solver, planner, executor, and judge call has an isolated provider UUID and transcript in a host-persisted opaque workspace (for example `/c/w/a1b2c3d4`); concurrent seeds never share one, runtime state is excluded from archived scratch, and Docker mounts only the current stage/model/arm checkpoint namespace so other interventions remain invisible.
- Every tool call is captured untruncated in the logs, so a run can be proven clean after the fact.

## Outputs — `results/<model>/<arm>/<problem_id>/seed_<k>/`

- `logs.jsonl.zst` — one JSON line per phase: prompt, full response text, every tool call (full input/result), raw provider usage, per-phase and cumulative output tokens, turns, duration, SDK-reported cost estimate, stop reason, and any credential reconnects.
- `solution.md` — the graded artifact: the last COMPLETE proof-producing phase (normally the wrap-up), same convention as the budget cuts so the full-budget point can never score below a lower cut by truncation artifact. The judge grades ONLY its `## Final Solution` section (anything before the heading is working notes; no heading at all scores 0).
- `scratch/` — copy of the proof solver's scratch directory. The live path is short and opaque (`/c/w/a1b2c3d4`) so the prompt carries no arm, contest, or seed identity. Uniform Strategy banks store the shared planner log and `strategies.json` at the seed root, archive its workspace as `plan_scratch/`, and store each independently audited executor under `branch_<k>/`.
- `meta.json` — attempt metadata, standard provider-usage totals, provider session UUIDs, process-recovery count, and sequential `termination_reason` (`self_converged` or `token_limit`); written last, so its presence is the completion marker.
- `audit.json` — the judge's verdict (full solution + per-cut scores/notes); `audit_scratch/` — any computations the judge ran while grading.

## Audit (grading)

After generation, `./run.sh audit --arm <slug>` grades every completed attempt — in the same firewalled container as generation, because a judge with tools and internet could fetch official solutions from public archives and its archived scratch would contaminate future runs. The judge (`audit_model`, config-enforced to differ from the solver) is given only the problem statement and the standalone `solution.md` (the hint is not included), grades only the `## Final Solution` section, and returns a structured verdict plus a `note` saying why the solution is valid or exactly what is missing/wrong (schema-enforced, always parses). The scale (`prompts/audit.md`, with calibration examples): **7** complete and rigorous; **6** complete in essence with exactly one small local obvious-fix gap; **5** complete in essence with two or three such gaps; **0** anything else — no other partial credit (a solution missing one of two required bounds scores 0). The judge HAS scratch tools, to audit rather than solve: it may recompute a bound or test a small case in its own opaque scratch dir (archived as `audit_scratch/` beside the attempt), but the prompt forbids filling gaps — a failing check is evidence of error, a passing check never substitutes for written proof. Sequential attempts' budget-cut snapshots are each judged as standalone proofs. The judge prompt is `prompts/audit.md`, editable like the rest. Verdicts land as `audit.json` per seed (resumable marker) and are compiled by scanning the arm's whole results tree into `results/<model>/<arm>/audit.jsonl` (a `--problems`-filtered re-audit can never truncate it). Audits share the token pool and rate-limit rotation.

## Running

### GPT-5.5 through a Codex subscription

There is one user-facing pool command: `scripts/codex_pool.sh`.
The Python files under `scripts/codex_pool_internal/` are implementation
helpers; do not invoke them directly. First-time setup for the Codex account
currently logged in on this machine is:

```bash
# One-time: copy the current Codex login into slot 1's private Docker volume.
./scripts/codex_pool.sh add 1 ~/.codex/auth.json

# Start the sidecar and verify a direct request plus Agent SDK tool use.
./scripts/codex_pool.sh start 1
./scripts/codex_pool.sh verify 1

# Copy these LITELLM_* lines into the repository's .env once.
test -f .env || cp .env.example .env
./scripts/codex_pool.sh env 1
```

After a reboot, only `start 1` is needed; the copied OAuth remains in its Docker
volume. `add` makes the sidecar's private runtime copy—it does not
modify `~/.codex/auth.json`. For multiple subscriptions, private auth-file
layout, revocation, and troubleshooting, see
`docs/codex-subscription-via-litellm.md`.

With ChatGPT login, `total_cost_usd` is the SDK/LiteLLM API-equivalent estimate,
not an additional subscription charge. Canonical compute accounting uses the
provider-reported output-token totals; actual plan consumption is governed by
ChatGPT credits and usage limits and is visible in the Codex usage dashboard.

```bash
test -f .env || cp .env.example .env   # configure the active provider

# generation (Docker), one arm per invocation
./run.sh run --arm baseline
./run.sh run --arm baseline-parallel
./run.sh run --arm baseline-uniform-strategy --problems <surviving-ids>
./run.sh run --arm hint
./run.sh run --arm hint-parallel --problems <full-search-hint-failed-ids>
./run.sh run --arm outline
./run.sh run --arm placebo-hint
./run.sh run --arm baseline-sequential --problems <failed-ids>
./run.sh run --arm hint-sequential --problems <full-search-hint-failed-ids>
./run.sh run --arm outline-sequential --problems <failed-ids>

# optional filters (combine freely)
./run.sh run --arm baseline --domain combinatorics   # one domain
./run.sh run --arm baseline --problems id1,id2       # explicit subset
./run.sh run --arm baseline --seeds 1                # pilot: seed subset (run stage only)

# model overrides (default: config.json; solver and judge must differ)
./run.sh run --arm baseline --model claude-fable-5 --audit-model claude-opus-4-8
./run.sh run --arm baseline --model litellm/gpt-5.5 --audit-model claude-opus-4-8
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

Resumable: attempts whose `meta.json` exists are skipped; unfinished single/sequential solves, the Uniform Strategy planner and each executor, and every audit call resume from private host checkpoints. The checkpoint couples the provider UUID/transcript, scratch, committed phase ledger, controller position, and every `BudgetTracker` counter. A killed in-progress response is retained as discarded audit evidence and the resumed session is asked to emit one complete replacement; all provider-reported prefix and replacement tokens remain charged. Because a hard kill can occur after the provider emits tokens but before its next usage event, `meta.json` flags every process-recovered attempt for a sensitivity analysis rather than claiming unknowable suffix tokens are exact. Phase commits are fsynced before controller advancement, exact duplicate invocations are locked, and run checkpoints are deleted only after the staged result has merged durably into the host results tree. Live quota rejection still rotates credentials within the same UUID and exact in-memory accounting; if every credential is cooling, the attempt waits.

## Type checking

```bash
npx pyright        # standard mode; must pass clean
python -m unittest discover -s tests -v
```
