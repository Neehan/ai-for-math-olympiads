# Harness

One harness implementing the paper's arms (top-level README): every top-level attempt is one (arm, problem, seed) run of the Claude Agent SDK.

- **single** — one solve phase (arms `baseline`, `placebo-hint`, `hint`, `outline`).
- **sequential** — solve → (critique → revise)× under one shared output-token budget, cut off mid-phase when the budget is spent (arms `baseline-sequential`, `hint-sequential`, `outline-sequential`). For an explicitly supplied problem set, `late-baseline-sequential` creates a fresh exact 3× native session, retains its transcript and scratch state, and continues a no-hint fork for one 1× block. Run second on the same problem set, `late-hint-sequential` forks that same retained state and differs only by the ordinary h2 block. All requested seeds are run, including prefixes already solved in the fresh replication; outcome conditioning is post hoc. Missing native sources are skipped, never reconstructed from text. Ordinary trajectories stop after the registered no-gap rule; the late prefix deliberately runs to its fixed 3× boundary before forking.
- **parallel** — three eight-run banks (`baseline-parallel`, seeds 1–3): within each bank, `run_01`–`run_08` are eight fresh IID 1× attempts. Each bank is exactly 8× and reports `c/8` plus pass@k; the three bank seeds also support the same 0/3–3/3 reliability summaries used for Sequential.
- **uniform_strategy** — the internal mode for Uniform-C-8, a proof-domain adaptation of coarse-grained TTS-Uniform without entropy filtering. One shared extractor (≤80k) consolidates up to eight semantically distinct whole-proof candidate strategies, then eight fresh executors receive one assigned strategy each. The stable arm slug is `baseline-uniform-strategy`.
- **uniform_strategy_only / uniform_compress** — freeze the realized seed-1 Uniform-C planner bank without executors, then reproducibly sample three raw entries without replacement or outcome filtering. A binary audit labels whether each raw proposal follows the same distinctive, load-bearing route as the frozen oracle strategy; complete execution is not required. GPT-5.6-sol compresses only the sampled entries into 18--25-word proof sketches; compressed sketches inherit their exact raw proposal's frozen label and have no separate audit stage. The target oracle is never shown to the compressor and is copied verbatim from the frozen <=25-word `hard_hints` record. Compression is tool-free and preserves generated routes and errors. The completed full Uniform-C planner artifact is copied exactly rather than regenerated. Missing banks and banks with fewer than three proposals are skipped and reported. Both arms accept only seed 1.
- **selection / selection_no_problem** — exploratory selector controls. Seeds 1–3 all rank the same frozen proposal-seed-1 four-item set (three compressed proposals plus the exact frozen oracle sketch) under the standard 1× protocol: a 200k output-token cap, offline scratch tools through the 180k working boundary, and one tool-free final-ranking turn within the reserved 20k. Candidate order is randomized independently by selector seed. `selection` includes the problem statement; `selection-no-problem` withholds only that statement to measure provenance leakage. Both arms use the same candidate order within a selector seed. All store the full ranking, exact oracle rank, and whether the top candidate matches the frozen route; an over-cap or missing ranking is a terminal `no_decision`. Expert viability review supplies the analysis label, with single-viable pools analyzed separately.

## Configuration — `config.json`

Single source of experiment knobs: `model` (solver), `audit_model` (judge; must differ from the solver) — both overridable per invocation with `--model` / `--audit-model` (config values are the defaults; the solver≠judge check applies to the effective pair, and audit's `--model` selects whose results tree to grade), `effort` (fixed `high` per the paper), `unit_output_tokens` (1× = 200k), the standard 20k wrap-up reserve, the Uniform Strategy planner/reserve/branch settings, `max_turns_per_phase` / `audit_max_turns` (per-phase guards; the token budget is the sequential stop), `max_concurrency` (operationally overridable with `--max-concurrency`), and the arm table (`hint`: none/h1/h2/h3, `mode`, `budget_units`, `seeds`). Models are Anthropic ids (`claude-opus-4-8`), OpenRouter `vendor/model` ids (`openai/gpt-5.5`, `deepseek/deepseek-v4-flash-0731`), local Codex-subscription aliases (`litellm/gpt-5.4`, `litellm/gpt-5.4-mini`, `litellm/gpt-5.5`, `litellm/gpt-5.6-luna`, `litellm/gpt-5.6-sol`), or native Anthropic-compatible vLLM ids (`vllm/qed-nano`, `vllm/muse-glimmer`). DeepSeek V4 Flash is restricted to Relace, Baidu Qianfan, StreamLake, and DeepInfra, sorted by live throughput and capped at $0.08/M input and $0.18/M output. Local routes use their `*_BASE_URL*` values as the round-robin/cooldown pool and a shared `*_API_KEY`; setup is in `docs/codex-subscription-via-litellm.md` and `docs/qed-nano-vllm.md`. Results paths use the model id with `/` replaced by `-`. Arm names are the slugs used everywhere — CLI, `results/` paths, and the top-level README arm table. Single, Sequential, and Parallel reliability arms use seeds 1–3; Uniform-C remains one top-level planner bank with eight dependent executors.

Meta's contributor-tier `muse-spark-1.2-contributor` model uses its native Anthropic-compatible endpoint and `META_API_KEY`; setup and invocation are in `docs/meta-muse-spark.md`.

## Compute ladder (1×/2×/4×/8×)

- **Parallel:** each of three prespecified bank seeds contains eight fresh IID 1× attempts. Auditing reports valid-proof coverage `c/8` and the standard unbiased pass@k estimate at `k∈{1,2,4,8}` for every bank. The arm is not a budget-prefix curve; bank-level outcomes additionally support 0/3–3/3 reliability summaries.
- **Sequential:** one 8× trajectory per seed, cut post hoc. Every phase's full write-up is preserved in the logs (phases are separate records; nothing is overwritten), and the harness snapshots `solution_1x.md` / `solution_2x.md` / `solution_4x.md` — the last COMPLETE non-critique write-up emitted before cumulative output tokens crossed each threshold, i.e. exactly what a hard-stopped run at that budget would have been graded on. `meta.json` records which phase each cut came from. The audit stage grades each unique snapshot as a standalone proof and reuses the verdict for byte-identical snapshots, so audit noise cannot create movement without a changed proof; a budget with no complete write-up scores 0 with an explanatory note. Caveat to state in the paper: the model knows its full 8× budget, so pacing at lower cuts approximates (rather than replays) a true smaller-budget run — the curve's 1× point should come from the real `baseline` arm.
- **Repeats:** single, Sequential, and Parallel arms use seeds 1–3. Each Parallel seed is a separate eight-run bank under `seed_<k>/run_01` through `run_08`. Uniform-C retains one top-level bank seed.
- **Uniform-C-8:** this proof-domain adaptation uses the coarse-grained TTS-Uniform extraction and uniform-allocation modules without answer-entropy filtering or majority-vote aggregation. The extractor receives 5% from each executor allocation, so an 80k shared call produces `m≤8` semantically distinct whole-proof candidate strategies and eight fresh executors receive 190k each: `80k + 8×190k = 1.6M = 8×`. Executors are allocated round-robin across strategies; each receives either `floor(8/m)` or `ceil(8/m)` assignments, differing by at most one. Every executor receives its assigned output through the same neutral strategy wrapper as the oracle-sketch arm: the candidate is proposed, not certified correct, and the executor must check it while proving every step. Generated strategies are not externally verified or selected before execution. Unused tokens do not transfer. Audit raw planner artifacts and executor artifacts independently under the same frozen access rule; a strategy observed only in an executor is eventual arm-level access, not explicit planner proposal. The dependent executor bank is not pass@`k`.

## Compute budget enforcement

Compute = the attempt's total output-token budget. Three layers:

1. `task_budget` (API-side, where supported): the model is told its remaining token budget so it paces itself. Meta's Anthropic-compatible adapter rejects this Claude extension, so Muse receives the same allocation in the task prompt while the local cutoff below remains authoritative.
2. Harness cutoff: streamed `message_delta` events carry each API message's real output-token count (partial messages enabled; the per-message `usage` on assistant events is only an initial snapshot) and feed a per-attempt `BudgetTracker` that interrupts after the response crossing the soft threshold; each phase's total is then trued-up from the ResultMessage's exact per-query usage. Working responses are capped at 64k via `CLAUDE_CODE_MAX_OUTPUT_TOKENS` — in Docker the CLI is pinned to npm `2.1.222` via `HARNESS_CLI_PATH`, because the SDK-bundled CLI ignores that cap on Opus and errors long thinking turns at 32k. Anthropic, LiteLLM, and OpenRouter routes use `--autocompact 900k`; `vllm/*` uses `200k` to leave roughly 62k headroom in a 262k context.
3. Strict wrap-up and eligibility: ordinary working phases stop around budget − 20k (180k for a 200k attempt and 170k for a 190k executor). The 80k Uniform planner uses a larger 40k consolidation reserve, so exploration stops around 40k. If still below the hard allocation, the same transcript is resumed for one tool-free turn capped at the configured reserve; the harness, not the model, writes that response to `solution.md`. A phase ending beyond its hard allocation remains fully logged but is ineligible for grading or strategy extraction. Thus accepted artifacts respect the exact 200k/1.6M tiers even though provider billing may include one response-boundary overrun; allocated, realized, and overrun tokens are recorded separately.

## Problem data (never committed)

Problems and hints are NOT in this repo — committing them would leak contest identity. They are fetched at runtime from the `notadib/math-contests-2026` Hugging Face repository with stdlib `urllib`, straight into memory (no `hf_hub`, no disk cache). The default `--dataset math-contests-2026` uses the four `hard_*` files below; `--dataset imobench` selects the corresponding `imobench_problems.jsonl`, `imobench_hints.jsonl`, `imobench_outlines.jsonl`, and `imobench_solutions.jsonl` files.

- `hard_problems.jsonl` — statements. Only `problem_id`, `statement`, and `domain` (for `--domain` filtering) are kept; contest-identifying metadata is dropped at load and the prompt carries the statement alone.
- `hard_hints.jsonl` — hint ladder source: scalar `hint` field → **h2** (the frozen ≤25-word oracle strategy hint, inserted verbatim). The loader derives **h1** deterministically: within each domain, sort `problem_id` lexicographically and give each problem the next problem's h2 hint, wrapping cyclically. This preserves the within-domain hint multiset while preventing self-assignment. The retired five-tag development file is archived on HuggingFace as `hard_hints-v1.jsonl` and is never fetched by the harness.
- `hard_outlines.jsonl` — audited strategy outlines → **h3** (numbered steps; the `outline` and `outline-sequential` arms).
- `hard_solutions.jsonl` — frozen human-verified references used by correctness and state annotation. Both receive the fixed outline-matching `reference_solutions[0]`, which alone carries `route_id: "hard_hint"`; correctness grading must still accept valid alternative routes.
- `hard_hint_selection.jsonl` — one frozen proposal-seed-1 candidate set per `source_model` and `problem_id`. Each record contains the exact frozen oracle sketch and exactly three compressed generated strategies with a boolean `oracle_strategy_match` label. The harness rejects any proposal seed other than 1, oracle drift, missing labels, duplicate identities, or any count other than three.

In Docker, the entrypoint prefetches data BEFORE the egress firewall closes (HuggingFace stays blocked while agents run). Generation never receives reference solutions. Audit containers receive the frozen full-solution file; each loader consumes and deletes its temp copy before any agent spawns.

## Prompts — `prompts/*.md`

All prompts are editable markdown files. Uniform planning has dedicated plan and wrap-up templates, while Uniform executors deliberately reuse the ordinary task and hint templates. Placeholders use `{{name}}` and are filled by literal replacement (LaTeX braces can never break rendering); an unfilled placeholder fails loud. A hint arm fails fast before spending tokens if a selected problem lacks its hint.

## Isolation (contamination control)

- **Docker is the boundary.** `./run.sh run --arm <slug>` builds a throwaway container (`--rm`) that holds only the harness, problems, and prompts — no reference solutions, nothing else from the machine.
- **Prior proof results never enter an ordinary generation container.** The run stage mounts a staging dir pre-seeded with only `meta.json` completion markers (resume still works), then merges only durable pairs containing both `solution.md` and the last-written `meta.json`. The matched late control retains its own exact native 3× transcript and scratch snapshot in the shared private checkpoint namespace; the treatment forks only its matching problem/seed snapshot and skips missing sources. Selection receives only empty, path-level markers for its own completed attempts, so prior rankings, oracle positions, and candidate labels are absent. The compression arm additionally receives the exact planner-only `strategies.json` dependency and its non-proof manifest; its worker is tool-free. For an interrupted bank, completed `run_<kk>` children are retained while marker-only and partial children are discarded. This prevents stale-result resurrection without losing paid bank members. The audit stage mounts the real tree (the judge must read solutions).
- **Network:** the entrypoint installs a default-DROP egress firewall and allows only the active provider: Anthropic, OpenRouter, or the explicitly configured local LiteLLM sidecars. It self-tests the selected route and confirms that a non-allowlisted host is unreachable before any token is spent.
- **Tools:** Anthropic, LiteLLM, and OpenRouter use `agent_settings.json`; `vllm/*` uses the reduced `agent_settings_small.json`. Both are passed via the SDK's `--settings` and deny network/download Bash commands. The VLLM profile additionally strips its named Claude built-ins through `disallowed_tools`; the common policy strips web, subagent/fleet, publishing, and background-scheduling tools. User/project settings are excluded by passing `--setting-sources` explicitly empty (the SDK silently drops a falsy `setting_sources=[]`).
- **Live transcripts:** each solver, planner, executor, and judge call has an isolated provider UUID and transcript in a host-persisted opaque workspace (for example `/c/w/a1b2c3d4`); concurrent seeds never share one and runtime state is excluded from archived scratch. The two late arms alone share a private checkpoint namespace so treatment can fork the exact retained control prefix; every branch receives a new provider UUID.
- Every tool call is captured untruncated in the logs, so a run can be proven clean after the fact.

## Outputs — `results/<model>/<arm>/<problem_id>/seed_<k>/`

- `logs.jsonl.zst` — one JSON line per phase: prompt, full response text, every tool call (full input/result), raw provider usage, per-phase and cumulative output tokens, turns, duration, SDK-reported cost estimate, stop reason, and any credential reconnects.
- `solution.md` — the graded artifact: the last COMPLETE proof-producing phase (normally the wrap-up), same convention as the budget cuts so the full-budget point can never score below a lower cut by truncation artifact. The judge grades ONLY its `## Final Solution` section (anything before the heading is working notes; no heading at all scores 0).
- `scratch/` — copy of the proof solver's scratch directory. The live path is short and opaque (`/c/w/a1b2c3d4`) so the prompt carries no arm, contest, or seed identity. Bank members live under `run_01`–`run_08`. Parallel stores eight fresh attempts; Uniform stores eight fresh executors plus the shared planner log, `strategies.json`, and `plan_scratch/` at the seed root.
- `meta.json` — attempt metadata, standard provider-usage totals, provider session UUIDs, recovery policy/events, token-accounting status, and the sequential round count, stopping policy, and `termination_reason` (`self_converged` or `token_limit`); written last, so its presence is the completion marker.
- `audit.json` — the judge's verdict (full solution + per-cut scores/notes); `audit_scratch/` — any computations the judge ran while grading.

## Audit (grading)

After generation, `./run.sh audit --arm <slug>` grades every completed attempt in a firewalled container. The judge (`audit_model`, config-enforced to differ from the solver) receives the problem statement, the fixed verified index-0 reference solution, and the standalone `solution.md`; it does not receive the hint or solver scratch. The prompt explicitly treats the reference as one example, accepts valid alternative routes, and forbids importing reasoning missing from the submission. It grades only the `## Final Solution` section and returns a structured score and concise note. The near-binary scale is: **7** complete and rigorous; **6/5** correct in substance with exactly one/two minor local defects; **0** anything else. The judge may use scratch tools to expose an error, but a passing computation never supplies missing proof. Sequential budget-cut snapshots are judged standalone. By default these are `1x/2x/4x/8x`; `--all-checkpoints` recovers and audits every integer `1x,...,8x` snapshot from the immutable phase log. Frozen verdicts and step annotations for byte-identical existing snapshots are reused. Verdicts land as `audit.json` per seed and compile into `results/<model>/<arm>/audit.jsonl`; audits share the token pool and rate-limit rotation.

For `baseline-sequential`, `hint-sequential`, both matched late arms, `baseline-parallel`, and `baseline-uniform-strategy`, the same `./run.sh audit` command then reuses those proof verdicts and writes `state_audit.json` beside every audited proof artifact, plus an arm-level `state_audit.jsonl`. Standalone fixed-compute proof arms stop after correctness grading. Sequential arms annotate every budget-cut snapshot; Parallel and Uniform banks retain one record per executor rather than assigning a state to the non-proof bank wrapper. A passing proof or complete 3/3 mechanism recognition enters acquired state `S`, which is carried forward; missing solution text before acquisition receives `state: null`; only nonempty score-below-5 artifacts invoke the tool-free outline annotator. Its prompt contains the problem, three-step outline, explicitly indexed matching reference solution, and submitted solution. For each outline step it returns one `present` boolean and one short reason saying whether the submission explicitly recognizes that ingredient and its role, regardless of whether the attempted proof is correct. Before acquisition, the harness compares the recognized-step count with the preceding observed artifact, starting from zero: an incomplete increase gives `P`; a flat or decreasing incomplete count gives `U`; missing artifacts do not change the comparison point. Problem statements are cross-checked across data files, and identical snapshots within one trajectory reuse one annotation. The planner-only arm has no proof to grade: its separate tool-free audit receives the problem, frozen oracle sketch, matching reference solution, and raw proposal, then returns a binary oracle-strategy match with a short reason. The compression arm has no audit stage.

## Running

### GPT-5.4 through a Codex subscription

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
./run.sh run --arm baseline-parallel --model litellm/gpt-5.4 --seeds 1,2,3
./run.sh run --arm baseline-uniform-strategy --problems <surviving-ids>
./run.sh run --arm baseline-uniform-strategy-only --model litellm/gpt-5.4 --seeds 1 --problems <ids>
./run.sh audit --arm baseline-uniform-strategy-only --model litellm/gpt-5.4 --seeds 1
./run.sh run --arm baseline-uniform-compress --model litellm/gpt-5.4 --worker-model litellm/gpt-5.6-sol --seeds 1
python scripts/build_selection_dataset.py
# Before any ranking run, review compression fidelity blind to ranking outcomes,
# then upload the frozen file.
./run.sh run --arm selection --model litellm/gpt-5.4 --problems <ids>
./run.sh run --arm selection-no-problem --model litellm/gpt-5.4 --problems <ids>
./run.sh run --arm hint
./run.sh run --arm outline
./run.sh run --arm placebo-hint
./run.sh run --arm baseline-sequential --problems <failed-ids>
./run.sh run --arm hint-sequential --problems <full-search-hint-failed-ids>
./run.sh run --arm late-baseline-sequential --model litellm/gpt-5.4 --seeds 1,2,3 --problems <ids>
./run.sh run --arm late-hint-sequential --model litellm/gpt-5.4 --seeds 1,2,3 --problems <same-ids>
./run.sh run --arm outline-sequential --problems <failed-ids>

# optional filters (combine freely)
./run.sh run --arm baseline --domain combinatorics   # one domain
./run.sh run --arm baseline --problems id1,id2       # explicit subset
./run.sh run --arm baseline --seeds 1                # pilot: seed subset (run stage only)
./run.sh run --dataset imobench --arm baseline       # writes results-imobench/

# model overrides (default: config.json; solver and judge must differ)
./run.sh run --arm baseline --model claude-fable-5 --audit-model claude-opus-4-8
./run.sh run --arm baseline --model litellm/gpt-5.4 --audit-model claude-opus-4-8
./run.sh run --arm baseline --model vllm/qed-nano --domain combinatorics --seeds 1
./run.sh audit --arm baseline --model claude-fable-5 --audit-model openai/gpt-5.6-sol

# audit (same container, same filters); compiles audit.jsonl
./run.sh audit --arm baseline
./run.sh audit --arm hint --domain combinatorics
./run.sh audit --arm baseline --seeds 1
./run.sh audit --arm baseline --seeds 1             # correctness only
./run.sh audit --arm baseline-sequential --seeds 1  # correctness + headline checkpoint states
./run.sh audit --arm baseline-sequential --model claude-opus-4-8 --all-checkpoints
./run.sh audit --arm late-baseline-sequential --model litellm/gpt-5.4 --all-checkpoints  # includes 3x
./run.sh audit --arm late-hint-sequential --model litellm/gpt-5.4      # ordinary 1x/2x/4x schedule
./run.sh audit --dataset imobench --arm baseline     # correctness in results-imobench/

# dev only — NO firewall, never for canonical data
python -m src.run --arm baseline
python -m src.audit --arm baseline
```

Concurrency is async (anyio) under capacity limiters: at most the effective `max_concurrency` agent sessions are admitted at once. The checked-in default is 8; `--max-concurrency 16` overrides it for one invocation without changing checkpoint identity. Parallel-8 banks retain their fixed eight-way internal fan-out, and the scheduler admits `floor(max_concurrency / 8)` complete banks concurrently. Uniform-C retains its original one-bank-at-a-time schedule. Available credentials are assigned round-robin; keys are not per-request concurrency limits, so one healthy key can serve concurrent sessions. Duplicate values under multiple env names are ignored.

Resumable: attempts whose `meta.json` exists are skipped; unfinished single/sequential solves, every Parallel/Uniform bank member, the Uniform planner, and every audit call resume from private host checkpoints. The checkpoint couples the provider UUID/transcript, scratch, committed phase ledger, controller position, stable provider message IDs, and every `BudgetTracker` counter. A rejected response's valid provider-metered output remains eligible; rejection/error prose, replayed IDs, zero-turn replay, and unmetered output can never certify completion. A killed in-progress response is retained as discarded audit evidence and the resumed session is asked to emit one complete replacement; all reported prefix and replacement tokens remain charged. Live quota rejection rotates credentials within the same UUID. Recognized transient stream failures resume that UUID up to six times with capped exponential backoff; proxy-level request replay stays disabled, and persistent or non-transient errors leave the attempt incomplete. Every reconnect is logged. The experimental budget counts transcript-visible eligible output: streamed usage plus completed per-query Result usage. Provider-side work that returned no usable transcript output is infrastructure overhead and is excluded. Recovered attempts are marked `recovered_eligible_output_accounted`; their exact process and transport provenance remains in `process_resume_count` and `session_reconnects`. Phase commits are fsynced before controller advancement, exact duplicate invocations are locked, and run checkpoints are deleted only after the staged result has merged durably into the host results tree. If every credential is cooling, the attempt waits.

## Type checking

```bash
npx pyright        # standard mode; must pass clean
python -m unittest discover -s tests -v
```
