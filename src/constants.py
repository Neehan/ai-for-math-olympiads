"""Single source of truth for paths, filenames, tool lists, and env var names."""

from pathlib import Path

from claude_agent_sdk import PermissionMode
from dotenv import load_dotenv

# Load .env (auth token etc.) as soon as constants is imported, so every
# entrypoint and env-reading path sees it. .env is git-ignored.
load_dotenv()

# --- Paths ---------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_PATH: Path = REPO_ROOT / "config.json"
PROMPTS_DIR: Path = REPO_ROOT / "prompts"
AGENT_SETTINGS_PATH: Path = REPO_ROOT / "agent_settings.json"
RESULTS_ROOT: Path = REPO_ROOT / "results"
# Per-attempt Claude transcript/config store. Keeping it under that attempt's
# opaque scratch dir prevents concurrent conversations from sharing ~/.claude.
SESSION_STATE_SUBDIR: str = ".claude-runtime"
CHECKPOINT_ROOT_ENV: str = "HARNESS_CHECKPOINT_ROOT"
CHECKPOINT_ROOT_DEFAULT: Path = REPO_ROOT / ".session-checkpoints" / "runtime"
DEFER_CHECKPOINT_CLEANUP_ENV: str = "HARNESS_DEFER_CHECKPOINT_CLEANUP"

# --- Per-seed output filenames -------------------------------------------
LOGS_FILENAME: str = "logs.jsonl.zst"
SOLUTION_FILENAME: str = "solution.md"
# Sequential arms also snapshot the solution at each lower budget cut
# (e.g. solution_2x.md): the last complete write-up emitted before cumulative
# output tokens crossed that budget — what a hard-stopped run would grade.
SOLUTION_CUT_FILENAME_FORMAT: str = "solution_{multiplier}x.md"
META_FILENAME: str = "meta.json"
SCRATCH_SUBDIR: str = "scratch"
PLAN_SCRATCH_SUBDIR: str = "plan_scratch"
UNIFORM_STRATEGIES_FILENAME: str = "strategies.json"
BANK_RUN_DIR_FORMAT: str = "run_{run:02d}"
RUN_REFERENCE_FILENAME: str = "reference.json"
# Versioned bank marker: old baseline-plus-seven layouts must never silently
# certify a fresh-IID-eight Parallel bank after the protocol change.
PARALLEL_BANK_PROTOCOL: str = "fresh_iid_8_v1"
ZSTD_LEVEL: int = 9

# --- Audit ----------------------------------------------------------------
# Per-seed judge verdict (completion marker for resumable audits) and the
# per-arm compiled file: one JSON line per (problem, seed) with score + note.
SEED_AUDIT_FILENAME: str = "audit.json"
ARM_AUDIT_FILENAME: str = "audit.jsonl"
# Grading scale (prompts/audit.md): 7 complete, 6/5 small fixable gap, 0 else.
AUDIT_SCORES: tuple[int, ...] = (0, 5, 6, 7)
AUDIT_SCORE_INVALID: int = 0
# Judge scratch copies are archived beside the attempt they graded.
AUDIT_SCRATCH_SUBDIR: str = "audit_scratch"

# --- Prompt template filenames (in PROMPTS_DIR) --------------------------
SYSTEM_PROMPT_FILE: str = "system.md"
TASK_PROMPT_FILE: str = "task.md"
HINT_PROMPT_FILE: str = "hint.md"
CRITIQUE_PROMPT_FILE: str = "critique.md"
REVISE_PROMPT_FILE: str = "revise.md"
WRAP_UP_PROMPT_FILE: str = "wrap_up.md"
UNIFORM_STRATEGY_PLAN_PROMPT_FILE: str = "uniform_strategy_plan.md"
UNIFORM_STRATEGY_PLAN_WRAP_UP_PROMPT_FILE: str = "uniform_strategy_plan_wrap_up.md"
UNIFORM_STRATEGY_EXECUTE_PROMPT_FILE: str = "uniform_strategy_execute.md"
AUDIT_PROMPT_FILE: str = "audit.md"

# --- Problem/hint data sources -------------------------------------------
# Never committed (contest identity); fetched straight into memory, or from
# the entrypoint's pre-firewall temp copies, which the loader deletes on read.
_DATASET_BASE: str = (
    "https://huggingface.co/datasets/notadib/math-contests-2026/resolve/main"
)
PROBLEMS_URL: str = f"{_DATASET_BASE}/hard_problems.jsonl"
HINTS_URL: str = f"{_DATASET_BASE}/hard_hints.jsonl"
OUTLINES_URL: str = f"{_DATASET_BASE}/hard_outlines.jsonl"
PROBLEMS_FILE_ENV: str = "PROBLEMS_FILE"
HINTS_FILE_ENV: str = "HINTS_FILE"
OUTLINES_FILE_ENV: str = "OUTLINES_FILE"
FETCH_TIMEOUT_SECONDS: int = 60

# --- Arm vocabulary -------------------------------------------------------
# Hint ladder: h1 placebo (unauthored — fail fast), h2 frozen one-sentence
# strategy hint (the hint arm), h3 numbered strategy outline (the outline arm).
HINT_NONE: str = "none"
HINT_H1: str = "h1"
HINT_H2: str = "h2"
HINT_H3: str = "h3"
HINT_KINDS: frozenset[str] = frozenset({HINT_NONE, HINT_H1, HINT_H2, HINT_H3})
MODE_SINGLE: str = "single"
MODE_SEQUENTIAL: str = "sequential"
MODE_PARALLEL: str = "parallel"
MODE_UNIFORM_STRATEGY: str = "uniform_strategy"
MODES: frozenset[str] = frozenset(
    {MODE_SINGLE, MODE_SEQUENTIAL, MODE_PARALLEL, MODE_UNIFORM_STRATEGY}
)

# --- Phase labels ---------------------------------------------------------
PHASE_SOLVE: str = "solve"
PHASE_CRITIQUE: str = "critique"
PHASE_REVISE: str = "revise"
PHASE_WRAP_UP: str = "wrap_up"
PHASE_PLAN: str = "plan"
PHASE_PLAN_WRAP_UP: str = "plan_wrap_up"

# Sequential self-refinement stops only after the model reaches the prompt's
# exact no-gap verdict twice in a row. One optimistic critique is
# not enough to terminate a trajectory.
NO_GENUINE_GAP_MARKER: str = "NO GENUINE GAP FOUND"
SEQUENTIAL_NO_GAP_STREAK_TO_STOP: int = 2

# Mechanical recovery instruction used only after the provider rejects a
# live request for quota/spend reasons. Keeping it fixed makes reconnects an
# auditable transport intervention rather than a problem-specific hint.
SESSION_RECOVERY_PROMPT: str = "Continue from exactly where you were interrupted."
# Used only when the local harness process died mid-response.  Unlike a live
# credential handoff, an abruptly terminated CLI may not have committed the
# partial assistant turn to its transcript, so continuation could silently
# omit material.  Re-emitting the complete response gives a gradeable phase;
# the discarded prefix remains in the private checkpoint/audit log and its
# tokens remain charged to the same BudgetTracker.
PROCESS_RECOVERY_PROMPT: str = (
    "The local harness restarted while submitting or answering a request. "
    "Produce one complete response to the pending request reproduced below."
)

# --- Tool policy ----------------------------------------------------------
# Pre-approved tools; network is blocked by the firewall + settings denies.
ALLOWED_TOOLS: list[str] = [
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "Bash",
    "Grep",
    "Glob",
    "TodoWrite",
]

# Strip network, subagent, publishing, and scheduling built-ins entirely;
# unknown names are a no-op in older CLIs, so the list blocks generously.
DISALLOWED_TOOLS: list[str] = [
    "WebSearch",
    "WebFetch",
    "Task",
    "Agent",
    "Workflow",
    "Skill",
    "SlashCommand",
    "Artifact",
    "SendMessage",
    "SendUserFile",
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskUpdate",
    "TaskOutput",
    "TaskStop",
    "CronCreate",
    "CronDelete",
    "CronList",
    "ScheduleWakeup",
    "Monitor",
    "PushNotification",
    "RemoteTrigger",
    "EnterWorktree",
    "ExitWorktree",
    "EnterPlanMode",
    "ExitPlanMode",
    "EndConversation",
    "ToolSearch",
    "AskUserQuestion",
    "NotebookEdit",
    "PowerShell",
]

# Permission mode: bypass interactive prompts; policy enforced by the
# agent_settings.json deny list and disallowed_tools.
PERMISSION_MODE: PermissionMode = "bypassPermissions"

# Per-RESPONSE output cap for the CLI (its 32k default kills long single
# thinking turns); the attempt budget is enforced by BudgetTracker, not this.
MAX_OUTPUT_TOKENS_ENV: str = "CLAUDE_CODE_MAX_OUTPUT_TOKENS"
MAX_OUTPUT_TOKENS_PER_RESPONSE: int = 64000
# Claude's task-budget API rejects smaller values for current frontier models.
# A late wrap-up may have fewer experiment tokens remaining; in that case the
# provider receives this minimum while BudgetTracker still enforces the exact
# local attempt cutoff and the prompt states the true remaining budget.
PROVIDER_MIN_TASK_BUDGET_TOKENS: int = 20000
# Docker sets this to the pinned npm CLI (the SDK-bundled CLI ignores the
# output cap on Opus); unset = the SDK's bundled CLI (dev host runs).
CLI_PATH_ENV: str = "HARNESS_CLI_PATH"

# --- Providers ------------------------------------------------------------
# 'vendor/model' ids (e.g. openai/gpt-5.4) route through OpenRouter's
# Anthropic-compatible endpoint; bare ids use the Anthropic API directly.
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api"
OPENROUTER_KEY_ENV: str = "OPENROUTER_API_KEY"
LITELLM_MODEL_PREFIX: str = "litellm/"
LITELLM_BASE_URL_ENV: str = "LITELLM_BASE_URL"
LITELLM_API_KEY_ENV: str = "LITELLM_API_KEY"
VLLM_MODEL_PREFIX: str = "vllm/"
VLLM_BASE_URL_ENV: str = "VLLM_BASE_URL"
VLLM_API_KEY_ENV: str = "VLLM_API_KEY"
ANTHROPIC_BASE_URL_ENV: str = "ANTHROPIC_BASE_URL"
ANTHROPIC_AUTH_TOKEN_ENV: str = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_API_KEY_ENV: str = "ANTHROPIC_API_KEY"
# GPT models can spend substantially longer than Claude's default request and
# stream-idle windows in hidden reasoning before emitting their first
# Anthropic-compatible SSE event. Keep both transport windows at one hour.
CLAUDE_API_TIMEOUT_ENV: str = "API_TIMEOUT_MS"
CLAUDE_API_TIMEOUT_MS: int = 3_600_000
CLAUDE_ENABLE_STREAM_WATCHDOG_ENV: str = "CLAUDE_ENABLE_STREAM_WATCHDOG"
CLAUDE_STREAM_IDLE_TIMEOUT_ENV: str = "CLAUDE_STREAM_IDLE_TIMEOUT_MS"
CLAUDE_STREAM_IDLE_TIMEOUT_MS: int = 3_600_000
# LiteLLM's ChatGPT Responses translation is streaming-only for this harness.
# A second, non-streaming request can duplicate inference/tool effects and has
# returned an empty ``output`` list in live tests, so fail instead.
CLAUDE_DISABLE_NONSTREAMING_FALLBACK_ENV: str = (
    "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK"
)
# Claude Code otherwise retries each failed request ten times. A failed request
# can consume backend inference without returning usage, so canonical
# token-matched runs must not transparently spend another request.
CLAUDE_MAX_API_RETRIES_ENV: str = "CLAUDE_CODE_MAX_RETRIES"
CLAUDE_MAX_API_RETRIES: int = 0
# The proxy must not blindly replay a request after an interrupted stream: it
# cannot know which assistant/tool events the local transcript committed.  The
# harness instead reopens that exact transcript and asks it to continue.  Keep
# retries bounded so a persistent provider outage remains an infrastructure
# failure rather than an indefinitely running benchmark attempt.
TRANSPORT_RECOVERY_MAX_RETRIES: int = 6
TRANSPORT_RECOVERY_BASE_DELAY_SECONDS: float = 2.0
TRANSPORT_RECOVERY_MAX_DELAY_SECONDS: float = 30.0

# --- Auth / resume --------------------------------------------------------
OAUTH_TOKEN_ENV: str = "CLAUDE_CODE_OAUTH_TOKEN"
CLAUDE_CONFIG_DIR_ENV: str = "CLAUDE_CONFIG_DIR"
# Extra seconds to wait past a reported rate-limit reset time (clock skew).
RESET_WAIT_BUFFER_SECONDS: int = 60
# Cooldown applied when a rejection reports no usable reset time; without it a
# resets_at of 0/None would put the token straight back into rotation.
RATE_LIMIT_FALLBACK_COOLDOWN_SECONDS: int = 300
# CLI stderr markers meaning the token's org is out of budget (no reset to
# wait for): the token is removed and the live conversation changes credentials.
SPEND_LIMIT_MARKERS: tuple[str, ...] = ("spend limit", "usage limit reached")

# --- Logging --------------------------------------------------------------
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s %(levelname)s %(name)s %(message)s"
