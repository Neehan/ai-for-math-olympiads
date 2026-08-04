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
# Transient agent working space; its contents are copied into the seed's
# results dir after the run, so this root itself is git-ignored.
SCRATCH_ROOT: Path = REPO_ROOT / ".scratch"

# --- Per-seed output filenames -------------------------------------------
LOGS_FILENAME: str = "logs.jsonl.zst"
SOLUTION_FILENAME: str = "solution.md"
# Sequential arms also snapshot the solution at each lower budget cut
# (e.g. solution_2x.md): the last complete write-up emitted before cumulative
# output tokens crossed that budget — what a hard-stopped run would grade.
SOLUTION_CUT_FILENAME_FORMAT: str = "solution_{multiplier}x.md"
META_FILENAME: str = "meta.json"
SCRATCH_SUBDIR: str = "scratch"
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
# Hint ladder: h1 placebo (unauthored — fail fast), h2 tags (the hint arm),
# h3 outline (bigger tier, unused).
HINT_NONE: str = "none"
HINT_H1: str = "h1"
HINT_H2: str = "h2"
HINT_H3: str = "h3"
HINT_KINDS: frozenset[str] = frozenset({HINT_NONE, HINT_H1, HINT_H2, HINT_H3})
MODE_SINGLE: str = "single"
MODE_SEQUENTIAL: str = "sequential"
MODES: frozenset[str] = frozenset({MODE_SINGLE, MODE_SEQUENTIAL})

# --- Phase labels ---------------------------------------------------------
PHASE_SOLVE: str = "solve"
PHASE_CRITIQUE: str = "critique"
PHASE_REVISE: str = "revise"
PHASE_WRAP_UP: str = "wrap_up"

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

# --- Providers ------------------------------------------------------------
# 'vendor/model' ids (e.g. openai/gpt-5.5) route through OpenRouter's
# Anthropic-compatible endpoint; bare ids use the Anthropic API directly.
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api"
OPENROUTER_KEY_ENV: str = "OPENROUTER_API_KEY"
ANTHROPIC_BASE_URL_ENV: str = "ANTHROPIC_BASE_URL"
ANTHROPIC_AUTH_TOKEN_ENV: str = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_API_KEY_ENV: str = "ANTHROPIC_API_KEY"

# --- Auth / resume --------------------------------------------------------
OAUTH_TOKEN_ENV: str = "CLAUDE_CODE_OAUTH_TOKEN"
# Extra seconds to wait past a reported rate-limit reset time (clock skew).
RESET_WAIT_BUFFER_SECONDS: int = 60
# Cooldown applied when a rejection reports no usable reset time; without it a
# resets_at of 0/None would put the token straight back into rotation.
RATE_LIMIT_FALLBACK_COOLDOWN_SECONDS: int = 300

# --- Logging --------------------------------------------------------------
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s %(levelname)s %(name)s %(message)s"
