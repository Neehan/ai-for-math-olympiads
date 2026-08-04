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
# The judge has no tools, so a single response suffices; small safety margin.
AUDIT_MAX_TURNS: int = 4
AUDIT_SCORE_VALID: int = 7
AUDIT_SCORE_INVALID: int = 0

# --- Prompt template filenames (in PROMPTS_DIR) --------------------------
SYSTEM_PROMPT_FILE: str = "system.md"
TASK_PROMPT_FILE: str = "task.md"
HINT_PROMPT_FILE: str = "hint.md"
CRITIQUE_PROMPT_FILE: str = "critique.md"
REVISE_PROMPT_FILE: str = "revise.md"
AUDIT_PROMPT_FILE: str = "audit.md"

# --- Problem/hint data sources -------------------------------------------
# Problems and hints are NEVER committed to this repo (contest identity would
# leak); they are fetched from these URLs straight into memory with stdlib
# urllib — no hf_hub, no disk cache. In Docker, the entrypoint prefetches them
# BEFORE the firewall closes (HF stays blocked while agents run) and points
# these env vars at the temp copies, which the loader deletes on read so no
# trace remains for the agent.
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
HINT_NONE: str = "none"
HINT_H1: str = "h1"
HINT_H2: str = "h2"
HINT_KINDS: frozenset[str] = frozenset({HINT_NONE, HINT_H1, HINT_H2})
MODE_SINGLE: str = "single"
MODE_SEQUENTIAL: str = "sequential"
MODES: frozenset[str] = frozenset({MODE_SINGLE, MODE_SEQUENTIAL})

# --- Phase labels ---------------------------------------------------------
PHASE_SOLVE: str = "solve"
PHASE_CRITIQUE: str = "critique"
PHASE_REVISE: str = "revise"

# --- Tool policy ----------------------------------------------------------
# Pre-approved tools (run headless without prompting). Network access is
# blocked by the container firewall and the agent_settings.json deny list,
# not by in-process guards.
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

# Built-in tools removed from the agent entirely (disallowed_tools is the
# mechanism that reliably strips SDK built-ins).
DISALLOWED_TOOLS: list[str] = [
    "WebSearch",
    "WebFetch",
    "Task",
    "Agent",
    "ToolSearch",
    "AskUserQuestion",
    "SlashCommand",
    "NotebookEdit",
]

# Permission mode: bypass interactive prompts; policy enforced by the
# agent_settings.json deny list and disallowed_tools.
PERMISSION_MODE: PermissionMode = "bypassPermissions"

# --- Auth / resume --------------------------------------------------------
OAUTH_TOKEN_ENV: str = "CLAUDE_CODE_OAUTH_TOKEN"
# Extra seconds to wait past a reported rate-limit reset time (clock skew).
RESET_WAIT_BUFFER_SECONDS: int = 60

# --- Logging --------------------------------------------------------------
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s %(levelname)s %(name)s %(message)s"
