"""Single source of truth for harness configuration constants."""

from pathlib import Path

from claude_agent_sdk import PermissionMode
from dotenv import load_dotenv

# Load .env (auth token etc.) as soon as constants is imported, so every
# entrypoint and env-reading path sees it. .env is git-ignored.
load_dotenv()

# --- Model ---------------------------------------------------------------
# Opus 4.5 has a May-2025 knowledge cutoff, so 2026 problems are provably
# novel to it (the NOVEL condition in the paper).
MODEL: str = "claude-opus-4-5"

# --- Paths ---------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
PROBLEMS_PATH: Path = REPO_ROOT / "problems.jsonl"
RESULTS_ROOT: Path = REPO_ROOT / "results"
# Full untruncated per-attempt logs (tool calls, inputs, results) for auditing.
LOGS_ROOT: Path = REPO_ROOT / "logs"
# Agent scratch space, kept out of results/ so it never contaminates outputs.
SCRATCH_ROOT: Path = REPO_ROOT / ".scratch"

SINGLE_LLM_DIR: str = "single_llm"
BEST_OF_N_DIR: str = "best_of_n"
RALPH_LOOP_DIR: str = "ralph_loop"

# --- Tool policy ---------------------------------------------------------
# Tools the agent is pre-approved to use (run headless without prompting).
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

# Tools removed from the agent. In this SDK, disallowed_tools is the mechanism
# that actually blocks built-ins (tested: allowed_tools and can_use_tool do NOT
# reliably gate WebSearch; disallowed_tools does). Network egress via Bash
# (curl/pip/etc.) is NOT covered here — that is handled by running under
# `docker --network none`.
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

# Regex patterns for Bash commands blocked by the PreToolUse network guard.
#
# Each pattern is anchored to a COMMAND POSITION — start of the command string
# or right after a shell separator (; | & && || newline `$( ( ) — via the
# _CMD_START prefix. This means a network binary is only blocked when actually
# invoked as a command, so tokens like "host_data.txt", a Python string
# containing "socket", or a variable named "http_status" are NOT false-blocked
# (which would silently corrupt a legitimate verification script).
#
# Deliberately NOT blocked: bare "urllib"/"socket."/"requests." substrings, and
# tokens like "host"/"dig" that collide with ordinary math code. Reaching the
# network through Python still requires the model to import and call those
# libraries, which is far less likely than a false positive on innocent code;
# and every Bash call is recorded in the audit log regardless.
_CMD_START = r"(?:^|[;&|`]|\$\(|\(|\n)\s*(?:sudo\s+)?"

BLOCKED_BASH_PATTERNS: list[str] = [
    _CMD_START + r"curl\b",
    _CMD_START + r"wget\b",
    _CMD_START + r"ncat\b",
    _CMD_START + r"telnet\b",
    _CMD_START + r"ssh\b",
    _CMD_START + r"scp\b",
    _CMD_START + r"sftp\b",
    _CMD_START + r"ftp\b",
    _CMD_START + r"rsync\b",
    _CMD_START + r"ping\b",
    _CMD_START + r"nslookup\b",
    _CMD_START + r"dig\s",
    _CMD_START + r"host\s",
    r"\bgit\s+(?:clone|pull|push|fetch|remote|submodule)\b",
    r"\bpip[23]?\s+install\b",
    r"\bpip[23]?\s+download\b",
    r"\bconda\s+install\b",
    r"\bnpm\s+(?:install|i|ci)\b",
    r"\byarn\s+(?:add|install)\b",
    r"\bapt(?:-get)?\s+install\b",
    r"\bbrew\s+install\b",
]

# --- Run parameters ------------------------------------------------------
# Max agent turns (assistant<->tool cycles) per single attempt. This is a
# runaway/cost guard AND the pacing budget the model is told about. Used by
# every harness and every Ralph iteration so all systems have an equivalent
# per-attempt turn budget.
MAX_TURNS_PER_ATTEMPT: int = 128

# Best-of-N: number of independent samples per problem.
N_SAMPLES: int = 5

# Global cap on concurrently-running agent sessions across a harness run. Each
# session is one `claude` subprocess and consumes API rate limit; tune to your
# Anthropic tier. Applies to all harnesses (problems and BoN samples).
MAX_CONCURRENCY: int = 9

# Ralph loop: number of self-refinement iterations per problem.
RALPH_ITERATIONS: int = 16

# Permission mode: bypass interactive prompts; policy enforced by can_use_tool.
PERMISSION_MODE: PermissionMode = "bypassPermissions"

# --- Auth ----------------------------------------------------------------
# The Claude Code OAuth token (from `claude setup-token`) that every session
# authenticates with. Read from this env var so it is never committed. If it is
# unset, the CLI falls back to its own stored login.
OAUTH_TOKEN_ENV: str = "CLAUDE_CODE_OAUTH_TOKEN"
# Extra seconds to wait past a reported reset time before resuming (clock skew).
RESET_WAIT_BUFFER_SECONDS: int = 60

# --- Logging -------------------------------------------------------------
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s %(levelname)s %(name)s %(message)s"

# --- Output --------------------------------------------------------------
# Max characters of each tool-call input/result kept in the markdown log.
# Tool names + inputs are always shown (they prove which tools ran); long
# results are truncated to keep files readable.
TOOL_LOG_TRUNCATE: int = 500
