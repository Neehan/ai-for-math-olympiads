"""PreToolUse hook that blocks network-capable Bash commands.

A defense-in-depth layer on top of removing the WebSearch/WebFetch tools: even
with those gone, the allowed Bash tool could reach the network via curl, wget,
pip install, git remote ops, ssh, nc, etc. This hook inspects each Bash command
and blocks any that match a known network binary/operation, so a run cannot
fetch external content (contamination) through the shell.

It is a pattern blocklist, so it is not a hard guarantee against a determined
adversary (aliases, obfuscation) — the model has no incentive to evade — but it
stops the obvious cases and every block is recorded in the audit log.
"""

import re

from claude_agent_sdk import HookContext, HookJSONOutput, HookInput

from src.shared.constants import BLOCKED_BASH_PATTERNS

_COMPILED: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in BLOCKED_BASH_PATTERNS
]


def _matches_blocked(command: str) -> str | None:
    """Return the offending pattern if the command looks network-capable."""
    for pattern in _COMPILED:
        if pattern.search(command):
            return pattern.pattern
    return None


async def bash_network_guard(
    input_data: HookInput,
    tool_use_id: str | None,
    context: HookContext,
) -> HookJSONOutput:
    """Block Bash commands that reach the network; allow everything else."""
    if input_data.get("tool_name") != "Bash":
        return {}
    tool_input = input_data.get("tool_input", {})
    command = str(tool_input.get("command", "")) if isinstance(tool_input, dict) else ""
    offending = _matches_blocked(command)
    if offending is None:
        return {}
    return {
        "decision": "block",
        "reason": (
            f"Blocked network command (pattern: {offending}). This experiment "
            "runs with no external network access."
        ),
    }
