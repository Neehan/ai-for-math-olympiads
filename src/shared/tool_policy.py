"""Tool permission policy.

Tested behavior of this SDK version:
- disallowed_tools reliably removes built-ins (WebSearch etc.) — this is the
  enforcement mechanism.
- allowed_tools pre-approves tools so they run headless without prompting.
- can_use_tool is kept as a secondary deny for anything not allowlisted (it
  fires for MCP/custom paths), but it does NOT reliably gate built-ins, so it
  is not the guarantee.

Network egress via Bash (curl/pip/git) is out of scope here and is handled by
running under `docker --network none`.
"""

from typing import Any

from claude_agent_sdk import (
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)

from src.shared.constants import ALLOWED_TOOLS, DISALLOWED_TOOLS

_ALLOWED_SET: set[str] = set(ALLOWED_TOOLS)


async def can_use_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    context: ToolPermissionContext,
) -> PermissionResultAllow | PermissionResultDeny:
    """Secondary gate: allow allowlisted tools, deny everything else."""
    if tool_name in _ALLOWED_SET:
        return PermissionResultAllow()
    return PermissionResultDeny(
        message=f"Tool '{tool_name}' is not permitted in this experiment.",
        interrupt=False,
    )


def allowed_tools() -> list[str]:
    """Return the pre-approved tool list."""
    return list(ALLOWED_TOOLS)


def disallowed_tools() -> list[str]:
    """Return the tools removed from the agent (the enforcing blocklist)."""
    return list(DISALLOWED_TOOLS)
