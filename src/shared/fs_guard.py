"""PreToolUse hook that confines all filesystem access to the scratch sandbox.

The tool policy pre-approves Read/Write/Edit/MultiEdit/Bash/Grep/Glob, but none
of those is path-restricted by the SDK: setting `cwd` only chooses where the
agent STARTS, it is not a jail. Without this guard the agent could `cat` the
reference solutions, read another harness's results, or read anywhere the
launching user can. That would void the experiment.

This hook binds to one per-problem scratch directory and blocks any tool call
that touches a path outside it. It covers both the file tools (via their path
argument) and Bash (by scanning the command string for out-of-scratch paths).
Two locations are treated as inside the sandbox:

- the scratch directory itself (and anything under it), and
- the SDK's own mirror of that scratch dir under the system temp tree
  (`.../claude-<uid>/<mangled-scratch-path>/...`), which the SDK creates to run
  the agent's Bash — blocking it would break every legitimate Bash call.

Like the network guard it is a best-effort layer (a determined adversary could
obfuscate a path), and every call — blocked or not — is in the audit log, so a
run can still be proven sandbox-confined after the fact. The model has no
incentive to evade.
"""

import re
from collections.abc import Awaitable, Callable
from pathlib import Path

from claude_agent_sdk import HookContext, HookInput, HookJSONOutput

HookFn = Callable[[HookInput, str | None, HookContext], Awaitable[HookJSONOutput]]

# Tools whose file argument must be checked, mapped to the input key holding the
# path. Grep/Glob take an optional `path` (absent => defaults to cwd = scratch,
# which is allowed); Read/Write/Edit/MultiEdit always carry `file_path`.
_PATH_TOOLS: dict[str, str] = {
    "Read": "file_path",
    "Write": "file_path",
    "Edit": "file_path",
    "MultiEdit": "file_path",
    "Grep": "path",
    "Glob": "path",
}

# Absolute-path tokens inside a Bash command. We only inspect ABSOLUTE paths
# (leading `/` or `~`): relative paths resolve under cwd = scratch and are safe,
# and trying to resolve every bare word as a path yields false positives on
# ordinary tokens. A token is a run of path characters starting at `/` or `~`.
_ABS_PATH_TOKEN = re.compile(r"(?<![\w./~-])(~|/)[^\s'\";|&<>()`]*")


def _is_inside(candidate: Path, scratch_root: Path, mirror_marker: str) -> bool:
    """True if candidate resolves inside the scratch dir or its SDK temp mirror.

    scratch_root is the absolute, resolved per-problem scratch directory.
    mirror_marker is the mangled form of that path the SDK embeds in its temp
    mirror directory name, so the agent's own Bash working area is permitted.
    A relative candidate is resolved against scratch_root (the agent's cwd), not
    the guard process's cwd, so relative paths correctly count as inside.
    """
    base = candidate if candidate.is_absolute() else scratch_root / candidate
    resolved = base.resolve()
    if resolved == scratch_root or scratch_root in resolved.parents:
        return True
    # SDK mirror: e.g. /private/tmp/claude-501/<mangled-scratch-path>/<uuid>/...
    # Allow only when the mangled marker for THIS scratch appears in the path,
    # so one problem's mirror can't be used to reach another's.
    return mirror_marker in str(resolved)


def _bad_paths(command: str, scratch_root: Path, mirror_marker: str) -> list[str]:
    """Return absolute paths in a Bash command that escape the sandbox."""
    bad: list[str] = []
    for match in _ABS_PATH_TOKEN.finditer(command):
        token = match.group(0)
        expanded = Path(token).expanduser()
        if not _is_inside(expanded, scratch_root, mirror_marker):
            bad.append(token)
    return bad


def make_fs_guard(scratch_dir: str) -> HookFn:
    """Build a PreToolUse hook that confines file access to scratch_dir.

    scratch_dir is the per-problem scratch path passed as the agent's cwd. The
    returned coroutine is registered as a PreToolUse hook for every file tool
    and Bash; it returns a block decision for any out-of-sandbox path.
    """
    scratch_root = Path(scratch_dir).resolve()
    # The SDK mangles the scratch path into its temp-mirror dir name by
    # replacing every path separator with '-'. Matching this marker lets the
    # agent's own Bash mirror through while still blocking other problems'.
    mirror_marker = str(scratch_root).replace("/", "-")

    async def fs_guard(
        input_data: HookInput,
        tool_use_id: str | None,
        context: HookContext,
    ) -> HookJSONOutput:
        """Block a tool call that reads or writes outside the scratch sandbox."""
        tool_name = input_data.get("tool_name")
        tool_input = input_data.get("tool_input", {})
        if not isinstance(tool_input, dict):
            return {}

        if tool_name == "Bash":
            command = str(tool_input.get("command", ""))
            bad = _bad_paths(command, scratch_root, mirror_marker)
            if bad:
                return {
                    "decision": "block",
                    "reason": (
                        f"Blocked: Bash command references paths outside the "
                        f"scratch sandbox {scratch_root}: {bad}. All work must "
                        f"stay inside your scratch directory."
                    ),
                }
            return {}

        path_key = _PATH_TOOLS.get(str(tool_name))
        if path_key is None:
            return {}
        raw = tool_input.get(path_key)
        if raw is None or not isinstance(raw, str):
            return {}
        if not _is_inside(Path(raw).expanduser(), scratch_root, mirror_marker):
            return {
                "decision": "block",
                "reason": (
                    f"Blocked: {tool_name} target '{raw}' is outside the scratch "
                    f"sandbox {scratch_root}. All file access must stay inside "
                    f"your scratch directory."
                ),
            }
        return {}

    return fs_guard
