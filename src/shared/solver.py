"""Core single-attempt runner: run the agent once and capture its result.

Shared by all three harnesses because they need identical tool policy and
option construction — differing only in how many attempts they run and how
they seed each one.
"""

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from src.shared.bash_guard import bash_network_guard
from src.shared.constants import (
    MODEL,
    PERMISSION_MODE,
)
from src.shared.models import AttemptResult, ToolCall
from src.shared.prompts import SYSTEM_PROMPT
from src.shared.tool_policy import allowed_tools, can_use_tool, disallowed_tools


def build_options(cwd: str, max_turns: int) -> ClaudeAgentOptions:
    """Construct agent options with the shared tool policy enforced.

    max_turns is the per-attempt runaway/cost guard; callers pass the cap for
    their harness (single/BoN vs Ralph).
    """
    return ClaudeAgentOptions(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=allowed_tools(),
        disallowed_tools=disallowed_tools(),
        can_use_tool=can_use_tool,
        hooks={
            "PreToolUse": [HookMatcher(matcher="Bash", hooks=[bash_network_guard])],
        },
        permission_mode=PERMISSION_MODE,
        max_turns=max_turns,
        cwd=cwd,
        setting_sources=[],
    )


def _collect_text(messages: list[AssistantMessage]) -> str:
    """Concatenate all assistant text blocks in order."""
    parts: list[str] = []
    for message in messages:
        for block in message.content:
            if isinstance(block, TextBlock):
                parts.append(block.text)
    return "\n".join(parts)


def _stringify_result(content: object) -> str:
    """Flatten a ToolResultBlock's content (str or list of blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _collect_tool_calls(
    tool_uses: dict[str, ToolUseBlock],
    tool_results: dict[str, ToolResultBlock],
) -> list[ToolCall]:
    """Pair each tool-use block with its result block by tool_use_id."""
    calls: list[ToolCall] = []
    for use_id, use in tool_uses.items():
        result = tool_results.get(use_id)
        calls.append(
            ToolCall(
                name=use.name,
                tool_input=dict(use.input),
                result=_stringify_result(result.content) if result else "",
                is_error=bool(result.is_error) if result else False,
            )
        )
    return calls


async def run_attempt(client: ClaudeSDKClient, prompt: str) -> AttemptResult:
    """Send one prompt on an existing client session and capture the result.

    Captures every tool call (name, input, result) so the attempt is auditable.
    Fails loud: if no ResultMessage is received the run raises, rather than
    silently returning an empty attempt.
    """
    await client.query(prompt)

    assistant_messages: list[AssistantMessage] = []
    tool_uses: dict[str, ToolUseBlock] = {}
    tool_results: dict[str, ToolResultBlock] = {}
    result_message: ResultMessage | None = None

    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            assistant_messages.append(message)
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    tool_uses[block.id] = block
        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    tool_results[block.tool_use_id] = block
        elif isinstance(message, ResultMessage):
            result_message = message

    if result_message is None:
        raise RuntimeError("Agent produced no ResultMessage for prompt")

    text = _collect_text(assistant_messages)
    if result_message.result is not None:
        text = result_message.result

    cost = result_message.total_cost_usd
    return AttemptResult(
        text=text,
        num_turns=result_message.num_turns,
        duration_ms=result_message.duration_ms,
        total_cost_usd=cost if cost is not None else 0.0,
        is_error=result_message.is_error,
        stop_reason=result_message.stop_reason
        if result_message.stop_reason is not None
        else "unknown",
        tool_calls=_collect_tool_calls(tool_uses, tool_results),
    )
