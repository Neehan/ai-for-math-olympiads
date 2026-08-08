#!/usr/bin/env python3
"""Internal: verify one ChatGPT subscription through a LiteLLM sidecar."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

DIRECT_MARKER = "CODEX_LITELLM_DIRECT_OK"
AGENT_MARKER = "CODEX_LITELLM_AGENT_SDK_OK"


def _concise_agent_error(value: object) -> str:
    """Describe known gateway failures without dumping HTML challenge tokens."""
    message = str(value)
    if not message:
        return type(value).__name__
    if "System messages are not allowed" in message:
        return (
            "LiteLLM's chatgpt/ Responses adapter rejected the Claude Agent SDK "
            "system prompt: System messages are not allowed"
        )
    if "403" in message and "<html>" in message:
        return "ChatGPT subscription backend returned an HTTP 403 HTML challenge"
    return message if len(message) <= 800 else f"{message[:800]}... [truncated]"


def _response_text(payload: dict[str, Any]) -> str:
    """Extract output text from an OpenAI Responses-compatible payload."""
    texts: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for block in item.get("content", []):
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                texts.append(block["text"])
    return "\n".join(texts)


def _parse_response_body(raw: bytes, content_type: str) -> tuple[dict[str, Any], str]:
    """Parse either JSON or the SSE stream required by the ChatGPT backend."""
    decoded = raw.decode(errors="replace")
    if (
        "text/event-stream" not in content_type.lower()
        and not decoded.lstrip().startswith(("event:", "data:"))
    ):
        payload = json.loads(decoded)
        return payload, _response_text(payload)

    completed: dict[str, Any] | None = None
    deltas: list[str] = []
    for line in decoded.splitlines():
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if not data or data == "[DONE]":
            continue
        event = json.loads(data)
        if event.get("type") == "response.output_text.delta" and isinstance(
            event.get("delta"), str
        ):
            deltas.append(event["delta"])
        if event.get("type") == "response.completed" and isinstance(
            event.get("response"), dict
        ):
            completed = event["response"]
    if completed is None:
        raise RuntimeError("SSE response ended without a response.completed event")
    return completed, _response_text(completed) or "".join(deltas)


def verify_direct(base_url: str, proxy_key: str, model: str, timeout: int) -> None:
    """Call LiteLLM's Responses endpoint before involving Claude Agent SDK."""
    body = json.dumps(
        {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Reply with exactly {DIRECT_MARKER}",
                        }
                    ],
                }
            ],
            "stream": True,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/responses",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {proxy_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            payload, text = _parse_response_body(
                raw, response.headers.get("Content-Type", "")
            )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"LiteLLM returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach LiteLLM at {base_url}: {exc}") from exc

    if DIRECT_MARKER not in text:
        raise RuntimeError(
            f"Direct call returned successfully but omitted {DIRECT_MARKER!r}: {text!r}"
        )
    usage = payload.get("usage", {})
    print(f"PASS direct Responses call: model={payload.get('model', model)!r}")
    print(f"usage={json.dumps(usage, sort_keys=True)}")


async def verify_agent(base_url: str, proxy_key: str, model: str, timeout: int) -> None:
    """Make one Claude Agent SDK call and require a real Write tool action."""
    with tempfile.TemporaryDirectory(prefix="codex-litellm-verify-") as temp_dir:
        root = Path(temp_dir)
        config_dir = root / "claude-config"
        config_dir.mkdir()
        output_path = root / "codex_litellm_verify.txt"
        stderr_lines: list[str] = []
        text_blocks: list[str] = []
        tool_names: list[str] = []
        result: ResultMessage | None = None

        options = ClaudeAgentOptions(
            model=model,
            cwd=root,
            env={
                "ANTHROPIC_BASE_URL": base_url.rstrip("/"),
                # Current Claude Code needs AUTH_TOKEN to send LiteLLM's key as
                # a Bearer token. Clear API_KEY so an inherited Anthropic key
                # cannot take precedence or be sent to the local proxy.
                "ANTHROPIC_AUTH_TOKEN": proxy_key,
                "ANTHROPIC_API_KEY": "",
                "CLAUDE_CONFIG_DIR": str(config_dir),
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "20000",
            },
            system_prompt="You are a minimal tool-using assistant.",
            effort="high",
            task_budget={"total": 20000},
            allowed_tools=["Write"],
            disallowed_tools=["Bash", "Read", "Edit", "WebFetch", "WebSearch"],
            permission_mode="bypassPermissions",
            max_turns=4,
            extra_args={"setting-sources": ""},
            stderr=stderr_lines.append,
        )
        prompt = (
            f"Use the Write tool exactly once to create {output_path} "
            f"containing exactly {AGENT_MARKER} followed by a newline. "
            "Then reply briefly that verification is complete."
        )
        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
                async with asyncio.timeout(timeout):
                    async for message in client.receive_response():
                        if isinstance(message, AssistantMessage):
                            for block in message.content:
                                if isinstance(block, ToolUseBlock):
                                    tool_names.append(block.name)
                                elif isinstance(block, TextBlock):
                                    text_blocks.append(block.text)
                        elif isinstance(message, ResultMessage):
                            result = message
        except Exception as exc:
            stderr_tail = _concise_agent_error("\n".join(stderr_lines[-20:]))
            raise RuntimeError(
                "Claude Agent SDK call failed: "
                f"{_concise_agent_error(exc)}\nCLI stderr tail:\n{stderr_tail}"
            ) from exc

        if result is None:
            raise RuntimeError("Claude Agent SDK stream ended without a ResultMessage")
        if result.is_error:
            raise RuntimeError(
                f"Claude Agent SDK reported an error: {result.subtype}; "
                f"result={_concise_agent_error(result.result)}"
            )
        if "Write" not in tool_names:
            raise RuntimeError(
                f"Agent returned without using Write; tools={tool_names!r}"
            )
        if not output_path.is_file():
            raise RuntimeError(
                "Agent reported success but codex_litellm_verify.txt was not created"
            )
        actual = output_path.read_text()
        expected = f"{AGENT_MARKER}\n"
        if actual != expected:
            raise RuntimeError(
                f"Unexpected file content: {actual!r}; expected {expected!r}"
            )

        print(f"PASS Claude Agent SDK call: model={model!r}, tools={tool_names!r}")
        print(f"assistant_text={' '.join(text_blocks).strip()!r}")
        print(
            "result="
            + json.dumps(
                {
                    "num_turns": result.num_turns,
                    "duration_ms": result.duration_ms,
                    "duration_api_ms": result.duration_api_ms,
                    "total_cost_usd": result.total_cost_usd,
                    "usage": result.usage,
                },
                default=str,
                sort_keys=True,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("direct", "agent"))
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CODEX_LITELLM_BASE_URL", "http://127.0.0.1:4000"),
    )
    parser.add_argument(
        "--proxy-key",
        default=os.environ.get(
            "LITELLM_API_KEY",
            os.environ.get("CODEX_LITELLM_PROXY_KEY", "sk-codex-local-only"),
        ),
    )
    parser.add_argument(
        "--model", default=os.environ.get("CODEX_LITELLM_MODEL", "gpt-5.5")
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("CODEX_LITELLM_TIMEOUT_SECONDS", "900")),
        help="Direct-call timeout; leave time to complete OAuth device flow.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "direct":
        verify_direct(args.base_url, args.proxy_key, args.model, args.timeout)
    else:
        asyncio.run(
            verify_agent(args.base_url, args.proxy_key, args.model, args.timeout)
        )


if __name__ == "__main__":
    main()
