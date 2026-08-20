"""Frozen OpenRouter request routing used by reproducible harness aliases."""

from __future__ import annotations

from typing import Any


GLM47_FP8_ALIAS = "openrouter/glm-4.7-fp8"
DEEPSEEK_V4_FLASH_0731 = "deepseek/deepseek-v4-flash-0731"


def route_for(model: str) -> dict[str, Any] | None:
    """Return an exact upstream model/provider route for a harness alias."""
    if model == DEEPSEEK_V4_FLASH_0731:
        return {
            "model": DEEPSEEK_V4_FLASH_0731,
            "provider": {
                # Exclude the two cheapest but slow endpoints. These four are
                # the inexpensive high-throughput endpoints frozen for this run.
                "only": ["relace", "baidu", "streamlake", "deepinfra"],
                "allow_fallbacks": True,
                "require_parameters": False,
                "sort": "throughput",
                "max_price": {"prompt": 0.08, "completion": 0.18},
            },
        }
    if model != GLM47_FP8_ALIAS:
        return None
    return {
        "model": "z-ai/glm-4.7",
        "provider": {
            "only": ["streamlake/fp8"],
            "allow_fallbacks": False,
            # Claude CLI adds Anthropic-only bookkeeping fields that no
            # third-party endpoint advertises verbatim. OpenRouter's
            # compatibility layer must be allowed to translate or omit them.
            "require_parameters": False,
            "quantizations": ["fp8"],
            # OpenRouter max_price values are USD per million tokens. Refuse
            # inference if the currently advertised endpoint price increases.
            "max_price": {"prompt": 0.48, "completion": 1.76},
        },
    }
