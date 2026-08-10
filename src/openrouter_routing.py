"""Frozen OpenRouter request routing used by reproducible harness aliases."""

from __future__ import annotations

from typing import Any


GLM47_FP8_ALIAS = "openrouter/glm-4.7-fp8"


def route_for(model: str) -> dict[str, Any] | None:
    """Return an exact upstream model/provider route for a harness alias."""
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
