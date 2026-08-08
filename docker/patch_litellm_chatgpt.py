#!/usr/bin/env python3
"""Patch LiteLLM 1.97.0 to merge list-valued Claude system content safely."""

from __future__ import annotations

import py_compile
from pathlib import Path

ROOT = Path("/app/.venv/lib")
TARGETS = tuple(
    ROOT.glob(
        "python*/site-packages/litellm/litellm_core_utils/prompt_templates/factory.py"
    )
)
OLD = '                    next_m["content"] = m["content"] + " " + next_m["content"]'
NEW = """                    if isinstance(m["content"], list) or isinstance(next_m["content"], list):
                        system_content = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": str(m["content"])}]
                        next_content = next_m["content"] if isinstance(next_m["content"], list) else [{"type": "text", "text": str(next_m["content"])}]
                        next_m["content"] = system_content + next_content
                    else:
                        next_m["content"] = str(m["content"]) + " " + str(next_m["content"])"""


def main() -> None:
    if len(TARGETS) != 1:
        raise RuntimeError(
            f"expected exactly one LiteLLM prompt factory, found {TARGETS!r}"
        )
    target = TARGETS[0]
    source = target.read_text()
    if source.count(OLD) != 1:
        raise RuntimeError(
            "LiteLLM source changed; refusing to apply an unverified patch"
        )
    target.write_text(source.replace(OLD, NEW))
    py_compile.compile(str(target), doraise=True)


if __name__ == "__main__":
    main()
