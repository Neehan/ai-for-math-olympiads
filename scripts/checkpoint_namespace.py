"""Compute checkpoint namespaces while preserving legacy-arm identities."""

import hashlib
import sys
from pathlib import Path


AUXILIARY_ARMS = {
    "baseline-uniform-strategy-only",
    "baseline-uniform-compress",
    "selection-10k",
    "selection",
    "selection-40k",
    "selection-no-problem",
}
AUXILIARY_PROMPTS = {
    "strategy_state_audit.md",
    "uniform_compress.md",
    "selection.md",
    "selection_no_problem.md",
}


def namespace(arguments: list[str], settings_path: Path) -> str:
    """Reproduce the historical digest for old arms; fully bind new arms."""
    if len(arguments) not in {4, 5}:
        raise ValueError("Expected stage/model/audit/arm plus optional dataset")
    arm_id = arguments[3]
    arm = arm_id.split(":", 1)[0]
    auxiliary = arm in AUXILIARY_ARMS
    digest = hashlib.sha256("\0".join(arguments).encode())

    config_bytes = Path("config.json").read_bytes()
    if not auxiliary:
        config_bytes = b"".join(
            line
            for line in config_bytes.splitlines(keepends=True)
            if not any(f'"{name}"'.encode() in line for name in AUXILIARY_ARMS)
        )
    digest.update(b"config.json\0" + config_bytes + b"\0")
    digest.update(
        settings_path.name.encode()
        + b"\0"
        + settings_path.read_bytes()
        + b"\0"
    )
    prompts = sorted(Path("prompts").glob("*.md"))
    for prompt in prompts:
        if prompt.name == "state_audit.md":
            continue
        if not auxiliary and prompt.name in AUXILIARY_PROMPTS:
            continue
        digest.update(prompt.name.encode() + b"\0" + prompt.read_bytes() + b"\0")
    return digest.hexdigest()[:24]


def main() -> None:
    if len(sys.argv) not in {6, 7}:
        raise SystemExit(
            "usage: checkpoint_namespace.py STAGE MODEL AUDIT ARM [DATASET] SETTINGS"
        )
    print(namespace(sys.argv[1:-1], Path(sys.argv[-1])))


if __name__ == "__main__":
    main()
