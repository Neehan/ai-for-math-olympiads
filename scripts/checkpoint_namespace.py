"""Compute checkpoint namespaces while preserving legacy-arm identities."""

import hashlib
import sys
from pathlib import Path


AUXILIARY_ARMS = {
    "baseline-uniform-strategy-only",
    "baseline-uniform-compress",
    "selection",
    "selection-no-problem",
}
AUXILIARY_PROMPTS = {
    "strategy_state_audit.md",
    "uniform_compress.md",
    "selection.md",
    "selection_wrap.md",
}
_SELECTION_ARMS = {"selection", "selection-no-problem"}
_LATE_INTERVENTION_ARMS = {
    "late-baseline-sequential",
    "late-hint-sequential",
    "late-intervention",
}
_LATE_CONTINUATION_PROMPT = "late_continuation.md"
_TWO_BLOCK_SEQUENTIAL_ARM = "baseline-sequential-2x"
_LEGACY_SELECTION_CONFIG = (
    b'  "selection_output_tokens": { "selection": 40000, '
    b'"selection-no-problem": 40000 },\n'
)
_THREE_PARALLEL_BANKS = (
    b'    "baseline-parallel": { "hint": "none", "mode": "parallel", '
    b'"budget_units": 8, "seeds": [1, 2, 3] },\n'
)
_LEGACY_ONE_PARALLEL_BANK = (
    b'    "baseline-parallel": { "hint": "none", "mode": "parallel", '
    b'"budget_units": 8, "seeds": [1] },\n'
)


def namespace(arguments: list[str], settings_path: Path) -> str:
    """Reproduce the historical digest for old arms; fully bind new arms."""
    if len(arguments) not in {4, 5}:
        raise ValueError("Expected stage/model/audit/arm plus optional dataset")
    arm_id = arguments[3]
    arm = arm_id.split(":", 1)[0]
    auxiliary = arm in AUXILIARY_ARMS
    digest = hashlib.sha256("\0".join(arguments).encode())

    config_bytes = Path("config.json").read_bytes()
    # Adding this independent replication arm must not move paid checkpoints
    # for any existing arm. The new arm remains fully bound in its namespace.
    if arm != _TWO_BLOCK_SEQUENTIAL_ARM:
        config_bytes = b"".join(
            line
            for line in config_bytes.splitlines(keepends=True)
            if f'"{_TWO_BLOCK_SEQUENTIAL_ARM}"'.encode() not in line
        )
    if arm not in _LATE_INTERVENTION_ARMS:
        config_bytes = b"".join(
            line
            for line in config_bytes.splitlines(keepends=True)
            if not any(
                f'"{name}"'.encode() in line
                for name in {
                    "late-baseline-sequential",
                    "late-hint-sequential",
                }
            )
        )
    # Bank seeds are independent repetitions of the unchanged Parallel-8
    # protocol.  Expanding the orchestration whitelist must not strand paid
    # seed-1 checkpoints (or unrelated-arm checkpoints) in a new namespace.
    if _THREE_PARALLEL_BANKS not in config_bytes:
        raise ValueError("config.json has no canonical baseline-parallel entry")
    config_bytes = config_bytes.replace(
        _THREE_PARALLEL_BANKS, _LEGACY_ONE_PARALLEL_BANK, 1
    )
    # Selection used to have a separate 40k config field.  It now uses the
    # ordinary 1x arm budget, but retaining that historical line solely in the
    # namespace input keeps every non-selection paid checkpoint resumable.
    if arm not in _SELECTION_ARMS:
        marker = b'  "arms": {\n'
        if marker not in config_bytes:
            raise ValueError("config.json has no arms object")
        config_bytes = config_bytes.replace(
            marker, _LEGACY_SELECTION_CONFIG + marker, 1
        )
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
        if (
            prompt.name == _LATE_CONTINUATION_PROMPT
            and arm not in _LATE_INTERVENTION_ARMS
        ):
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
