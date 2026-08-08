#!/usr/bin/env python3
"""Internal: convert Codex auth.json into LiteLLM's ChatGPT auth schema."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REQUIRED_TOKEN_FIELDS = ("access_token", "refresh_token", "id_token")
IDENTITY_FILENAME = "identity.sha256"


def _atomic_write(destination: Path, payload: str) -> None:
    """Write one credential-adjacent file atomically with owner-only mode."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", text=True
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def convert(source: Path, destination: Path) -> tuple[str, ...]:
    """Copy required OAuth fields without ever logging their values."""
    with source.open() as stream:
        raw: dict[str, Any] = json.load(stream)
    tokens = raw.get("tokens")
    if not isinstance(tokens, dict):
        raise ValueError("Codex auth file has no object-valued 'tokens' field")

    missing = [
        field
        for field in REQUIRED_TOKEN_FIELDS
        if not isinstance(tokens.get(field), str) or not tokens[field]
    ]
    if missing:
        raise ValueError(f"Codex auth file is missing token fields: {missing}")

    converted = {field: tokens[field] for field in REQUIRED_TOKEN_FIELDS}
    if isinstance(tokens.get("account_id"), str) and tokens["account_id"]:
        converted["account_id"] = tokens["account_id"]

    identity_source = converted.get("account_id") or converted["refresh_token"]
    identity = hashlib.sha256(identity_source.encode()).hexdigest()
    _atomic_write(destination, json.dumps(converted) + "\n")
    _atomic_write(destination.with_name(IDENTITY_FILENAME), identity + "\n")
    return tuple(sorted(converted))


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: {sys.argv[0]} SOURCE_AUTH_JSON DEST_AUTH_JSON")
    fields = convert(Path(sys.argv[1]), Path(sys.argv[2]))
    print(
        "Copied Codex OAuth credential for the local gateway; "
        f"fields={','.join(fields)} mode=0600"
    )


if __name__ == "__main__":
    main()
