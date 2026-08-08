"""Durable, private checkpoints for every paid model/judge conversation.

Checkpoints are operational state, not experimental outputs.  They live on a
host bind mount outside ``results/`` and are deleted only after the canonical
result marker has been written.  Per-phase files are committed before the
small controller manifest, so a process death at either side of a phase
boundary can be reconciled without repeating or losing charged tokens.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import os
import re
import shutil
import uuid
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Any

from src.constants import (
    AGENT_SETTINGS_PATH,
    CHECKPOINT_ROOT_DEFAULT,
    CHECKPOINT_ROOT_ENV,
    DEFER_CHECKPOINT_CLEANUP_ENV,
    PROMPTS_DIR,
)
from src.models import PhaseResult, ReconnectEvent, ToolCall
from src.solver import BudgetTracker

SCHEMA_VERSION = 2
_ROLE_RE = re.compile(r"^[a-z0-9_-]+$")
_SCRATCH_RE = re.compile(r"^[0-9a-f]{8}$")


@cache
def protocol_fingerprint() -> str:
    """Hash every mounted prompt/tool-policy file that can affect a session."""
    digest = hashlib.sha256()
    for path in [AGENT_SETTINGS_PATH, *sorted(PROMPTS_DIR.glob("*.md"))]:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    with tmp.open("w", encoding="utf-8") as handle:
        os.chmod(tmp, 0o600)
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _tool_record(call: ToolCall) -> dict[str, object]:
    return {
        "name": call.name,
        "input": call.tool_input,
        "result": call.result,
        "is_error": call.is_error,
    }


def _tool_from_record(record: dict[str, Any]) -> ToolCall:
    raw_input = record.get("input", {})
    return ToolCall(
        name=str(record["name"]),
        tool_input=dict(raw_input) if isinstance(raw_input, dict) else {},
        result=str(record.get("result", "")),
        is_error=bool(record.get("is_error", False)),
    )


def phase_record(phase: PhaseResult) -> dict[str, object]:
    """Lossless JSON form shared by checkpoint restoration and final logs."""
    return {
        "label": phase.label,
        "prompt": phase.prompt,
        "text": phase.text,
        "output_tokens": phase.output_tokens,
        "cumulative_output_tokens": phase.cumulative_output_tokens,
        "num_turns": phase.num_turns,
        "duration_ms": phase.duration_ms,
        "total_cost_usd": phase.total_cost_usd,
        "is_error": phase.is_error,
        "stop_reason": phase.stop_reason,
        "budget_exhausted": phase.budget_exhausted,
        "tool_calls": [_tool_record(call) for call in phase.tool_calls],
        "reconnects": [dataclasses.asdict(event) for event in phase.reconnects],
        "provider_usage": phase.provider_usage,
        "process_resume_count": phase.process_resume_count,
        "discarded_output_text": phase.discarded_output_text,
        "discarded_tool_calls": [
            _tool_record(call) for call in phase.discarded_tool_calls
        ],
    }


def phase_from_record(record: dict[str, Any]) -> PhaseResult:
    """Restore one committed phase, including operational recovery metadata."""
    return PhaseResult(
        label=str(record["label"]),
        prompt=str(record["prompt"]),
        text=str(record["text"]),
        output_tokens=int(record["output_tokens"]),
        cumulative_output_tokens=int(record["cumulative_output_tokens"]),
        num_turns=int(record["num_turns"]),
        duration_ms=int(record["duration_ms"]),
        total_cost_usd=float(record["total_cost_usd"]),
        is_error=bool(record["is_error"]),
        stop_reason=str(record["stop_reason"]),
        budget_exhausted=bool(record["budget_exhausted"]),
        tool_calls=[_tool_from_record(item) for item in record.get("tool_calls", [])],
        reconnects=[ReconnectEvent(**item) for item in record.get("reconnects", [])],
        provider_usage=dict(record.get("provider_usage", {})),
        process_resume_count=int(record.get("process_resume_count", 0)),
        discarded_output_text=str(record.get("discarded_output_text", "")),
        discarded_tool_calls=[
            _tool_from_record(item) for item in record.get("discarded_tool_calls", [])
        ],
    )


def progress_tool_calls(progress: dict[str, Any]) -> list[ToolCall]:
    """Flatten the tool prefix saved for a killed in-progress response."""
    uses = progress.get("tool_uses", {})
    results = progress.get("tool_results", {})
    if not isinstance(uses, dict) or not isinstance(results, dict):
        return []
    calls: list[ToolCall] = []
    for use_id, raw_use in uses.items():
        if not isinstance(raw_use, dict):
            continue
        raw_result = results.get(use_id, {})
        if not isinstance(raw_result, dict):
            raw_result = {}
        raw_input = raw_use.get("input", {})
        calls.append(
            ToolCall(
                name=str(raw_use.get("name", "unknown")),
                tool_input=dict(raw_input) if isinstance(raw_input, dict) else {},
                result=str(raw_result.get("result", "<no result returned>")),
                is_error=bool(raw_result.get("is_error", use_id not in results)),
            )
        )
    return calls


def tool_calls_from_records(records: object) -> list[ToolCall]:
    """Restore the accumulated discarded-call ledger for a resumed phase."""
    if not isinstance(records, list):
        return []
    return [_tool_from_record(record) for record in records if isinstance(record, dict)]


class AttemptCheckpoint:
    """Exclusive durable state for one stage/model/arm/problem/seed attempt."""

    def __init__(self, identity: dict[str, object]) -> None:
        canonical = json.dumps(
            identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        self.identity_digest = hashlib.sha256(canonical.encode()).hexdigest()
        self.root = Path(
            os.environ.get(CHECKPOINT_ROOT_ENV, str(CHECKPOINT_ROOT_DEFAULT))
        )
        self.path = self.root / "attempts" / self.identity_digest[:24]
        self.path.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path, 0o700)
        self._lock_handle = (self.path / ".lock").open("a+")
        os.chmod(self.path / ".lock", 0o600)
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._lock_handle.close()
            raise RuntimeError(
                "This exact attempt is already running in another process"
            ) from error

        self.state_path = self.path / "state.json"
        try:
            if self.state_path.exists():
                self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
                if (
                    self.state.get("schema_version") != SCHEMA_VERSION
                    or self.state.get("identity_digest") != self.identity_digest
                ):
                    raise ValueError(f"Incompatible checkpoint at {self.path}")
            else:
                self.state: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "identity_digest": self.identity_digest,
                    "roles": {},
                    "data": {},
                }
                self._save()
        except BaseException:
            self.close()
            raise

    def _save(self) -> None:
        _atomic_json(self.state_path, self.state)

    @staticmethod
    def _validate_role(role: str) -> None:
        if not _ROLE_RE.fullmatch(role):
            raise ValueError(f"Invalid checkpoint role: {role!r}")

    def scratch_dir(self, role: str) -> Path:
        """Persistent opaque cwd containing this role's provider transcript."""
        state = self._role(role)
        scratch_name = state.get("scratch_name")
        if not scratch_name:
            workspace_root = self.root / "w"
            workspace_root.mkdir(parents=True, exist_ok=True)
            os.chmod(workspace_root, 0o700)
            while True:
                candidate = uuid.uuid4().hex[:8]
                path = workspace_root / candidate
                try:
                    path.mkdir()
                    os.chmod(path, 0o700)
                except FileExistsError:
                    continue
                scratch_name = candidate
                state["scratch_name"] = scratch_name
                self._save()
                return path
        path = self.root / "w" / str(scratch_name)
        if not _SCRATCH_RE.fullmatch(str(scratch_name)):
            raise ValueError("Checkpoint scratch name is corrupt")
        if not path.exists():
            has_paid_state = any(
                role_state.get("session_id")
                or role_state.get("active") is not None
                or int(role_state.get("next_sequence", 0)) > 0
                for role_state in self.state.get("roles", {}).values()
                if isinstance(role_state, dict)
            )
            if has_paid_state:
                raise FileNotFoundError(
                    f"Checkpoint workspace disappeared; refusing transcript reset: {path}"
                )
            path.mkdir(parents=True)
            os.chmod(path, 0o700)
        elif not path.is_dir():
            raise NotADirectoryError(path)
        return path

    def _role(self, role: str) -> dict[str, Any]:
        self._validate_role(role)
        roles = self.state.setdefault("roles", {})
        value = roles.setdefault(
            role,
            {
                "session_id": None,
                "reconnects": [],
                "tracker": None,
                "active": None,
                "next_sequence": 0,
            },
        )
        if not isinstance(value, dict):
            raise TypeError(f"Checkpoint role {role!r} is corrupt")
        return value

    def session_id(self, role: str) -> str | None:
        value = self._role(role).get("session_id")
        return str(value) if value else None

    def reconnects(self, role: str) -> list[ReconnectEvent]:
        return [ReconnectEvent(**item) for item in self._role(role)["reconnects"]]

    def save_session(
        self, role: str, session_id: str, reconnects: list[ReconnectEvent]
    ) -> None:
        state = self._role(role)
        previous = state.get("session_id")
        if previous not in (None, session_id):
            raise ValueError("Provider session UUID changed inside one checkpoint role")
        state["session_id"] = session_id
        state["reconnects"] = [dataclasses.asdict(event) for event in reconnects]
        self._save()

    def tracker(
        self, role: str, budget_tokens: int, reserve_tokens: int
    ) -> BudgetTracker:
        state = self._role(role)
        snapshot = state.get("tracker")
        if snapshot is None:
            tracker = BudgetTracker(budget_tokens, reserve_tokens)
            state["tracker"] = tracker.snapshot()
            self._save()
            return tracker
        if not isinstance(snapshot, dict):
            raise TypeError("Checkpoint tracker is corrupt")
        return BudgetTracker.restore(snapshot, budget_tokens, reserve_tokens)

    def phases(self, role: str) -> list[PhaseResult]:
        """Load committed phase files and reconcile a killed boundary commit."""
        state = self._role(role)
        phase_dir = self.path / "phases"
        records: list[tuple[int, dict[str, Any]]] = []
        if phase_dir.exists():
            for path in phase_dir.glob(f"{role}-*.json"):
                record = json.loads(path.read_text(encoding="utf-8"))
                records.append((int(record["sequence"]), record))
        records.sort(key=lambda item: item[0])
        sequences = [sequence for sequence, _ in records]
        if sequences != list(range(len(sequences))):
            raise ValueError(f"Non-contiguous phase ledger for checkpoint role {role}")

        active = state.get("active")
        if active is not None and records:
            last = records[-1][1]
            if last.get("phase_id") == active.get("phase_id"):
                state["tracker"] = last["tracker"]
                state["active"] = None
                state["next_sequence"] = len(records)
                self._save()
        if int(state.get("next_sequence", 0)) != len(records):
            raise ValueError(f"Phase ledger/state mismatch for checkpoint role {role}")
        return [phase_from_record(record["phase"]) for _, record in records]

    def active(self, role: str) -> dict[str, Any] | None:
        value = self._role(role).get("active")
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError("Checkpoint active phase is corrupt")
        return value

    def begin_phase(
        self,
        role: str,
        label: str,
        prompt: str,
        stop_at_tokens: int,
        tracker: BudgetTracker,
    ) -> dict[str, Any]:
        state = self._role(role)
        if state.get("active") is not None:
            raise RuntimeError("Cannot begin a phase while another is active")
        active: dict[str, Any] = {
            "phase_id": uuid.uuid4().hex,
            "sequence": int(state["next_sequence"]),
            "label": label,
            "prompt": prompt,
            "stop_at_tokens": stop_at_tokens,
            "reconnect_start": len(state["reconnects"]),
            "process_resume_count": 0,
            "discarded_output_text": "",
            "discarded_tool_calls": [],
            "progress": {},
        }
        state["active"] = active
        state["tracker"] = tracker.snapshot()
        self._save()
        return active

    def begin_call(self, role: str, prompt: str) -> dict[str, Any]:
        """Mark a non-budgeted judge call active before submitting it."""
        state = self._role(role)
        if state.get("active") is not None:
            raise RuntimeError("Cannot begin a call while another is active")
        active: dict[str, Any] = {
            "phase_id": uuid.uuid4().hex,
            "sequence": int(state["next_sequence"]),
            "label": "judge",
            "prompt": prompt,
            "process_resume_count": 0,
            "discarded_output_text": "",
            "discarded_tool_calls": [],
            "progress": {},
        }
        state["active"] = active
        self._save()
        return active

    def call_result(self, role: str) -> dict[str, Any] | None:
        calls = self.data().setdefault("calls", {})
        if not isinstance(calls, dict):
            raise TypeError("Checkpoint call results are corrupt")
        value = calls.get(role)
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError("Checkpoint call result is corrupt")
        return value

    def finish_call(
        self,
        role: str,
        result: dict[str, object],
        session_id: str,
        reconnects: list[ReconnectEvent],
    ) -> None:
        """Durably store a completed judge verdict and clear its active marker."""
        state = self._role(role)
        if not isinstance(state.get("active"), dict):
            raise RuntimeError("Cannot finish a call with no active marker")
        calls = self.data().setdefault("calls", {})
        if not isinstance(calls, dict):
            raise TypeError("Checkpoint call results are corrupt")
        calls[role] = result
        state["session_id"] = session_id
        state["reconnects"] = [dataclasses.asdict(event) for event in reconnects]
        state["active"] = None
        state["next_sequence"] = int(state["next_sequence"]) + 1
        self._save()

    def prepare_process_resume(self, role: str) -> dict[str, Any]:
        """Discard an incomplete prefix while retaining its audit evidence."""
        active = self.active(role)
        if active is None:
            raise RuntimeError("No active phase to resume")
        progress = active.get("progress", {})
        if not isinstance(progress, dict):
            raise TypeError("Checkpoint phase progress is corrupt")
        parts = progress.get("text_parts", [])
        prior_text = "\n".join(str(part) for part in parts)
        accumulated = str(active.get("discarded_output_text", ""))
        active["discarded_output_text"] = "\n".join(
            part for part in (accumulated, prior_text) if part
        )
        existing_calls = active.get("discarded_tool_calls", [])
        if not isinstance(existing_calls, list):
            existing_calls = []
        active["discarded_tool_calls"] = existing_calls + [
            _tool_record(call) for call in progress_tool_calls(progress)
        ]
        active["progress"] = {}
        active["process_resume_count"] = int(active.get("process_resume_count", 0)) + 1
        self._save()
        return active

    def save_progress(
        self,
        role: str,
        tracker: BudgetTracker,
        session_id: str,
        reconnects: list[ReconnectEvent],
        progress: dict[str, object],
    ) -> None:
        state = self._role(role)
        active = state.get("active")
        if not isinstance(active, dict):
            raise RuntimeError("Received progress with no active phase")
        state["session_id"] = session_id
        state["reconnects"] = [dataclasses.asdict(event) for event in reconnects]
        state["tracker"] = tracker.snapshot()
        active["progress"] = progress
        self._save()

    def finish_phase(
        self,
        role: str,
        phase: PhaseResult,
        tracker: BudgetTracker,
        session_id: str,
        reconnects: list[ReconnectEvent],
    ) -> None:
        """Commit phase first, then advance state; restoration reconciles a gap."""
        state = self._role(role)
        active = state.get("active")
        if not isinstance(active, dict):
            raise RuntimeError("Cannot finish with no active phase")
        if phase.label != active["label"] or phase.prompt != active["prompt"]:
            raise ValueError("Completed phase does not match checkpoint controller")
        sequence = int(active["sequence"])
        record = {
            "sequence": sequence,
            "phase_id": active["phase_id"],
            "tracker": tracker.snapshot(),
            "phase": phase_record(phase),
        }
        phase_path = self.path / "phases" / f"{role}-{sequence:06d}.json"
        _atomic_json(phase_path, record)
        state["session_id"] = session_id
        state["reconnects"] = [dataclasses.asdict(event) for event in reconnects]
        state["tracker"] = tracker.snapshot()
        state["active"] = None
        state["next_sequence"] = sequence + 1
        self._save()

    def data(self) -> dict[str, Any]:
        value = self.state.setdefault("data", {})
        if not isinstance(value, dict):
            raise TypeError("Checkpoint data is corrupt")
        return value

    def save_data(self) -> None:
        self._save()

    def session_ids(self) -> dict[str, str]:
        return {
            role: str(value["session_id"])
            for role, value in self.state.get("roles", {}).items()
            if value.get("session_id")
        }

    def close(self) -> None:
        if not self._lock_handle.closed:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()

    def prepare_completion(self, result_marker: str) -> None:
        """Bind this checkpoint to its canonical relative completion marker.

        This is persisted before output finalization. Host cleanup may remove
        the checkpoint only after that marker is visible in canonical results,
        closing the crash window between a staged write and its host merge.
        """
        marker = PurePosixPath(result_marker)
        if (
            marker.is_absolute()
            or not marker.parts
            or any(part in {"", ".", ".."} for part in marker.parts)
        ):
            raise ValueError(f"Invalid relative result marker: {result_marker!r}")
        normalized = marker.as_posix()
        previous = self.state.get("completion_marker")
        if previous not in (None, normalized):
            raise ValueError("Checkpoint completion marker changed")
        self.state["completion_marker"] = normalized
        self._save()

    def clear(self) -> None:
        """Delete private state only after the canonical result is durable."""
        path = self.path
        workspace_paths: list[Path] = []
        for value in self.state.get("roles", {}).values():
            scratch_name = value.get("scratch_name")
            if not scratch_name:
                continue
            if not _SCRATCH_RE.fullmatch(str(scratch_name)):
                raise ValueError("Refusing to delete a corrupt scratch path")
            workspace_paths.append(self.root / "w" / str(scratch_name))
        self.close()
        for workspace_path in workspace_paths:
            shutil.rmtree(workspace_path, ignore_errors=True)
        shutil.rmtree(path)

    def complete(self) -> None:
        """Clear now, or defer until run-stage outputs are merged on the host."""
        if not self.state.get("completion_marker"):
            raise RuntimeError("Completion marker was not prepared")
        self.state["completed"] = True
        self._save()
        if os.environ.get(DEFER_CHECKPOINT_CLEANUP_ENV) == "1":
            return
        self.clear()
