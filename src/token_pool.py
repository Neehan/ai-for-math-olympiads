"""Round-robin pool of provider API keys with rate-limit cooldown.

The .env file may hold several keys per provider (NAME, NAME_2, NAME_3, ...).
Concurrent attempts are distributed round-robin; one key may serve several
attempts. A rate-limited key cools until its reported reset time, and acquire()
waits only when every key is cooling.
"""

import logging
import os
import re
import time

import anyio

from src.constants import (
    RATE_LIMIT_FALLBACK_COOLDOWN_SECONDS,
    RESET_WAIT_BUFFER_SECONDS,
)

log = logging.getLogger("token_pool")


class TokenPool:
    """Round-robin token dispenser with per-token rate-limit cooldowns."""

    def __init__(self, tokens: list[str], env_name: str) -> None:
        """tokens must be non-empty; order defines the round-robin rotation."""
        tokens = list(dict.fromkeys(tokens))
        if not tokens:
            raise ValueError(
                f"No API keys found; set {env_name} (and optionally "
                f"{env_name}_2, _3, ...) in .env"
            )
        self._tokens = tokens
        self._credential_labels = {
            token: f"credential_{index}"
            for index, token in enumerate(tokens, start=1)
        }
        self._cool_until: dict[str, float] = {token: 0.0 for token in tokens}
        self._next_index = 0
        self._lock = anyio.Lock()
        self._changed = anyio.Event()

    def credential_label(self, token: str) -> str:
        """Return a stable, non-secret identifier suitable for result logs."""
        return self._credential_labels[token]

    @classmethod
    def from_env(cls, env_name: str) -> "TokenPool":
        """Collect env_name and its numbered variants, sorted by name.

        Matches exactly NAME or NAME_<digits> — a loose prefix match would
        swallow unrelated vars like ..._EXPIRES_AT.
        """
        pattern = re.compile(rf"^{env_name}(_\d+)?$")
        names = sorted(
            name
            for name, value in os.environ.items()
            if pattern.match(name) and value.strip()
        )
        configured = [os.environ[name].strip() for name in names]
        # The same OAuth token under two env names is still one quota. Counting
        # aliases separately would overweight it in round-robin assignment and
        # make the reported credential count misleading.
        tokens = list(dict.fromkeys(configured))
        duplicate_count = len(configured) - len(tokens)
        log.info(
            "Token pool: %d unique key(s) from %d env var(s) (%s)",
            len(tokens),
            len(names),
            ", ".join(names),
        )
        if duplicate_count:
            log.warning(
                "Token pool: ignored %d duplicate credential value(s)",
                duplicate_count,
            )
        return cls(tokens, env_name)

    def _signal_changed_locked(self) -> None:
        """Wake acquirers after cooldown or live-token membership changes.

        Called only while holding _lock. Replacing the event before setting the
        old one prevents a fast waiter from missing a later state transition.
        """
        event = self._changed
        self._changed = anyio.Event()
        event.set()

    async def mark_dead(self, token: str) -> None:
        """Permanently remove a token (e.g. org monthly spend limit reached)."""
        async with self._lock:
            if token in self._tokens:
                self._tokens.remove(token)
                del self._cool_until[token]
                self._next_index = self._next_index % len(self._tokens) if self._tokens else 0
                self._signal_changed_locked()
        log.warning(
            "Token ...%s DISABLED (spend limit reached); %d token(s) remain",
            token[-6:],
            len(self._tokens),
        )

    async def acquire(self) -> str:
        """Return the next non-cooling token in round-robin order.

        Multiple callers may receive the same key concurrently. Fails loud
        when every token has been disabled — nothing can proceed.
        """
        while True:
            async with self._lock:
                if not self._tokens:
                    raise RuntimeError(
                        "All API tokens are disabled (spend limits reached); "
                        "raise the org limit or add fresh keys to .env"
                    )
                now = time.time()
                for offset in range(len(self._tokens)):
                    index = (self._next_index + offset) % len(self._tokens)
                    token = self._tokens[index]
                    if self._cool_until[token] <= now:
                        self._next_index = (index + 1) % len(self._tokens)
                        return token
                event = self._changed
                wake_at = min(self._cool_until.values())
            delay = max(0.0, wake_at - time.time())
            if delay > 1.0:
                log.warning(
                    "All tokens rate-limited; waiting %.0f s until "
                    "the earliest reset",
                    delay,
                )
            with anyio.move_on_after(max(0.01, delay)):
                await event.wait()

    async def release(self, token: str) -> None:
        """Compatibility no-op: keys are assigned, not exclusively leased."""
        del token

    async def mark_rate_limited(self, token: str, resets_at: int) -> None:
        """Put a token on cooldown until its reported reset time (plus buffer).

        A missing/past reset time (the SDK can report resets_at=None → 0)
        falls back to a fixed cooldown — otherwise the token would re-enter
        rotation immediately and hot-loop.
        """
        now = time.time()
        cool_until = float(resets_at + RESET_WAIT_BUFFER_SECONDS)
        if cool_until <= now:
            cool_until = now + RATE_LIMIT_FALLBACK_COOLDOWN_SECONDS
        async with self._lock:
            if token in self._cool_until:
                self._cool_until[token] = max(self._cool_until[token], cool_until)
                self._signal_changed_locked()
        log.warning(
            "Token ...%s rate-limited; cooling for %.0f min",
            token[-6:],
            (cool_until - now) / 60,
        )
