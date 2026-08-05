"""Round-robin pool of provider API keys with rate-limit cooldown.

The .env file may hold several keys per provider (NAME, NAME_2, NAME_3, ...).
Attempts acquire keys round-robin; a rate-limited key cools down until its
reported reset time, and acquire() only sleeps when EVERY key is cooling.
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
        if not tokens:
            raise ValueError(
                f"No API keys found; set {env_name} (and optionally "
                f"{env_name}_2, _3, ...) in .env"
            )
        self._tokens = tokens
        self._cool_until: dict[str, float] = {token: 0.0 for token in tokens}
        self._next_index = 0
        self._lock = anyio.Lock()

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
        tokens = [os.environ[name].strip() for name in names]
        log.info("Token pool: %d key(s) loaded (%s)", len(tokens), ", ".join(names))
        return cls(tokens, env_name)

    async def mark_dead(self, token: str) -> None:
        """Permanently remove a token (e.g. org monthly spend limit reached)."""
        async with self._lock:
            if token in self._tokens:
                self._tokens.remove(token)
                del self._cool_until[token]
                self._next_index = self._next_index % len(self._tokens) if self._tokens else 0
        log.warning(
            "Token ...%s DISABLED (spend limit reached); %d token(s) remain",
            token[-6:],
            len(self._tokens),
        )

    async def acquire(self) -> str:
        """Return the next available token, sleeping only if all are cooling.

        Fails loud when every token has been disabled — nothing can proceed.
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
                wake_at = min(self._cool_until.values())
            delay = max(1.0, wake_at - time.time())
            log.warning(
                "All %d tokens rate-limited; sleeping %.0f s until earliest reset",
                len(self._tokens),
                delay,
            )
            await anyio.sleep(delay)

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
            self._cool_until[token] = max(self._cool_until[token], cool_until)
        log.warning(
            "Token ...%s rate-limited; cooling for %.0f min",
            token[-6:],
            (cool_until - now) / 60,
        )
