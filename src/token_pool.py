"""Round-robin pool of Claude OAuth tokens with rate-limit cooldown.

The .env file may hold several subscription tokens (CLAUDE_CODE_OAUTH_TOKEN,
CLAUDE_CODE_OAUTH_TOKEN_2, ...). Attempts acquire tokens round-robin; a token
whose account hits its usage limit is put on cooldown until its reported reset
time, and acquire() only sleeps when EVERY token is cooling — so a rate limit
rotates to the next key instead of stalling (or killing) the run.
"""

import logging
import os
import re
import time

import anyio

from src.constants import (
    OAUTH_TOKEN_ENV,
    RATE_LIMIT_FALLBACK_COOLDOWN_SECONDS,
    RESET_WAIT_BUFFER_SECONDS,
)

log = logging.getLogger("token_pool")

# Exactly CLAUDE_CODE_OAUTH_TOKEN or a numbered variant (_2, _3, ...) — a
# loose prefix match would swallow unrelated vars like ..._EXPIRES_AT.
_TOKEN_VAR = re.compile(rf"^{OAUTH_TOKEN_ENV}(_\d+)?$")


class TokenPool:
    """Round-robin token dispenser with per-token rate-limit cooldowns."""

    def __init__(self, tokens: list[str]) -> None:
        """tokens must be non-empty; order defines the round-robin rotation."""
        if not tokens:
            raise ValueError(
                f"No OAuth tokens found; set {OAUTH_TOKEN_ENV} (and optionally "
                f"{OAUTH_TOKEN_ENV}_2, _3, ...) in .env"
            )
        self._tokens = tokens
        self._cool_until: dict[str, float] = {token: 0.0 for token in tokens}
        self._next_index = 0
        self._lock = anyio.Lock()

    @classmethod
    def from_env(cls) -> "TokenPool":
        """Collect CLAUDE_CODE_OAUTH_TOKEN and numbered variants, sorted by name."""
        names = sorted(
            name
            for name, value in os.environ.items()
            if _TOKEN_VAR.match(name) and value.strip()
        )
        tokens = [os.environ[name].strip() for name in names]
        log.info("Token pool: %d token(s) loaded (%s)", len(tokens), ", ".join(names))
        return cls(tokens)

    async def acquire(self) -> str:
        """Return the next available token, sleeping only if all are cooling."""
        while True:
            async with self._lock:
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
            "Token ...%s rate-limited; cooling until unix %.0f", token[-6:], cool_until
        )
