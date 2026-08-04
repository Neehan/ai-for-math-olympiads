"""Bounded-concurrency fan-out over async tasks.

Each ClaudeSDKClient is one subprocess running one conversation; to run many
at once we launch them under a CapacityLimiter so no more than the configured
number of agent sessions are in flight at any moment.

Each task is isolated: if one raises, it is logged and the others keep
running. A failed task produces no completion marker, so the affected attempt
is picked up again on the next resumable run.
"""

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

import anyio

log = logging.getLogger("concurrency")

T = TypeVar("T")


async def run_all(tasks: Sequence[Callable[[], Awaitable[T]]], limit: int) -> None:
    """Run zero-arg async task factories, at most `limit` in flight.

    A task that raises is logged and skipped; siblings are unaffected.
    """
    limiter = anyio.CapacityLimiter(limit)

    async def _guarded(factory: Callable[[], Awaitable[T]]) -> None:
        async with limiter:
            try:
                await factory()
            except Exception:
                log.exception("Task failed; continuing with remaining tasks")

    async with anyio.create_task_group() as group:
        for factory in tasks:
            group.start_soon(_guarded, factory)
