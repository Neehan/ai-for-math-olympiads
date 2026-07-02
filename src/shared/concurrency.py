"""Bounded-concurrency fan-out over async tasks.

The Agent SDK does not parallelize on its own: each ClaudeSDKClient is one
subprocess running one conversation. To run many at once we launch them
concurrently under a global CapacityLimiter so no more than MAX_CONCURRENCY
agent sessions are in flight at any moment (bounded by API rate limits and
local resources).

Each task is isolated: if one raises, it is logged and the others keep running,
rather than one failure cancelling the whole batch (a 39-problem run must not
die because one problem errored). A failed task simply produces no result, so
the affected problem is picked up again on the next resumable run.
"""

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

import anyio

from src.shared.constants import MAX_CONCURRENCY

log = logging.getLogger("concurrency")

T = TypeVar("T")


async def run_all(tasks: Sequence[Callable[[], Awaitable[T]]]) -> None:
    """Run zero-arg async task factories, at most MAX_CONCURRENCY in flight.

    A task that raises is logged and skipped; siblings are unaffected.
    """
    limiter = anyio.CapacityLimiter(MAX_CONCURRENCY)

    async def _guarded(factory: Callable[[], Awaitable[T]]) -> None:
        async with limiter:
            try:
                await factory()
            except Exception:
                log.exception("Task failed; continuing with remaining tasks")

    async with anyio.create_task_group() as group:
        for factory in tasks:
            group.start_soon(_guarded, factory)
