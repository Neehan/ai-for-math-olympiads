"""Bounded-concurrency fan-out over async tasks.

The Agent SDK does not parallelize on its own: each ClaudeSDKClient is one
subprocess running one conversation. To run many at once we launch them
concurrently under a global CapacityLimiter so no more than MAX_CONCURRENCY
agent sessions are in flight at any moment (bounded by API rate limits and
local resources).
"""

from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

import anyio

from src.shared.constants import MAX_CONCURRENCY

T = TypeVar("T")


async def run_bounded(
    tasks: Sequence[Callable[[], Awaitable[T]]], limit: int
) -> None:
    """Run zero-arg async task factories with at most `limit` in flight.

    Fails loud: if any task raises, anyio propagates it out of the task group.
    """
    limiter = anyio.CapacityLimiter(limit)

    async def _guarded(factory: Callable[[], Awaitable[T]]) -> None:
        async with limiter:
            await factory()

    async with anyio.create_task_group() as group:
        for factory in tasks:
            group.start_soon(_guarded, factory)


async def run_all(tasks: Sequence[Callable[[], Awaitable[T]]]) -> None:
    """Run task factories at the global MAX_CONCURRENCY limit."""
    await run_bounded(tasks, MAX_CONCURRENCY)
