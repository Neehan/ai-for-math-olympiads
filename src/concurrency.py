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
    After each task finishes (either way), a [done/total] progress line is
    logged so long runs show how much work remains.
    """
    limiter = anyio.CapacityLimiter(limit)
    total = len(tasks)
    finished = 0

    async def _guarded(factory: Callable[[], Awaitable[T]]) -> None:
        nonlocal finished
        async with limiter:
            try:
                await factory()
            except Exception:
                log.exception("Task failed; continuing with remaining tasks")
            finished += 1
            log.info("[%d/%d] tasks finished", finished, total)

    async with anyio.create_task_group() as group:
        for factory in tasks:
            group.start_soon(_guarded, factory)


def nested_controller_limit(max_concurrency: int, child_width: int) -> int:
    """Number of child fan-out controllers that may run concurrently.

    Each controller can launch at most ``child_width`` sessions. Using the
    integer quotient keeps their combined peak at or below the configured
    global capacity. A single controller remains runnable when its fixed bank
    is wider than that capacity; its own limiter enforces the actual cap.
    """
    if max_concurrency < 1 or child_width < 1:
        raise ValueError("concurrency limits must be positive")
    return max(1, max_concurrency // child_width)
