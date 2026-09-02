import unittest
from pathlib import Path

import anyio

from src.concurrency import nested_controller_limit, run_all
from src.config import load_config, override_max_concurrency


class NestedControllerLimitTests(unittest.TestCase):
    def test_two_eight_way_banks_fit_at_sixteen(self) -> None:
        self.assertEqual(nested_controller_limit(16, 8), 2)

    def test_limit_never_admits_zero_controllers(self) -> None:
        self.assertEqual(nested_controller_limit(4, 8), 1)

    def test_partial_extra_bank_is_not_admitted(self) -> None:
        self.assertEqual(nested_controller_limit(15, 8), 1)

    def test_rejects_nonpositive_limits(self) -> None:
        with self.assertRaises(ValueError):
            nested_controller_limit(0, 8)
        with self.assertRaises(ValueError):
            nested_controller_limit(16, 0)

    def test_cli_override_does_not_mutate_loaded_config(self) -> None:
        configured = load_config(Path("config.json"))
        overridden = override_max_concurrency(configured, 16)
        self.assertEqual(configured.max_concurrency, 8)
        self.assertEqual(overridden.max_concurrency, 16)
        self.assertEqual(nested_controller_limit(overridden.max_concurrency, 8), 2)

    def test_cli_override_rejects_nonpositive_value(self) -> None:
        configured = load_config(Path("config.json"))
        with self.assertRaises(ValueError):
            override_max_concurrency(configured, 0)


class NestedControllerSchedulingTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_parallel_banks_peak_at_sixteen_sessions(self) -> None:
        active = 0
        peak = 0
        lock = anyio.Lock()

        async def child() -> None:
            nonlocal active, peak
            async with lock:
                active += 1
                peak = max(peak, active)
            await anyio.sleep(0.01)
            async with lock:
                active -= 1

        async def bank() -> None:
            await run_all([child for _ in range(8)], 8)

        await run_all(
            [bank for _ in range(2)], nested_controller_limit(16, 8)
        )
        self.assertEqual(peak, 16)


if __name__ == "__main__":
    unittest.main()
