import unittest

from src.concurrency import nested_controller_limit


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


if __name__ == "__main__":
    unittest.main()
