"""Selection of gradeable full and budget-cut solution artifacts."""

import unittest

from src.models import PhaseResult, ReconnectEvent
from src.storage import _phase_at_cut, _token_accounting_status, final_solution_text


def _phase(
    label: str,
    text: str,
    cumulative_tokens: int,
    *,
    interrupted: bool = False,
) -> PhaseResult:
    return PhaseResult(
        label=label,
        prompt="prompt",
        text=text,
        output_tokens=10,
        cumulative_output_tokens=cumulative_tokens,
        num_turns=1,
        duration_ms=1,
        total_cost_usd=0.01,
        is_error=False,
        stop_reason="budget_exhausted" if interrupted else "end_turn",
        budget_exhausted=interrupted,
        tool_calls=[],
        reconnects=[],
    )


class StorageSelectionTests(unittest.TestCase):
    def test_transport_recovery_is_disclosed_in_token_status(self) -> None:
        phase = _phase("solve", "proof", 100)
        phase.reconnects.append(
            ReconnectEvent("transport", None, "credential_1", "credential_1")
        )
        self.assertEqual(
            _token_accounting_status([phase], 0),
            "transport_recovered_unreported_suffix_possible",
        )
        self.assertEqual(
            _token_accounting_status([phase], 1),
            "process_and_transport_recovered_unreported_suffix_possible",
        )

    def test_full_solution_skips_empty_terminal_revision(self) -> None:
        phases = [
            _phase("solve", "first proof", 100),
            _phase("critique", "gap", 110),
            _phase("revise", "better proof", 150),
            _phase("critique", "gap", 160),
            _phase("revise", "   \n", 180),
        ]
        self.assertEqual(final_solution_text(phases, 200), "better proof")

    def test_full_solution_rejects_over_budget_terminal_phase(self) -> None:
        phases = [
            _phase("solve", "within budget", 180),
            _phase("wrap_up", "too late", 205),
        ]
        self.assertEqual(final_solution_text(phases, 200), "within budget")

    def test_full_solution_has_no_interrupted_fallback(self) -> None:
        phases = [_phase("solve", "crossing response", 205, interrupted=True)]
        with self.assertRaises(ValueError):
            final_solution_text(phases, 200)

    def test_cut_omits_empty_or_interrupted_proof_phases(self) -> None:
        phases = [
            _phase("solve", "", 100),
            _phase("revise", "partial", 150, interrupted=True),
        ]
        self.assertIsNone(_phase_at_cut(phases, 200))

    def test_cut_uses_last_nonempty_completed_proof_within_threshold(self) -> None:
        phases = [
            _phase("solve", "first", 100),
            _phase("revise", "second", 190),
            _phase("revise", "too late", 210),
        ]
        selected = _phase_at_cut(phases, 200)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected[0], 1)
        self.assertEqual(selected[1].text, "second")


if __name__ == "__main__":
    unittest.main()
