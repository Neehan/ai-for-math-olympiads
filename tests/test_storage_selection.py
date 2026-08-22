"""Selection of gradeable full and budget-cut solution artifacts."""

import json
import tempfile
import unittest
from pathlib import Path

import zstandard

from src.models import ArmConfig, ExperimentConfig, PhaseResult, ReconnectEvent
from src.storage import (
    _phase_at_cut,
    _token_accounting_status,
    all_budget_cut_multipliers,
    final_solution_text,
    materialize_budget_cut_snapshots,
)


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
    def test_dense_snapshots_are_recovered_from_phase_log(self) -> None:
        arm = ArmConfig("baseline-sequential", "none", "sequential", 8, [1])
        config = ExperimentConfig(
            model="solver",
            audit_model="judge",
            effort="high",
            unit_output_tokens=200_000,
            wrap_up_reserve_tokens=20_000,
            uniform_strategy_plan_tokens=80_000,
            uniform_strategy_plan_wrap_up_reserve_tokens=40_000,
            uniform_strategy_branches=8,
            max_turns_per_phase=128,
            audit_max_turns=64,
            max_concurrency=8,
            arms={arm.name: arm},
        )
        records = [
            {
                "label": "solve",
                "text": "proof A",
                "cumulative_output_tokens": 150_000,
                "budget_exhausted": False,
            },
            {
                "label": "critique",
                "text": "not a proof",
                "cumulative_output_tokens": 220_000,
                "budget_exhausted": False,
            },
            {
                "label": "revise",
                "text": "proof B",
                "cumulative_output_tokens": 350_000,
                "budget_exhausted": False,
            },
            {
                "label": "revise",
                "text": "interrupted proof",
                "cumulative_output_tokens": 550_000,
                "budget_exhausted": True,
            },
            {
                "label": "revise",
                "text": "proof C",
                "cumulative_output_tokens": 610_000,
                "budget_exhausted": False,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            log_text = "\n".join(json.dumps(record) for record in records) + "\n"
            (output / "logs.jsonl.zst").write_bytes(
                zstandard.ZstdCompressor().compress(log_text.encode("utf-8"))
            )
            (output / "solution_1x.md").write_text("proof A\n", encoding="utf-8")

            materialize_budget_cut_snapshots(
                config, arm, output, all_budget_cut_multipliers(8)
            )

            expected = {
                1: "proof A\n",
                2: "proof B\n",
                3: "proof B\n",
                4: "proof C\n",
                5: "proof C\n",
                6: "proof C\n",
                7: "proof C\n",
            }
            for multiplier, text in expected.items():
                self.assertEqual(
                    (output / f"solution_{multiplier}x.md").read_text(
                        encoding="utf-8"
                    ),
                    text,
                )

    def test_recovery_is_valid_accounted_output_with_separate_provenance(self) -> None:
        phase = _phase("solve", "proof", 100)
        phase.reconnects.append(
            ReconnectEvent("transport", None, "credential_1", "credential_1")
        )
        self.assertEqual(
            _token_accounting_status([phase], 0),
            "recovered_eligible_output_accounted",
        )
        self.assertEqual(
            _token_accounting_status([phase], 1),
            "recovered_eligible_output_accounted",
        )

    def test_unrecovered_output_uses_provider_complete_status(self) -> None:
        self.assertEqual(
            _token_accounting_status([_phase("solve", "proof", 100)], 0),
            "provider_reported_complete",
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
