"""Conservative raw-transcript recovery for empty SDK result text."""

import json
import tempfile
import unittest
from pathlib import Path

from src.recover_empty_phase_text import finalized_solutions


class RecoverEmptyPhaseTextTests(unittest.TestCase):
    def test_only_finalized_proofs_are_recoverable(self) -> None:
        records = [
            {
                "type": "assistant",
                "message": {
                    "stop_reason": "max_tokens",
                    "content": [{"type": "text", "text": "## Final Solution\npartial"}],
                },
            },
            {
                "type": "assistant",
                "message": {
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "ordinary commentary"}],
                },
            },
            {
                "type": "assistant",
                "message": {
                    "stop_reason": "end_turn",
                    "content": [
                        {"type": "text", "text": "## Final Solution\ncomplete"}
                    ],
                },
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            self.assertEqual(
                finalized_solutions(path), ["## Final Solution\ncomplete"]
            )


if __name__ == "__main__":
    unittest.main()
