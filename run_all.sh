#!/usr/bin/env bash
# Run all three harnesses (Single-LLM, Best-of-N, Ralph-loop) over every problem
# at the C0 baseline condition (no knowledge base, no oracle hints).
#
# Each harness is resumable: problems whose result file already exists are
# skipped, so you can re-run this after an interruption. Results go to
# results/<harness>/<problem_id>.md and full audit logs to
# logs/<harness>/<problem_id>.jsonl.
#
# Usage:
#   export ANTHROPIC_API_KEY=...        # or rely on Claude Code auth
#   ./run_all.sh                        # run all three, in order
#   ./run_all.sh single_llm             # run only one harness
#
# Fails loud: `set -e` stops the whole run if any harness errors.

set -euo pipefail

cd "$(dirname "$0")"

HARNESSES=("single_llm" "best_of_n" "ralph_loop")

if [ "$#" -gt 0 ]; then
    HARNESSES=("$@")
fi

for harness in "${HARNESSES[@]}"; do
    echo "=============================================================="
    echo ">>> Running harness: ${harness}"
    echo "=============================================================="
    python3 -m "src.${harness}.run"
done

echo "=============================================================="
echo ">>> All runs complete."
echo ">>> Results: results/<harness>/<problem_id>.md"
echo ">>> Logs:    logs/<harness>/<problem_id>.jsonl"
echo "=============================================================="
