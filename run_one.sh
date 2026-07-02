#!/usr/bin/env bash
# Run EXACTLY ONE harness over every problem at the C0 baseline condition
# (no knowledge base, no oracle hints).
#
# Only one harness per invocation, BY DESIGN: running two harnesses in the same
# working tree would let the later one Read/Grep the earlier one's results/,
# logs/, and .scratch/ — cross-harness contamination that voids the experiment.
# Isolate each harness in its own git branch:
#
#   git checkout main && git checkout -b run/single_llm
#   ./run_one.sh single_llm         # then commit results, merge to main
#   git checkout main && git checkout -b run/best_of_n
#   ./run_one.sh best_of_n          # fresh tree, no trace of single_llm
#   ...
#
# Each harness is resumable: problems whose result file already exists are
# skipped, so you can re-run after an interruption. Results go to
# results/<harness>/<problem_id>.md and full audit logs to
# logs/<harness>/<problem_id>.jsonl.
#
# Usage:
#   cp .env.example .env                     # then set CLAUDE_CODE_OAUTH_TOKEN in it
#   ./run_one.sh single_llm                  # one of: single_llm | best_of_n | ralph_loop
#   PYTHON=/path/to/python ./run_one.sh ...  # force a specific interpreter
#
# Fails loud: `set -e` stops on any error.

set -euo pipefail

cd "$(dirname "$0")"

VALID_HARNESSES=("single_llm" "best_of_n" "ralph_loop")

if [ "$#" -ne 1 ]; then
    echo "ERROR: exactly one harness required (got $#)." >&2
    echo "Usage: ./run_one.sh <${VALID_HARNESSES[0]}|${VALID_HARNESSES[1]}|${VALID_HARNESSES[2]}>" >&2
    echo "Running more than one harness in the same tree contaminates the experiment;" >&2
    echo "isolate each in its own git branch. See the header of this script." >&2
    exit 2
fi

harness="$1"
ok=0
for h in "${VALID_HARNESSES[@]}"; do
    [ "$h" = "$harness" ] && ok=1
done
if [ "$ok" -ne 1 ]; then
    echo "ERROR: unknown harness '${harness}'." >&2
    echo "Must be one of: ${VALID_HARNESSES[*]}" >&2
    exit 2
fi

# Load environment (auth token etc.) from .env if present. The file is
# git-ignored, so the token is never committed. Format: KEY=value per line.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

if [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
    echo "WARNING: CLAUDE_CODE_OAUTH_TOKEN is not set (no .env and no export)." >&2
    echo "The CLI will fall back to its stored login if you have run 'claude login'." >&2
fi

# Resolve a Python >=3.12 that has the SDK installed. The plain `python3` on
# PATH is often an older framework/system build (e.g. 3.9), which lacks the
# deps, so we don't rely on it. Override with: PYTHON=/path/to/python ./run_one.sh
select_python() {
    if [ -n "${PYTHON:-}" ]; then
        echo "$PYTHON"; return 0
    fi
    for cand in python3.13 python3.12 python3; do
        if command -v "$cand" >/dev/null 2>&1 \
            && "$cand" -c 'import sys,claude_agent_sdk; sys.exit(0 if sys.version_info>=(3,12) else 1)' >/dev/null 2>&1; then
            command -v "$cand"; return 0
        fi
    done
    return 1
}

if ! PY="$(select_python)"; then
    echo "ERROR: no Python >=3.12 with claude_agent_sdk found on PATH." >&2
    echo "Install it (python3.12 -m pip install -e .) or set PYTHON=/path/to/python." >&2
    exit 1
fi

echo "=============================================================="
echo ">>> Harness:     ${harness}"
echo ">>> Interpreter: ${PY} ($("$PY" --version 2>&1))"
echo "=============================================================="
"$PY" -m "src.${harness}.run"

echo "=============================================================="
echo ">>> Done: ${harness}"
echo ">>> Results: results/${harness}/<problem_id>.md"
echo ">>> Logs:    logs/${harness}/<problem_id>.jsonl"
echo ">>> Commit these and merge to main before running the next harness."
echo "=============================================================="
