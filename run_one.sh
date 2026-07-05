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
#   ./run_one.sh --reset single_llm          # wipe this harness's outputs first, then run
#   PYTHON=/path/to/python ./run_one.sh ...  # force a specific interpreter
#
# --reset deletes results/<harness>/, logs/<harness>/, and .scratch/<harness>/
# before running, so the run starts from a clean slate (used when the prompt or
# harness changed and old outputs must not be reused by the resumable skip).
#
# Fails loud: `set -e` stops on any error.

set -euo pipefail

cd "$(dirname "$0")"

VALID_HARNESSES=("single_llm" "best_of_n" "ralph_loop" "self_refine")

reset=0
positional=()
for arg in "$@"; do
    case "$arg" in
        --reset) reset=1 ;;
        -*)
            echo "ERROR: unknown flag '${arg}'. Only --reset is supported." >&2
            exit 2
            ;;
        *) positional+=("$arg") ;;
    esac
done

if [ "${#positional[@]}" -ne 1 ]; then
    echo "ERROR: exactly one harness required (got ${#positional[@]})." >&2
    valid_joined=$(IFS='|'; echo "${VALID_HARNESSES[*]}")
    echo "Usage: ./run_one.sh [--reset] <${valid_joined}>" >&2
    echo "Running more than one harness in the same tree contaminates the experiment;" >&2
    echo "isolate each in its own git branch. See the header of this script." >&2
    exit 2
fi

harness="${positional[0]}"
ok=0
for h in "${VALID_HARNESSES[@]}"; do
    [ "$h" = "$harness" ] && ok=1
done
if [ "$ok" -ne 1 ]; then
    echo "ERROR: unknown harness '${harness}'." >&2
    echo "Must be one of: ${VALID_HARNESSES[*]}" >&2
    exit 2
fi

# --reset: wipe this harness's outputs so the resumable skip can't reuse stale
# results generated under an old prompt/harness. Only this harness's dirs are
# touched, never another harness's or the repo root.
if [ "$reset" -eq 1 ]; then
    echo ">>> --reset: clearing results/${harness}/, logs/${harness}/, .scratch/${harness}/"
    rm -rf "results/${harness}" "logs/${harness}" ".scratch/${harness}"
    echo ">>> reset done."
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

# Pin the agent's compute environment. The agent runs Bash in a subprocess that
# inherits this shell's PATH, so `python`/`python3` it invokes must resolve to
# the SAME interpreter we launch the harness with — the one that has the SDK and
# the math libs. Otherwise a stale `python3` on the caller's PATH could hand the
# agent a numpy-less interpreter, silently changing the compute environment
# between runs (a confound). Prepending $PY's dir makes both names resolve to it.
PY_DIR="$(cd "$(dirname "$PY")" && pwd)"
export PATH="${PY_DIR}:${PATH}"

# Preflight: the math libs must import under BOTH `python` and `python3` (the two
# names the agent may call) before we spend a cent on the API. Fail loud here
# rather than let the agent silently degrade to no-numpy midway through a run.
for name in python python3; do
    if ! command -v "$name" >/dev/null 2>&1; then
        echo "ERROR: '$name' not found on PATH after pinning ${PY_DIR}." >&2
        exit 1
    fi
    if ! "$name" -c 'import numpy, sympy, scipy, mpmath' >/dev/null 2>&1; then
        echo "ERROR: '$name' cannot import numpy/sympy/scipy/mpmath." >&2
        echo "The agent's Bash would see a compute environment missing math libs." >&2
        echo "Install them into the interpreter at ${PY_DIR} (python -m pip install -e .)." >&2
        exit 1
    fi
done

echo "=============================================================="
echo ">>> Harness:     ${harness}"
echo ">>> Interpreter: ${PY} ($("$PY" --version 2>&1))"
echo ">>> PATH pinned: ${PY_DIR} (python & python3 -> this; numpy/sympy/scipy OK)"
echo "=============================================================="
"$PY" -m "src.${harness}.run"

echo "=============================================================="
echo ">>> Done: ${harness}"
echo ">>> Results: results/${harness}/<problem_id>.md"
echo ">>> Logs:    logs/${harness}/<problem_id>.jsonl"
echo ">>> Commit these and merge to main before running the next harness."
echo "=============================================================="
