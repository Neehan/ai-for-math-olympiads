#!/usr/bin/env bash
# Build the image and run ONE harness stage (run = generation, audit =
# grading) in a throwaway container. Both stages sit behind the same egress
# firewall — a judge with internet could fetch official solutions.
#
# The container is removed on exit (--rm); results land in ./results/ via the
# bind mount. Problems and hints are fetched from the dataset URLs by the
# entrypoint (never stored in the image); prompts/, config.json, and
# agent_settings.json are mounted read-only so editing them needs no rebuild.
#
# The run stage mounts a staging dir holding only the selected model/arm's
# meta.json resume markers (never prior solutions/logs). On exit, only newly
# completed attempts from that same model/arm are merged into results/; the
# audit stage mounts the full tree (the judge reads solutions).
#
# Usage:
#   cp .env.example .env             # set CLAUDE_CODE_OAUTH_TOKEN in it
#   ./run.sh run --arm baseline
#   ./run.sh run --arm baseline --domain combinatorics
#   ./run.sh audit --arm baseline
set -euo pipefail

cd "$(dirname "$0")"

if [ $# -lt 3 ] || { [ "$1" != "run" ] && [ "$1" != "audit" ]; }; then
    echo "Usage: ./run.sh <run|audit> --arm <ARM> [--problems id1,id2] [--domain d]" >&2
    exit 2
fi

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi
if [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
    echo "ERROR: CLAUDE_CODE_OAUTH_TOKEN is not set (put it in .env)." >&2
    exit 1
fi

IMAGE=olympiad-harness
docker build -q -t "$IMAGE" -f docker/Dockerfile . >/dev/null
mkdir -p results

# Pass every provider key var through (round-robin token pools).
token_args=()
while IFS= read -r name; do
    token_args+=(-e "$name")
done < <(compgen -v | grep -E '^(CLAUDE_CODE_OAUTH_TOKEN|OPENROUTER_API_KEY)')

RESULTS_MOUNT="$PWD/results"
STAGING=""
STAGING_ARM_ROOT=""
RESULTS_ARM_ROOT=""
merge_staging() {
    if [ -n "$STAGING" ] && [ -d "$STAGING" ]; then
        if [ -n "$STAGING_ARM_ROOT" ] && [ -d "$STAGING_ARM_ROOT" ]; then
            # Pre-seeded meta.json files are resume inputs, not outputs. Never
            # copy a marker-only directory back: a concurrent older arm may
            # otherwise resurrect results that were archived or cleared while
            # it was running. A real completed attempt also has solution.md.
            while IFS= read -r -d '' meta_file; do
                seed_dir="${meta_file%/meta.json}"
                if [ ! -f "$seed_dir/solution.md" ]; then
                    rm -- "$meta_file"
                fi
            done < <(find "$STAGING_ARM_ROOT" -type f -name meta.json -print0)
            find "$STAGING_ARM_ROOT" -depth -type d -empty -exec rmdir {} \;

            if [ -d "$STAGING_ARM_ROOT" ]; then
                mkdir -p "$RESULTS_ARM_ROOT"
                rsync -a "$STAGING_ARM_ROOT"/ "$RESULTS_ARM_ROOT"/
            fi
        fi
        rm -rf "$STAGING"
    fi
}
trap merge_staging EXIT

if [ "$1" = "run" ]; then
    run_arm=""
    run_model=""
    cli_args=("$@")
    for ((i = 0; i < ${#cli_args[@]}; i++)); do
        case "${cli_args[i]}" in
            --arm)
                if ((i + 1 >= ${#cli_args[@]})); then
                    echo "ERROR: --arm requires a value" >&2
                    exit 2
                fi
                run_arm="${cli_args[i + 1]}"
                i=$((i + 1))
                ;;
            --arm=*) run_arm="${cli_args[i]#--arm=}" ;;
            --model)
                if ((i + 1 >= ${#cli_args[@]})); then
                    echo "ERROR: --model requires a value" >&2
                    exit 2
                fi
                run_model="${cli_args[i + 1]}"
                i=$((i + 1))
                ;;
            --model=*) run_model="${cli_args[i]#--model=}" ;;
        esac
    done
    if [ -z "$run_arm" ] || [[ "$run_arm" == */* ]] || [ "$run_arm" = "." ] || [ "$run_arm" = ".." ]; then
        echo "ERROR: run requires a filesystem-safe --arm value" >&2
        exit 2
    fi
    if [ -z "$run_model" ]; then
        run_model="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["model"])' config.json)"
    fi
    run_model_dir="${run_model//\//-}"
    if [ -z "$run_model_dir" ] || [ "$run_model_dir" = "." ] || [ "$run_model_dir" = ".." ]; then
        echo "ERROR: effective model does not map to a safe results directory" >&2
        exit 2
    fi

    STAGING=$(mktemp -d "$PWD/.results-staging.XXXXXX")
    RESULTS_ARM_ROOT="$PWD/results/$run_model_dir/$run_arm"
    STAGING_ARM_ROOT="$STAGING/$run_model_dir/$run_arm"
    mkdir -p "$STAGING_ARM_ROOT"
    if [ -d "$RESULTS_ARM_ROOT" ]; then
        rsync -a --include='*/' --include='meta.json' --exclude='*' \
            "$RESULTS_ARM_ROOT"/ "$STAGING_ARM_ROOT"/
    fi
    RESULTS_MOUNT="$STAGING"
fi

docker run --rm --cap-add=NET_ADMIN \
    "${token_args[@]}" \
    -v "$PWD/prompts:/app/prompts:ro" \
    -v "$PWD/config.json:/app/config.json:ro" \
    -v "$PWD/agent_settings.json:/app/agent_settings.json:ro" \
    -v "$RESULTS_MOUNT:/app/results" \
    "$IMAGE" "$@"
