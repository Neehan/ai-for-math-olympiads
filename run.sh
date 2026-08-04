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
# The run stage mounts a staging dir holding only meta.json resume markers
# (never prior solutions/logs), merged back into results/ after the container
# exits; the audit stage mounts the full tree (the judge reads solutions).
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
merge_staging() {
    if [ -n "$STAGING" ] && [ -d "$STAGING" ]; then
        rsync -a "$STAGING"/ "$PWD/results"/
        rm -rf "$STAGING"
    fi
}
trap merge_staging EXIT

if [ "$1" = "run" ]; then
    STAGING=$(mktemp -d "$PWD/.results-staging.XXXXXX")
    rsync -a --include='*/' --include='meta.json' --exclude='*' \
        "$PWD/results"/ "$STAGING"/
    RESULTS_MOUNT="$STAGING"
fi

docker run --rm --cap-add=NET_ADMIN \
    "${token_args[@]}" \
    -v "$PWD/prompts:/app/prompts:ro" \
    -v "$PWD/config.json:/app/config.json:ro" \
    -v "$PWD/agent_settings.json:/app/agent_settings.json:ro" \
    -v "$RESULTS_MOUNT:/app/results" \
    "$IMAGE" "$@"
