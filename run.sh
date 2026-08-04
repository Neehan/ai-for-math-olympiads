#!/usr/bin/env bash
# Build the image and run ONE experiment arm in a throwaway container.
#
# The container is removed on exit (--rm); results land in ./results/ via the
# bind mount. problems.jsonl, prompts/, config.json, and agent_settings.json
# are mounted read-only so editing them needs no image rebuild.
#
# Usage:
#   cp .env.example .env             # set CLAUDE_CODE_OAUTH_TOKEN in it
#   ./run.sh --arm baseline
#   ./run.sh --arm hint --problems usamo-2026-3,china-2026-5
set -euo pipefail

cd "$(dirname "$0")"

if [ $# -lt 2 ]; then
    echo "Usage: ./run.sh --arm <ARM> [--problems id1,id2,...]" >&2
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

# Pass every CLAUDE_CODE_OAUTH_TOKEN* var through (round-robin token pool).
token_args=()
while IFS= read -r name; do
    token_args+=(-e "$name")
done < <(compgen -v | grep '^CLAUDE_CODE_OAUTH_TOKEN')

exec docker run --rm --cap-add=NET_ADMIN \
    "${token_args[@]}" \
    -v "$PWD/prompts:/app/prompts:ro" \
    -v "$PWD/config.json:/app/config.json:ro" \
    -v "$PWD/agent_settings.json:/app/agent_settings.json:ro" \
    -v "$PWD/results:/app/results" \
    "$IMAGE" "$@"
