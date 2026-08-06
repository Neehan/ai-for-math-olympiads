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
# (never prior solutions/logs). On exit, only newly completed attempts are
# merged into results/; the audit stage mounts the full tree (the judge reads
# solutions).
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

# Mount only this stage/model/arm's opaque checkpoint namespace.  Mounting the
# whole bank would let a tool-using solver inspect interrupted work from a
# different intervention arm.  The namespace is stable across reruns but its
# host name reveals no experiment identity inside the container.
ARM_NAME=""
MODEL_NAME=$(python -c 'import json; print(json.load(open("config.json"))["model"])')
AUDIT_MODEL_NAME=$(python -c 'import json; print(json.load(open("config.json"))["audit_model"])')
EXPECT_VALUE=""
for ARGUMENT in "$@"; do
    if [ "$EXPECT_VALUE" = "arm" ]; then
        ARM_NAME="$ARGUMENT"
        EXPECT_VALUE=""
        continue
    fi
    if [ "$EXPECT_VALUE" = "model" ]; then
        MODEL_NAME="$ARGUMENT"
        EXPECT_VALUE=""
        continue
    fi
    if [ "$EXPECT_VALUE" = "audit_model" ]; then
        AUDIT_MODEL_NAME="$ARGUMENT"
        EXPECT_VALUE=""
        continue
    fi
    case "$ARGUMENT" in
        --arm) EXPECT_VALUE="arm" ;;
        --model) EXPECT_VALUE="model" ;;
        --audit-model) EXPECT_VALUE="audit_model" ;;
        --arm=*) ARM_NAME=${ARGUMENT#--arm=} ;;
        --model=*) MODEL_NAME=${ARGUMENT#--model=} ;;
        --audit-model=*) AUDIT_MODEL_NAME=${ARGUMENT#--audit-model=} ;;
    esac
done
if [ -z "$ARM_NAME" ]; then
    echo "ERROR: --arm requires a value" >&2
    exit 2
fi
CHECKPOINT_NAMESPACE=$(python -c \
    'import hashlib,pathlib,sys; h=hashlib.sha256("\0".join(sys.argv[1:]).encode()); files=[pathlib.Path("config.json"),pathlib.Path("agent_settings.json"),*sorted(pathlib.Path("prompts").glob("*.md"))]; [h.update(p.name.encode()+b"\0"+p.read_bytes()+b"\0") for p in files]; print(h.hexdigest()[:24])' \
    "$1" "$MODEL_NAME" "$AUDIT_MODEL_NAME" "$ARM_NAME")
CHECKPOINT_MOUNT="$PWD/.session-checkpoints/runtime/$CHECKPOINT_NAMESPACE"
mkdir -p "$CHECKPOINT_MOUNT"
chmod 700 .session-checkpoints .session-checkpoints/runtime "$CHECKPOINT_MOUNT"

# Pass every provider key var through (round-robin token pools).
token_args=()
while IFS= read -r name; do
    token_args+=(-e "$name")
done < <(compgen -v | grep -E '^(CLAUDE_CODE_OAUTH_TOKEN|OPENROUTER_API_KEY)')

RESULTS_MOUNT="$PWD/results"
STAGING=""
cleanup_completed_checkpoints() {
    if [ ! -d "$CHECKPOINT_MOUNT/attempts" ]; then
        return
    fi
    python src/cleanup_checkpoints.py "$CHECKPOINT_MOUNT" "$PWD/results"
}
merge_staging() {
    if [ -n "$STAGING" ] && [ -d "$STAGING" ]; then
        # A completed attempt has both files: solution.md is written first and
        # meta.json last. Pre-seeded resume markers have no solution, so never
        # merge them back; doing so could resurrect results archived while a
        # different arm was running. Partial writes are also excluded.
        shopt -s nullglob
        for seed_dir in "$STAGING"/*/*/*/seed_*; do
            if [ ! -f "$seed_dir/meta.json" ] || [ ! -f "$seed_dir/solution.md" ]; then
                rm -rf "$seed_dir"
            fi
        done
        shopt -u nullglob
        find "$STAGING" -mindepth 1 -depth -type d -empty -exec rmdir {} \;
        rsync -a "$STAGING"/ "$PWD/results"/
        rm -rf "$STAGING"
    fi
}
finish_stage() {
    merge_staging
    cleanup_completed_checkpoints
}
trap finish_stage EXIT

if [ "$1" = "run" ]; then
    STAGING=$(mktemp -d "$PWD/.results-staging.XXXXXX")
    rsync -a --include='*/' --include='meta.json' --exclude='*' \
        "$PWD/results"/ "$STAGING"/
    RESULTS_MOUNT="$STAGING"
fi

checkpoint_args=()
if [ "$1" = "run" ]; then
    checkpoint_args=(-e HARNESS_DEFER_CHECKPOINT_CLEANUP=1)
fi

docker run --rm --cap-add=NET_ADMIN \
    "${token_args[@]}" \
    "${checkpoint_args[@]}" \
    -v "$PWD/prompts:/app/prompts:ro" \
    -v "$PWD/config.json:/app/config.json:ro" \
    -v "$PWD/agent_settings.json:/app/agent_settings.json:ro" \
    -v "$RESULTS_MOUNT:/app/results" \
    -v "$CHECKPOINT_MOUNT:/c" \
    -e HARNESS_CHECKPOINT_ROOT=/c \
    "$IMAGE" "$@"
