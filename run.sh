#!/usr/bin/env bash
# Build the image and run the generation or audit pipeline in throwaway
# containers behind the same egress firewall. For a sequential arm, one
# public `audit` command first grades correctness, then launches a separate
# internal container for route-state annotation. The separation ensures the
# correctness judge can never inspect the reference solutions used later.
#
# The container is removed on exit (--rm); results land in ./results/ via the
# bind mount. Problems and hints are fetched from the dataset URLs by the
# entrypoint (never stored in the image); prompts/, config.json, and
# agent settings profiles are mounted read-only so editing them needs no rebuild.
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
IMAGE=olympiad-harness
docker build -q -t "$IMAGE" -f docker/Dockerfile . >/dev/null
mkdir -p results state-results

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

if [ "$1" = "run" ]; then
    ACTIVE_MODEL=$MODEL_NAME
else
    ACTIVE_MODEL=$AUDIT_MODEL_NAME
fi
network_args=()
ACTIVE_AGENT_SETTINGS=agent_settings.json
case "$ACTIVE_MODEL" in
    vllm/*)
        ACTIVE_AGENT_SETTINGS=agent_settings_small.json
        if [ -z "${VLLM_API_KEY:-}" ] || [ -z "${VLLM_BASE_URL:-}" ]; then
            echo "ERROR: $ACTIVE_MODEL requires VLLM_API_KEY and VLLM_BASE_URL in .env." >&2
            exit 1
        fi
        # Docker Desktop defines this name itself; overriding it on macOS
        # breaks the Desktop VM route. Native Linux needs the explicit alias.
        if [ "$(uname -s)" != "Darwin" ]; then
            network_args=(--add-host host.docker.internal:host-gateway)
        fi
        TOKEN_PATTERN='^(VLLM_API_KEY|VLLM_BASE_URL(_[0-9]+)?)$'
        PROVIDER_KIND=vllm
        ;;
    litellm/*)
        if [ -z "${LITELLM_API_KEY:-}" ] || [ -z "${LITELLM_BASE_URL:-}" ]; then
            echo "ERROR: $ACTIVE_MODEL requires LITELLM_API_KEY and LITELLM_BASE_URL in .env." >&2
            exit 1
        fi
        LITELLM_NETWORK=${CODEX_LITELLM_NETWORK:-olympiad-codex-litellm}
        if ! docker network inspect "$LITELLM_NETWORK" >/dev/null 2>&1; then
            echo "ERROR: LiteLLM network '$LITELLM_NETWORK' is absent; start the pool first." >&2
            exit 1
        fi
        network_args=(--network "$LITELLM_NETWORK")
        TOKEN_PATTERN='^(LITELLM_API_KEY|LITELLM_BASE_URL)(_[0-9]+)?$'
        PROVIDER_KIND=litellm
        ;;
    */*)
        if [ -z "${OPENROUTER_API_KEY:-}" ]; then
            echo "ERROR: $ACTIVE_MODEL requires OPENROUTER_API_KEY in .env." >&2
            exit 1
        fi
        TOKEN_PATTERN='^OPENROUTER_API_KEY(_[0-9]+)?$'
        PROVIDER_KIND=openrouter
        ;;
    *)
        if [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
            echo "ERROR: $ACTIVE_MODEL requires CLAUDE_CODE_OAUTH_TOKEN in .env." >&2
            exit 1
        fi
        TOKEN_PATTERN='^CLAUDE_CODE_OAUTH_TOKEN(_[0-9]+)?$'
        PROVIDER_KIND=anthropic
        ;;
esac

CHECKPOINT_NAMESPACE=$(python -c \
    'import hashlib,pathlib,sys; h=hashlib.sha256("\0".join(sys.argv[1:5]).encode()); prompts=sorted(pathlib.Path("prompts").glob("*.md")); prompts=[p for p in prompts if p.name!="state_audit.md"]; files=[pathlib.Path("config.json"),pathlib.Path(sys.argv[5]),*prompts]; [h.update(p.name.encode()+b"\0"+p.read_bytes()+b"\0") for p in files]; print(h.hexdigest()[:24])' \
    "$1" "$MODEL_NAME" "$AUDIT_MODEL_NAME" "$ARM_NAME" "$ACTIVE_AGENT_SETTINGS")
CHECKPOINT_MOUNT="$PWD/.session-checkpoints/runtime/$CHECKPOINT_NAMESPACE"
mkdir -p "$CHECKPOINT_MOUNT"
chmod 700 .session-checkpoints .session-checkpoints/runtime "$CHECKPOINT_MOUNT"

# Pass every provider key/endpoint var through (round-robin token pools).
token_args=()
while IFS= read -r name; do
    token_args+=(-e "$name")
done < <(compgen -v | grep -E "$TOKEN_PATTERN")

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
        # meta.json is written last. Drop pre-seeded markers and partial writes,
        # but retain completed run_<kk> children from an interrupted bank so a
        # restart spends only on its unfinished members.
        python scripts/prune_staging.py "$STAGING"
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

# Bash 3.2 + nounset rejects "${empty_array[@]}"; the guarded form emits zero
# arguments when an optional array is empty.
docker run --rm --cap-add=NET_ADMIN \
    ${token_args[@]+"${token_args[@]}"} \
    ${checkpoint_args[@]+"${checkpoint_args[@]}"} \
    ${network_args[@]+"${network_args[@]}"} \
    -v "$PWD/prompts:/app/prompts:ro" \
    -v "$PWD/config.json:/app/config.json:ro" \
    -v "$PWD/agent_settings.json:/app/agent_settings.json:ro" \
    -v "$PWD/agent_settings_small.json:/app/agent_settings_small.json:ro" \
    -v "$RESULTS_MOUNT:/app/results" \
    -v "$CHECKPOINT_MOUNT:/c" \
    -e HARNESS_CHECKPOINT_ROOT=/c \
    -e "HARNESS_PROVIDER_KIND=$PROVIDER_KIND" \
    -e "HARNESS_OPENROUTER_ALLOWED_MODEL=$ACTIVE_MODEL" \
    "$IMAGE" "$@"

# State annotation is part of the public audit pipeline for sequential arms,
# but runs in a fresh container whose dataset includes reference solutions.
if [ "$1" = "audit" ] && \
    { [ "$ARM_NAME" = "baseline-sequential" ] || [ "$ARM_NAME" = "hint-sequential" ]; }; then
    STATE_CHECKPOINT_NAMESPACE=$(python -c \
        'import hashlib,pathlib,sys; h=hashlib.sha256("\0".join(sys.argv[1:5]).encode()); files=[pathlib.Path("config.json"),pathlib.Path(sys.argv[5]),*sorted(pathlib.Path("prompts").glob("*.md"))]; [h.update(p.name.encode()+b"\0"+p.read_bytes()+b"\0") for p in files]; print(h.hexdigest()[:24])' \
        "state-audit" "$MODEL_NAME" "$AUDIT_MODEL_NAME" "$ARM_NAME" "$ACTIVE_AGENT_SETTINGS")
    STATE_CHECKPOINT_MOUNT="$PWD/.session-checkpoints/runtime/$STATE_CHECKPOINT_NAMESPACE"
    mkdir -p "$STATE_CHECKPOINT_MOUNT"
    chmod 700 "$STATE_CHECKPOINT_MOUNT"
    docker run --rm --cap-add=NET_ADMIN \
        ${token_args[@]+"${token_args[@]}"} \
        ${network_args[@]+"${network_args[@]}"} \
        -v "$PWD/prompts:/app/prompts:ro" \
        -v "$PWD/config.json:/app/config.json:ro" \
        -v "$PWD/agent_settings.json:/app/agent_settings.json:ro" \
        -v "$PWD/agent_settings_small.json:/app/agent_settings_small.json:ro" \
        -v "$PWD/results:/app/results:ro" \
        -v "$PWD/state-results:/app/state-results" \
        -v "$STATE_CHECKPOINT_MOUNT:/c" \
        -e HARNESS_CHECKPOINT_ROOT=/c \
        -e "HARNESS_PROVIDER_KIND=$PROVIDER_KIND" \
        -e "HARNESS_OPENROUTER_ALLOWED_MODEL=$ACTIVE_MODEL" \
        "$IMAGE" state-audit "${@:2}"
    python src/cleanup_checkpoints.py "$STATE_CHECKPOINT_MOUNT" "$PWD/state-results"
fi
