#!/usr/bin/env bash
# Build the image and run the generation or audit pipeline in throwaway
# containers behind the same egress firewall. One public `audit` command first
# grades correctness, then launches a separate
# internal container for route-state annotation on eligible arms.
# Keeping the stages separate prevents state annotations from influencing
# correctness verdicts.
#
# The container is removed on exit (--rm); math-contests-2026 results land in
# ./results/ and IMO-ProofBench results in ./results-imobench/ via the bind
# mount. Problems and hints are fetched from the selected dataset URLs by the
# entrypoint (never stored in the image); prompts/, config.json, and
# agent settings profiles are mounted read-only so editing them needs no rebuild.
#
# The run stage normally mounts only meta.json resume markers. Compression
# additionally receives planner artifacts. Matched late interventions share
# only their private native-prefix checkpoint store.
# On exit, only newly completed attempts are merged into results/; the audit
# stage mounts the full tree because the judge must read solutions.
#
# Usage:
#   cp .env.example .env             # set CLAUDE_CODE_OAUTH_TOKEN in it
#   ./run.sh run --arm baseline
#   ./run.sh run --arm baseline --domain combinatorics
#   ./run.sh run --dataset imobench --arm baseline
#   ./run.sh audit --arm baseline
set -euo pipefail

cd "$(dirname "$0")"

if [ $# -lt 3 ] || { [ "$1" != "run" ] && [ "$1" != "audit" ] && [ "$1" != "state-audit" ]; }; then
    echo "Usage: ./run.sh <run|audit|state-audit> --arm <ARM> [--dataset math-contests-2026|imobench] [--problems id1,id2] [--domain d]" >&2
    exit 2
fi

# Dataset selection is a host-controller concern: it determines both the
# protected URLs fetched inside Docker and the host result tree. Strip the
# public flag before forwarding the remaining arguments to Python.
DATASET_NAME=math-contests-2026
STAGE=$1
shift
FORWARD_ARGS=("$STAGE")
while [ $# -gt 0 ]; do
    case "$1" in
        --dataset)
            if [ $# -lt 2 ] || [ -z "$2" ]; then
                echo "ERROR: --dataset requires a value" >&2
                exit 2
            fi
            DATASET_NAME=$2
            shift 2
            ;;
        --dataset=*)
            DATASET_NAME=${1#--dataset=}
            if [ -z "$DATASET_NAME" ]; then
                echo "ERROR: --dataset requires a value" >&2
                exit 2
            fi
            shift
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${FORWARD_ARGS[@]}"

case "$DATASET_NAME" in
    math-contests-2026) RESULTS_DIR_NAME=results ;;
    imobench) RESULTS_DIR_NAME=results-imobench ;;
    *)
        echo "ERROR: unknown dataset '$DATASET_NAME' (expected math-contests-2026 or imobench)" >&2
        exit 2
        ;;
esac
RESULTS_HOST_ROOT="$PWD/$RESULTS_DIR_NAME"

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

# Mount only this stage/model/arm's opaque checkpoint namespace.  Mounting the
# whole bank would let a tool-using solver inspect interrupted work from a
# different intervention arm.  The namespace is stable across reruns but its
# host name reveals no experiment identity inside the container.
ARM_NAME=""
MODEL_NAME=$(python -c 'import json; print(json.load(open("config.json"))["model"])')
AUDIT_MODEL_NAME=$(python -c 'import json; print(json.load(open("config.json"))["audit_model"])')
WORKER_MODEL_NAME=""
PROBLEMS_FILTER=""
DOMAIN_FILTER=""
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
    if [ "$EXPECT_VALUE" = "worker_model" ]; then
        WORKER_MODEL_NAME="$ARGUMENT"
        EXPECT_VALUE=""
        continue
    fi
    if [ "$EXPECT_VALUE" = "problems" ]; then
        PROBLEMS_FILTER="$ARGUMENT"
        EXPECT_VALUE=""
        continue
    fi
    if [ "$EXPECT_VALUE" = "domain" ]; then
        DOMAIN_FILTER="$ARGUMENT"
        EXPECT_VALUE=""
        continue
    fi
    case "$ARGUMENT" in
        --arm) EXPECT_VALUE="arm" ;;
        --model) EXPECT_VALUE="model" ;;
        --audit-model) EXPECT_VALUE="audit_model" ;;
        --worker-model) EXPECT_VALUE="worker_model" ;;
        --problems) EXPECT_VALUE="problems" ;;
        --domain) EXPECT_VALUE="domain" ;;
        --arm=*) ARM_NAME=${ARGUMENT#--arm=} ;;
        --model=*) MODEL_NAME=${ARGUMENT#--model=} ;;
        --audit-model=*) AUDIT_MODEL_NAME=${ARGUMENT#--audit-model=} ;;
        --worker-model=*) WORKER_MODEL_NAME=${ARGUMENT#--worker-model=} ;;
        --problems=*) PROBLEMS_FILTER=${ARGUMENT#--problems=} ;;
        --domain=*) DOMAIN_FILTER=${ARGUMENT#--domain=} ;;
    esac
done
if [ -n "$EXPECT_VALUE" ]; then
    echo "ERROR: --${EXPECT_VALUE//_/-} requires a value" >&2
    exit 2
fi
if [ -z "$ARM_NAME" ]; then
    echo "ERROR: --arm requires a value" >&2
    exit 2
fi

if [ "$1" = "run" ]; then
    case "$ARM_NAME" in
        baseline-uniform-compress)
            WORKER_MODEL_NAME=${WORKER_MODEL_NAME:-litellm/gpt-5.6-sol}
            ;;
        selection|selection-no-problem)
            if [ -n "$WORKER_MODEL_NAME" ]; then
                echo "ERROR: selection arms must use the source model; omit --worker-model" >&2
                exit 2
            fi
            ;;
    esac
fi
IS_SELECTION_ARM=0
case "$ARM_NAME" in
    selection|selection-no-problem)
        IS_SELECTION_ARM=1
        ;;
esac
if [ "$1" = "audit" ] && [ "$ARM_NAME" = "baseline-uniform-compress" ]; then
    echo "ERROR: baseline-uniform-compress has no audit stage; audit baseline-uniform-strategy-only instead" >&2
    exit 2
fi
if [ "$1" = "audit" ] && [ "$IS_SELECTION_ARM" -eq 1 ]; then
    # Selection writes its deterministic per-attempt verdict during generation.
    # Recompiling the arm index needs neither a judge nor provider connectivity.
    python scripts/compile_selection_audit.py \
        --results-root "$RESULTS_HOST_ROOT" \
        --model "$MODEL_NAME" \
        --arm "$ARM_NAME"
    exit 0
fi

IMAGE=olympiad-harness
docker build -q -t "$IMAGE" -f docker/Dockerfile . >/dev/null
mkdir -p "$RESULTS_HOST_ROOT"
CHECKPOINT_ARM_ID=$ARM_NAME
if [ -n "$WORKER_MODEL_NAME" ]; then
    CHECKPOINT_ARM_ID="$ARM_NAME:$WORKER_MODEL_NAME"
fi
case "$ARM_NAME" in
    late-baseline-sequential|late-hint-sequential)
        # Both arms must see the same retained native-prefix store. Their
        # per-attempt identities remain distinct inside this shared namespace.
        CHECKPOINT_ARM_ID=late-intervention
        ;;
esac

if [ "$1" = "run" ] && [ "$ARM_NAME" = "baseline-uniform-strategy-only" ]; then
    if [ -z "$DOMAIN_FILTER" ] || [ -n "$PROBLEMS_FILTER" ]; then
        reuse_args=(--results-root "$RESULTS_HOST_ROOT" --model "$MODEL_NAME")
        if [ -n "$PROBLEMS_FILTER" ]; then
            reuse_args+=(--problems "$PROBLEMS_FILTER")
        fi
        python scripts/reuse_uniform_strategies.py "${reuse_args[@]}"
    else
        echo "planner reuse skipped for --domain-only selection; pass explicit --problems to reuse existing banks" >&2
    fi
fi

if [ "$1" = "run" ]; then
    ACTIVE_MODEL=${WORKER_MODEL_NAME:-$MODEL_NAME}
elif [ "$IS_SELECTION_ARM" -eq 1 ]; then
    # Selection verdicts are deterministic fields written during generation;
    # recompiling them does not call a judge model.
    ACTIVE_MODEL=$MODEL_NAME
else
    ACTIVE_MODEL=$AUDIT_MODEL_NAME
fi
network_args=()
ACTIVE_AGENT_SETTINGS=agent_settings.json
case "$ACTIVE_MODEL" in
    muse-spark-1.2-contributor)
        if [ -z "${META_API_KEY:-}" ]; then
            echo "ERROR: $ACTIVE_MODEL requires META_API_KEY in .env." >&2
            exit 1
        fi
        TOKEN_PATTERN='^META_API_KEY(_[0-9]+)?$'
        PROVIDER_KIND=meta
        ;;
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

if [ "$DATASET_NAME" = "math-contests-2026" ]; then
    # Preserve the exact legacy identity so paid in-flight checkpoints remain
    # resumable after adding the dataset selector.
    CHECKPOINT_NAMESPACE=$(python scripts/checkpoint_namespace.py \
        "$1" "$MODEL_NAME" "$AUDIT_MODEL_NAME" "$CHECKPOINT_ARM_ID" \
        "$ACTIVE_AGENT_SETTINGS")
else
    CHECKPOINT_NAMESPACE=$(python scripts/checkpoint_namespace.py \
        "$1" "$MODEL_NAME" "$AUDIT_MODEL_NAME" "$CHECKPOINT_ARM_ID" \
        "$DATASET_NAME" "$ACTIVE_AGENT_SETTINGS")
fi
CHECKPOINT_MOUNT="$PWD/.session-checkpoints/runtime/$CHECKPOINT_NAMESPACE"
mkdir -p "$CHECKPOINT_MOUNT"
chmod 700 .session-checkpoints .session-checkpoints/runtime "$CHECKPOINT_MOUNT"

# Pass every provider key/endpoint var through (round-robin token pools).
token_args=()
while IFS= read -r name; do
    token_args+=(-e "$name")
done < <(compgen -v | grep -E "$TOKEN_PATTERN")

RESULTS_MOUNT="$RESULTS_HOST_ROOT"
STAGING=""
cleanup_completed_checkpoints() {
    if [ ! -d "$CHECKPOINT_MOUNT/attempts" ]; then
        return
    fi
    python src/cleanup_checkpoints.py "$CHECKPOINT_MOUNT" "$RESULTS_HOST_ROOT"
}
merge_staging() {
    if [ -n "$STAGING" ] && [ -d "$STAGING" ]; then
        # meta.json is written last. Drop pre-seeded markers and partial writes,
        # but retain completed run_<kk> children from an interrupted bank so a
        # restart spends only on its unfinished members.
        python scripts/prune_staging.py "$STAGING"
        find "$STAGING" -mindepth 1 -depth -type d -empty -exec rmdir {} \;
        rsync -a "$STAGING"/ "$RESULTS_HOST_ROOT"/
        rm -rf "$STAGING"
    fi
}
finish_stage() {
    merge_staging
    cleanup_completed_checkpoints
}
trap finish_stage EXIT

if [ "$1" = "run" ]; then
    STAGING=$(mktemp -d "$PWD/.$RESULTS_DIR_NAME-staging.XXXXXX")
    if [ "$IS_SELECTION_ARM" -eq 1 ]; then
        MODEL_DIR_NAME=${MODEL_NAME//\//-}
        # A selector needs completion-path existence to skip finished attempts,
        # never prior rankings, oracle positions, or other experimental output.
        python scripts/stage_selection_markers.py \
            --source-root "$RESULTS_HOST_ROOT" \
            --destination-root "$STAGING" \
            --model-dir "$MODEL_DIR_NAME" \
            --arm "$ARM_NAME"
    else
        rsync -a --include='*/' --include='meta.json' --exclude='*' \
            "$RESULTS_HOST_ROOT"/ "$STAGING"/
    fi
    if [ "$ARM_NAME" = "baseline-uniform-strategy-only" ]; then
        MODEL_DIR_NAME=${MODEL_NAME//\//-}
        PLANNER_ONLY_ROOT="$RESULTS_HOST_ROOT/$MODEL_DIR_NAME/baseline-uniform-strategy-only"
        STAGED_PLANNER_ONLY_ROOT="$STAGING/$MODEL_DIR_NAME/baseline-uniform-strategy-only"
        if [ -d "$PLANNER_ONLY_ROOT" ]; then
            mkdir -p "$STAGED_PLANNER_ONLY_ROOT"
            rsync -a --include='*/' --include='meta.json' --include='solution.md' \
                --include='strategies.json' --exclude='*' \
                "$PLANNER_ONLY_ROOT"/ "$STAGED_PLANNER_ONLY_ROOT"/
        fi
    fi
    if [ "$ARM_NAME" = "baseline-uniform-compress" ]; then
        MODEL_DIR_NAME=${MODEL_NAME//\//-}
        SOURCE_STRATEGY_ROOT="$RESULTS_HOST_ROOT/$MODEL_DIR_NAME/baseline-uniform-strategy-only"
        STAGED_STRATEGY_ROOT="$STAGING/$MODEL_DIR_NAME/baseline-uniform-strategy-only"
        if [ -d "$SOURCE_STRATEGY_ROOT" ]; then
            mkdir -p "$STAGED_STRATEGY_ROOT"
            rsync -a --include='*/' --include='meta.json' --include='solution.md' \
                --include='strategies.json' --exclude='*' \
                "$SOURCE_STRATEGY_ROOT"/ "$STAGED_STRATEGY_ROOT"/
        fi
    fi
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
    -e "HARNESS_DATASET=$DATASET_NAME" \
    -e "HARNESS_ARM=$ARM_NAME" \
    -e "HARNESS_PROVIDER_KIND=$PROVIDER_KIND" \
    -e "HARNESS_OPENROUTER_ALLOWED_MODEL=$ACTIVE_MODEL" \
    "$IMAGE" "$@"

if [ "$1" = "run" ] && [ "$IS_SELECTION_ARM" -eq 1 ]; then
    # Merge the isolated per-attempt outputs before compiling the complete arm.
    # Compilation is mechanical and runs on the host, outside the selector's
    # tool-visible container.
    finish_stage
    STAGING=""
    python scripts/compile_selection_audit.py \
        --results-root "$RESULTS_HOST_ROOT" \
        --model "$MODEL_NAME" \
        --arm "$ARM_NAME"
fi

# State/strategy annotation is needed only for the temporal trajectories,
# search controls, and the raw planner proposals later displayed in compressed form.
# Standalone fixed-compute arms stop after correctness grading.
RUN_STATE_AUDIT=0
case "$ARM_NAME" in
    baseline-sequential|baseline-sequential-2x|baseline-sequential-4x|hint-sequential|late-baseline-sequential|late-hint-sequential|baseline-parallel|baseline-uniform-strategy|baseline-uniform-strategy-only)
        RUN_STATE_AUDIT=1
        ;;
esac

# State annotation runs in a fresh reference-bearing container. Keep the
# existing checkpoint namespace for the primary dataset and isolate IMO-Bench.
if [ "$1" = "audit" ] && [ "$RUN_STATE_AUDIT" -eq 1 ]; then
    if [ "$DATASET_NAME" = "math-contests-2026" ]; then
        STATE_CHECKPOINT_MOUNT="$PWD/.session-checkpoints/state-audit"
    else
        STATE_CHECKPOINT_MOUNT="$PWD/.session-checkpoints/state-audit-$DATASET_NAME"
    fi
    mkdir -p "$STATE_CHECKPOINT_MOUNT"
    chmod 700 "$STATE_CHECKPOINT_MOUNT"
    docker run --rm --cap-add=NET_ADMIN \
        ${token_args[@]+"${token_args[@]}"} \
        ${network_args[@]+"${network_args[@]}"} \
        -v "$PWD/prompts:/app/prompts:ro" \
        -v "$PWD/config.json:/app/config.json:ro" \
        -v "$PWD/agent_settings.json:/app/agent_settings.json:ro" \
        -v "$PWD/agent_settings_small.json:/app/agent_settings_small.json:ro" \
        -v "$RESULTS_HOST_ROOT:/app/results" \
        -v "$STATE_CHECKPOINT_MOUNT:/c" \
        -e HARNESS_CHECKPOINT_ROOT=/c \
        -e "HARNESS_DATASET=$DATASET_NAME" \
        -e "HARNESS_ARM=$ARM_NAME" \
        -e "HARNESS_PROVIDER_KIND=$PROVIDER_KIND" \
        -e "HARNESS_OPENROUTER_ALLOWED_MODEL=$ACTIVE_MODEL" \
        "$IMAGE" state-audit "${@:2}"
    python src/cleanup_checkpoints.py "$STATE_CHECKPOINT_MOUNT" "$RESULTS_HOST_ROOT"
elif [ "$1" = "audit" ]; then
    echo "state audit skipped for arm '$ARM_NAME' (correctness audit only)"
fi
