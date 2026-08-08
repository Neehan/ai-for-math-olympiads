#!/usr/bin/env bash
# Start and manage isolated local gateways for authorized Codex subscriptions.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

IMAGE=${CODEX_LITELLM_IMAGE:-olympiad-litellm-codex-compat:1.97.0}
NETWORK=${CODEX_LITELLM_NETWORK:-olympiad-codex-litellm}
CONTAINER_PREFIX=${CODEX_LITELLM_CONTAINER_PREFIX:-olympiad-codex-litellm}
AUTH_VOLUME_PREFIX=${CODEX_LITELLM_AUTH_VOLUME_PREFIX:-olympiad-codex-litellm-auth}
AUTH_ROOT=${CODEX_LITELLM_AUTH_ROOT:-"${HOME}/.codex-subscriptions"}
BASE_PORT=${CODEX_LITELLM_BASE_PORT:-4100}
PROXY_KEY=${LITELLM_API_KEY:-sk-codex-local-only}
MODEL=${CODEX_LITELLM_MODEL:-gpt-5.4}
STARTUP_TIMEOUT=${CODEX_LITELLM_TIMEOUT_SECONDS:-120}

usage() {
    printf '%s\n' \
        "Usage: $0 <command> [arguments]" \
        "" \
        "  build" \
        "  add SLOT [AUTH_JSON]" \
        "  add-all COUNT [AUTH_ROOT]" \
        "  start COUNT" \
        "  verify COUNT" \
        "  status COUNT" \
        "  env COUNT" \
        "  logs SLOT" \
        "  stop COUNT" \
        "  delete-auth SLOT --yes" \
        "" \
        "Default source: ${AUTH_ROOT}/slot-N/auth.json" \
        "'add' copies one login into its own private Docker volume."
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

validate_number() {
    local VALUE=$1
    local LABEL=$2
    case "$VALUE" in
        ''|*[!0-9]*) die "$LABEL must be an integer from 1 through 64" ;;
    esac
    if [ "$VALUE" -lt 1 ] || [ "$VALUE" -gt 64 ]; then
        die "$LABEL must be an integer from 1 through 64"
    fi
}

container_name() {
    printf '%s-%s' "$CONTAINER_PREFIX" "$1"
}

auth_volume() {
    printf '%s-%s' "$AUTH_VOLUME_PREFIX" "$1"
}

host_port() {
    printf '%s' $((BASE_PORT + $1))
}

internal_url() {
    printf 'http://%s:4000' "$(container_name "$1")"
}

host_url() {
    printf 'http://127.0.0.1:%s' "$(host_port "$1")"
}

build_image() {
    docker build \
        -f docker/Dockerfile.litellm-codex \
        -t "$IMAGE" .
}

ensure_image() {
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        build_image
    fi
}

ensure_network() {
    if ! docker network inspect "$NETWORK" >/dev/null 2>&1; then
        docker network create "$NETWORK" >/dev/null
    fi
}

container_running() {
    [ "$(docker inspect -f '{{.State.Running}}' "$(container_name "$1")" 2>/dev/null || true)" = "true" ]
}

volume_has_auth() {
    local VOLUME
    VOLUME=$(auth_volume "$1")
    docker volume inspect "$VOLUME" >/dev/null 2>&1 || return 1
    docker run --rm --entrypoint sh -v "$VOLUME:/auth:ro" "$IMAGE" \
        -c 'test -s /auth/auth.json && test -s /auth/identity.sha256' >/dev/null
}

volume_identity() {
    local VOLUME
    VOLUME=$(auth_volume "$1")
    docker run --rm --entrypoint sh -v "$VOLUME:/auth:ro" "$IMAGE" \
        -c 'cat /auth/identity.sha256'
}

import_slot() {
    local SLOT=$1
    local SOURCE_AUTH=${2:-"${AUTH_ROOT}/slot-${SLOT}/auth.json"}
    local CONTAINER VOLUME
    validate_number "$SLOT" SLOT
    [ -f "$SOURCE_AUTH" ] || die "Codex auth file not found: $SOURCE_AUTH"
    ensure_image
    CONTAINER=$(container_name "$SLOT")
    VOLUME=$(auth_volume "$SLOT")
    if docker inspect "$CONTAINER" >/dev/null 2>&1; then
        docker rm -f "$CONTAINER" >/dev/null
    fi
    docker volume create "$VOLUME" >/dev/null
    docker run --rm \
        --entrypoint python \
        -v "$SOURCE_AUTH:/source/auth.json:ro" \
        -v "$VOLUME:/auth" \
        -v "$PWD/scripts/codex_pool_internal/import_auth.py:/tool/import_auth.py:ro" \
        "$IMAGE" /tool/import_auth.py /source/auth.json /auth/auth.json
    printf 'Added account slot %s to private Docker volume %s.\n' "$SLOT" "$VOLUME"
}

import_all() {
    local COUNT=$1
    local ROOT=${2:-$AUTH_ROOT}
    local SLOT SOURCE
    validate_number "$COUNT" COUNT
    # Validate the complete batch before stopping or replacing any sidecar.
    for SLOT in $(seq 1 "$COUNT"); do
        SOURCE="${ROOT}/slot-${SLOT}/auth.json"
        [ -f "$SOURCE" ] || die "Codex auth file not found: $SOURCE"
    done
    for SLOT in $(seq 1 "$COUNT"); do
        import_slot "$SLOT" "${ROOT}/slot-${SLOT}/auth.json"
    done
}

wait_for_slot() {
    local SLOT=$1
    local URL CONTAINER
    URL=$(host_url "$SLOT")
    CONTAINER=$(container_name "$SLOT")
    for _ in $(seq 1 "$STARTUP_TIMEOUT"); do
        if curl --max-time 1 -fsS "$URL/health/liveliness" >/dev/null 2>&1; then
            printf 'slot=%s url=%s container=%s ready\n' "$SLOT" "$URL" "$CONTAINER"
            return
        fi
        if ! container_running "$SLOT"; then
            docker logs "$CONTAINER"
            die "slot $SLOT exited during startup"
        fi
        sleep 1
    done
    die "slot $SLOT did not become ready within $STARTUP_TIMEOUT seconds"
}

start_slot() {
    local SLOT=$1
    local CONTAINER VOLUME PORT RUNNING_IMAGE_ID DESIRED_IMAGE_ID
    validate_number "$SLOT" SLOT
    ensure_image
    ensure_network
    volume_has_auth "$SLOT" || die "slot $SLOT has no account; run '$0 add $SLOT'"
    CONTAINER=$(container_name "$SLOT")
    VOLUME=$(auth_volume "$SLOT")
    PORT=$(host_port "$SLOT")
    if container_running "$SLOT"; then
        RUNNING_IMAGE_ID=$(docker inspect -f '{{.Image}}' "$CONTAINER")
        DESIRED_IMAGE_ID=$(docker image inspect -f '{{.Id}}' "$IMAGE")
        if [ "$RUNNING_IMAGE_ID" != "$DESIRED_IMAGE_ID" ]; then
            die "slot $SLOT uses an outdated image; stop the pool before restarting it"
        fi
        wait_for_slot "$SLOT"
        if ! curl --max-time 3 -fsS \
            -H "Authorization: Bearer ${PROXY_KEY}" \
            "$(host_url "$SLOT")/v1/models" >/dev/null 2>&1; then
            die "slot $SLOT rejects the configured LITELLM_API_KEY; stop it before changing keys"
        fi
        return
    elif docker inspect "$CONTAINER" >/dev/null 2>&1; then
        docker rm "$CONTAINER" >/dev/null
    fi
    docker run -d --name "$CONTAINER" \
        --network "$NETWORK" \
        --label ai.olympiad.codex-litellm-pool=true \
        --label "ai.olympiad.codex-litellm-slot=${SLOT}" \
        -p "127.0.0.1:${PORT}:4000" \
        -e LITELLM_MASTER_KEY="$PROXY_KEY" \
        -e CHATGPT_TOKEN_DIR=/var/lib/litellm-chatgpt \
        -e DISABLE_AIOHTTP_TRANSPORT=true \
        -v "$VOLUME:/var/lib/litellm-chatgpt" \
        -v "$PWD/docker/litellm.codex.yaml:/app/config.yaml:ro" \
        "$IMAGE" --config /app/config.yaml --host 0.0.0.0 --port 4000 >/dev/null
    wait_for_slot "$SLOT"
}

start_all() {
    local COUNT=$1
    local SLOT IDENTITY IDENTITIES="|"
    validate_number "$COUNT" COUNT
    ensure_image
    for SLOT in $(seq 1 "$COUNT"); do
        volume_has_auth "$SLOT" || die "slot $SLOT has no account; run '$0 add $SLOT'"
        IDENTITY=$(volume_identity "$SLOT")
        case "$IDENTITIES" in
            *"|${IDENTITY}|"*) die "slot $SLOT duplicates another subscription identity" ;;
        esac
        IDENTITIES="${IDENTITIES}${IDENTITY}|"
    done
    for SLOT in $(seq 1 "$COUNT"); do
        start_slot "$SLOT"
    done
}

run_python() {
    if [ -x .venv/bin/python ]; then
        .venv/bin/python scripts/codex_pool_internal/verify.py "$@"
    else
        python scripts/codex_pool_internal/verify.py "$@"
    fi
}

verify_slot() {
    local SLOT=$1
    local URL
    validate_number "$SLOT" SLOT
    container_running "$SLOT" || die "slot $SLOT is not running"
    URL=$(host_url "$SLOT")
    printf 'Verifying subscription slot %s.\n' "$SLOT"
    run_python direct --base-url "$URL" --proxy-key "$PROXY_KEY" --model "$MODEL"
    run_python agent --base-url "$URL" --proxy-key "$PROXY_KEY" --model "$MODEL"
}

verify_all() {
    local COUNT=$1
    local SLOT
    validate_number "$COUNT" COUNT
    for SLOT in $(seq 1 "$COUNT"); do
        verify_slot "$SLOT"
    done
}

status_all() {
    local COUNT=$1
    local SLOT STATE URL
    validate_number "$COUNT" COUNT
    for SLOT in $(seq 1 "$COUNT"); do
        URL=$(host_url "$SLOT")
        if container_running "$SLOT" && curl --max-time 1 -fsS "$URL/health/liveliness" >/dev/null 2>&1; then
            STATE=ready
        elif container_running "$SLOT"; then
            STATE=starting
        else
            STATE=stopped
        fi
        printf 'slot=%s state=%s host_url=%s harness_url=%s\n' \
            "$SLOT" "$STATE" "$URL" "$(internal_url "$SLOT")"
    done
}

print_env() {
    local COUNT=$1
    local SLOT SUFFIX
    validate_number "$COUNT" COUNT
    printf 'LITELLM_API_KEY=%s\n' "$PROXY_KEY"
    for SLOT in $(seq 1 "$COUNT"); do
        SUFFIX=""
        if [ "$SLOT" -gt 1 ]; then
            SUFFIX="_${SLOT}"
        fi
        printf 'LITELLM_BASE_URL%s=%s\n' "$SUFFIX" "$(internal_url "$SLOT")"
    done
}

stop_all() {
    local COUNT=$1
    local SLOT CONTAINER
    validate_number "$COUNT" COUNT
    for SLOT in $(seq 1 "$COUNT"); do
        CONTAINER=$(container_name "$SLOT")
        if docker inspect "$CONTAINER" >/dev/null 2>&1; then
            docker rm -f "$CONTAINER" >/dev/null
            printf 'Stopped slot %s; retained OAuth volume %s.\n' \
                "$SLOT" "$(auth_volume "$SLOT")"
        fi
    done
}

delete_auth() {
    local SLOT=$1
    local CONFIRM=${2:-}
    local CONTAINER VOLUME
    validate_number "$SLOT" SLOT
    [ "$CONFIRM" = "--yes" ] || die "deleting OAuth requires: $0 delete-auth $SLOT --yes"
    CONTAINER=$(container_name "$SLOT")
    VOLUME=$(auth_volume "$SLOT")
    if docker inspect "$CONTAINER" >/dev/null 2>&1; then
        docker rm -f "$CONTAINER" >/dev/null
    fi
    if docker volume inspect "$VOLUME" >/dev/null 2>&1; then
        docker volume rm "$VOLUME" >/dev/null
        printf 'Permanently deleted copied OAuth volume %s.\n' "$VOLUME"
    fi
}

COMMAND=${1:-}
case "$COMMAND" in
    build)
        build_image
        ;;
    add)
        [ $# -ge 2 ] || die "add requires SLOT"
        import_slot "$2" "${3:-}"
        ;;
    add-all)
        [ $# -ge 2 ] || die "add-all requires COUNT"
        import_all "$2" "${3:-}"
        ;;
    start)
        [ $# -eq 2 ] || die "start requires COUNT"
        start_all "$2"
        ;;
    verify)
        [ $# -eq 2 ] || die "verify requires COUNT"
        verify_all "$2"
        ;;
    status)
        [ $# -eq 2 ] || die "status requires COUNT"
        status_all "$2"
        ;;
    env)
        [ $# -eq 2 ] || die "env requires COUNT"
        print_env "$2"
        ;;
    logs)
        [ $# -eq 2 ] || die "logs requires SLOT"
        validate_number "$2" SLOT
        docker logs -f "$(container_name "$2")"
        ;;
    stop)
        [ $# -eq 2 ] || die "stop requires COUNT"
        stop_all "$2"
        ;;
    delete-auth)
        [ $# -ge 2 ] || die "delete-auth requires SLOT --yes"
        delete_auth "$2" "${3:-}"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
