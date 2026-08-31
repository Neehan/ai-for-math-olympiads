#!/bin/sh
# Pull and safely merge both active result trees from the private HF dataset.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
REPO_ID=${HF_RESULTS_REPO:-notadib/strategy-ceiling}
REPO_OWNER=${HF_RESULTS_USER:-${REPO_ID%%/*}}
DRY_RUN=

if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=--dry-run
    shift
fi
if [ "$#" -ne 0 ]; then
    echo "usage: $0 [--dry-run]" >&2
    exit 2
fi

if [ -z "${HF_TOKEN:-}" ]; then
    if [ ! -f "$ROOT/.env" ]; then
        echo "HF_TOKEN is unset and $ROOT/.env does not exist" >&2
        exit 2
    fi
    set -a
    # shellcheck disable=SC1091
    . "$ROOT/.env"
    set +a
fi

: "${HF_TOKEN:?HF_TOKEN must be set in the environment or .env}"
command -v git >/dev/null 2>&1 || { echo "git is required" >&2; exit 2; }
command -v git-lfs >/dev/null 2>&1 || { echo "git-lfs is required" >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "python3 is required" >&2; exit 2; }

SNAPSHOT=$(mktemp -d "${TMPDIR:-/tmp}/strategy-ceiling-hf.XXXXXX")
cleanup() {
    case "$SNAPSHOT" in
        "${TMPDIR:-/tmp}"/strategy-ceiling-hf.*) rm -rf -- "$SNAPSHOT" ;;
        *) echo "refusing to remove unexpected temporary path: $SNAPSHOT" >&2 ;;
    esac
}
trap cleanup EXIT HUP INT TERM

# Git transfers the 60k-file repository as a pack; Git LFS then retrieves logs
# in batches. This avoids the per-file HEAD requests that make `hf download`
# hit rate limits. Supply credentials through process-local Git configuration,
# never through the clone URL or the repository's persisted config.
BASIC_AUTH=$(printf '%s:%s' "$REPO_OWNER" "$HF_TOKEN" | base64 | tr -d '\n')
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=http.extraHeader
export GIT_CONFIG_VALUE_0="Authorization: Basic $BASIC_AUTH"

echo "Cloning hf://datasets/$REPO_ID ..."
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
    "https://huggingface.co/datasets/$REPO_ID" "$SNAPSHOT"
git -C "$SNAPSHOT" lfs pull

python3 "$ROOT/scripts/merge_results_snapshot.py" \
    --source "$SNAPSHOT" \
    --destination "$ROOT" \
    ${DRY_RUN:+"$DRY_RUN"}
