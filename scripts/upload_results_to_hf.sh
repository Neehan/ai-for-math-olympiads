#!/bin/sh
# Incrementally upload both result trees, or exactly replace named seed folders.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
REPO_ID=${HF_RESULTS_REPO:-notadib/strategy-ceiling}
WORKERS=${HF_UPLOAD_WORKERS:-8}

usage() {
    cat <<'EOF'
Usage:
  ./scripts/upload_results_to_hf.sh
  ./scripts/upload_results_to_hf.sh --replace-seed RESULTS_SEED [RESULTS_SEED ...]

The default mode incrementally uploads both active result trees. The explicit
--replace-seed mode makes each named remote seed directory exactly match its
local directory, including deleting remote-only files within that seed.
EOF
}

valid_seed_path() (
    seed_path=${1%/}
    case "$seed_path" in
        /*) return 1 ;;
    esac

    IFS=/ read -r tree model arm problem seed extra <<EOF
$seed_path
EOF
    [ -n "$tree" ] && [ -n "$model" ] && [ -n "$arm" ] &&
        [ -n "$problem" ] && [ -n "$seed" ] && [ -z "$extra" ] || return 1
    case "$tree" in
        results|results-imobench) ;;
        *) return 1 ;;
    esac
    case "$model/$arm/$problem" in
        *'/../'*|*'/./'*|../*|./*) return 1 ;;
    esac
    case "$seed" in
        seed_[0-9]*) ;;
        *) return 1 ;;
    esac
    seed_number=${seed#seed_}
    case "$seed_number" in
        ''|*[!0-9]*) return 1 ;;
    esac
)

MODE=incremental
if [ "${1:-}" = "--replace-seed" ]; then
    MODE=replace-seed
    shift
    if [ "$#" -eq 0 ]; then
        usage >&2
        exit 2
    fi
elif [ "$#" -ne 0 ]; then
    usage >&2
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

cd "$ROOT"

if [ "$MODE" = replace-seed ]; then
    for requested_path do
        seed_path=${requested_path%/}
        if ! valid_seed_path "$seed_path"; then
            echo "Invalid result seed path: $requested_path" >&2
            exit 2
        fi
        if [ ! -d "$ROOT/$seed_path" ]; then
            echo "Local result seed directory does not exist: $seed_path" >&2
            exit 2
        fi
        seed_abs=$(CDPATH= cd -- "$ROOT/$seed_path" && pwd -P)
        case "$seed_abs" in
            "$ROOT/results/"*|"$ROOT/results-imobench/"*) ;;
            *)
                echo "Result seed resolves outside an active result tree: $seed_path" >&2
                exit 2
                ;;
        esac

        echo "Replacing remote seed exactly from local: $seed_path"
        hf upload \
            "$REPO_ID" "$seed_path" "$seed_path" \
            --repo-type dataset \
            --delete '**' \
            --commit-message "Replace result seed from authoritative local copy" \
            --quiet
    done
    exit 0
fi

exec hf upload-large-folder \
    "$REPO_ID" . \
    --repo-type dataset \
    --private \
    --include 'results/**' 'results-imobench/**' \
    --exclude '.DS_Store' '**/.DS_Store' \
    --num-workers "$WORKERS" \
    --no-bars
