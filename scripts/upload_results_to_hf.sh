#!/bin/sh
# Incrementally upload both result trees to the private Hugging Face dataset.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
REPO_ID=${HF_RESULTS_REPO:-notadib/strategy-ceiling}
WORKERS=${HF_UPLOAD_WORKERS:-8}

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
exec hf upload-large-folder \
    "$REPO_ID" . \
    --repo-type dataset \
    --private \
    --include 'results/**' 'results-imobench/**' \
    --exclude '.DS_Store' '**/.DS_Store' \
    --num-workers "$WORKERS" \
    --no-bars
