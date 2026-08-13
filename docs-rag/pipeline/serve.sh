#!/bin/bash
# Deployment entrypoint for both services.
#
#   bash pipeline/serve.sh fastapi
#   bash pipeline/serve.sh gradio
#
# Invoked by model-definition-{fastapi,gradio}.yaml. The service is chosen by
# argument rather than by an environment variable, because a cluster that drops
# the YAML `envs` block would otherwise boot two copies of the same service.
#
# A serving container sees model storage at /models and nothing of /pipeline, so
# everything here comes from /models — the code, the indices and the credentials
# were all put there by the stage-service node.
set -euo pipefail

SERVICE="${1:-}"
case "$SERVICE" in
    fastapi) MODULE="docs_rag.server" ;;
    gradio)  MODULE="docs_rag.ui" ;;
    *)
        echo "[serve] FATAL: expected 'fastapi' or 'gradio', got '${SERVICE:-<nothing>}'" >&2
        exit 2
        ;;
esac
shift

# Staged copy first, then a local checkout — so the same script serves a
# deployment and a laptop.
if [ -d "/models/docs-rag" ]; then
    CODE_ROOT="/models/docs-rag"
else
    CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$CODE_ROOT"

echo "[serve] service=$SERVICE code_root=$CODE_ROOT"
echo "[serve] python=$(python3 --version 2>&1)"

# Model storage is read-only in a serving container, so the venv cannot live
# beside the code the way the batch tasks' venv does.
VENV="${HOME:-/tmp}/.docsrag-service-venv"
STAMP="$VENV/.installed"

if [ ! -x "$VENV/bin/python" ]; then
    echo "[serve] creating venv at $VENV"
    python3 -m venv "$VENV"
fi
if [ ! -f "$STAMP" ] || [ requirements-service.txt -nt "$STAMP" ]; then
    echo "[serve] installing service dependencies"
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install --quiet -r requirements-service.txt
    touch "$STAMP"
fi

# Where the indices were staged. Set explicitly rather than inferred, because
# the failure mode otherwise is a service that starts healthy and answers
# nothing.
export DOCSRAG_INDICES="$CODE_ROOT/pipeline/data/03_indices"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -d "$DOCSRAG_INDICES" ]; then
    echo "[serve] WARNING: no indices at $DOCSRAG_INDICES — did the stage-service node run?" >&2
else
    echo "[serve] indices: $(ls "$DOCSRAG_INDICES" | tr '\n' ' ')"
fi

exec "$VENV/bin/python" -m "$MODULE" --host 0.0.0.0 --port 8000 "$@"
