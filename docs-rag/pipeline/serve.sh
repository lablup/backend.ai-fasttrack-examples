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
# Serving containers can read the previous task's /pipeline/outputs, so the code
# and indices are read from wherever the model definition points this script —
# the named model vfolder need only carry the definition file itself.
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

# Resolve from this script's own location, which is correct wherever the model
# definition launches it from — staged model storage, the pipeline output mount,
# or a local checkout. Do not special-case /models: a stale staged copy there
# would then win over the tree the definition actually pointed at.
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CODE_ROOT"

echo "[serve] service=$SERVICE code_root=$CODE_ROOT"
echo "[serve] python=$(python3 --version 2>&1)"

# Model storage is read-only in a serving container, so the venv cannot live
# beside the code the way the batch tasks' venv does.
VENV="${HOME:-/tmp}/.docsrag-service-venv"
STAMP="$VENV/.installed"

# Ubuntu ships ensurepip separately and the container has no root to install it,
# so venv creation can succeed at making the tree and fail wiring pip into it.
# Mirrors pipeline/bootstrap.sh — pip is in the guard because a failed attempt
# leaves bin/python behind and would otherwise skip the repair.
if [ ! -x "$VENV/bin/python" ] || [ ! -x "$VENV/bin/pip" ]; then
    echo "[serve] creating venv at $VENV"
    rm -rf "$VENV"
    if ! python3 -m venv "$VENV"; then
        echo "[serve] ensurepip unavailable — building venv without pip"
        rm -rf "$VENV"
        python3 -m venv --without-pip "$VENV"
        curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
        "$VENV/bin/python" /tmp/get-pip.py --quiet
        rm -f /tmp/get-pip.py
    fi
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

# Backend.AI routes to 8080 when a deployment has no model definition to say
# otherwise, so default to that: the service then answers on the expected port
# whether or not the definition was found. The definitions below pin the same
# value, so both paths agree rather than one silently listening elsewhere.
SERVICE_PORT="${DOCSRAG_PORT:-8080}"
echo "[serve] port=$SERVICE_PORT"

exec "$VENV/bin/python" -m "$MODULE" --host 0.0.0.0 --port "$SERVICE_PORT" "$@"
