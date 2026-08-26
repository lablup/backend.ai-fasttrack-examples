#!/bin/bash
# Batch-task bootstrap: ensure a Python environment, then run a task script.
#
#   bash pipeline/bootstrap.sh pipeline/tasks/03_build_indices.py --project all
#
# The stock container images carry a Python runtime but none of this project's
# dependencies and no pandoc, and installing system packages needs root. So we
# build a virtualenv and drop in a static pandoc binary instead.
#
# The venv lives on the persistent pipeline folder when there is one, so it is
# built once by the first node and reused by every later node. Tasks run
# sequentially, so there is no race to create it.
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CODE_ROOT"

PANDOC_VERSION=3.5

# --- evidence ------------------------------------------------------------
# Printed before any work, because the two ways this fails in production are a
# stale checkout and a missing mount, and both are obvious from these lines.
echo "[bootstrap] host=$(hostname) user=$(id -un) pwd=$(pwd)"
echo "[bootstrap] python=$(command -v python3 || echo MISSING) $(python3 --version 2>&1)"
if git -C "$CODE_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    echo "[bootstrap] checkout=$(git -C "$CODE_ROOT" log -1 --format='%h %ad %s' --date=short)"
else
    echo "[bootstrap] checkout=<not a git working tree>"
fi
for var in PIPELINE_OUTPUT_ROOT PIPELINE_INPUT_ROOT PIPELINE_VFROOT BACKENDAI_PIPELINE_JOB_INDEX; do
    echo "[bootstrap] ${var}=${!var:-<unset>}"
done
# Secrets by length only — never the value.
for secret in OPENAI_API_KEY GITHUB_TOKEN; do
    value="${!secret:-}"
    if [ -n "$value" ]; then
        echo "[bootstrap] ${secret}=<set, ${#value} chars>"
    else
        echo "[bootstrap] ${secret}=<unset in shell; may still come from .env>"
    fi
done
unset value

# --- the task script must exist before we spend a minute building a venv ---
TASK_SCRIPT="${1:-}"
if [ -z "$TASK_SCRIPT" ]; then
    echo "[bootstrap] FATAL: no task script given" >&2
    exit 2
fi
if [ ! -f "$TASK_SCRIPT" ]; then
    echo "[bootstrap] FATAL: task script '$TASK_SCRIPT' not found in $CODE_ROOT" >&2
    echo "[bootstrap] available tasks:" >&2
    ls -la pipeline/tasks/ >&2 || true
    echo "[bootstrap] the fetch-code node may have cloned an older revision." >&2
    exit 127
fi

# --- venv ----------------------------------------------------------------
if [ -n "${PIPELINE_VFROOT:-}" ] && mkdir -p "$PIPELINE_VFROOT" 2>/dev/null; then
    VENV="$PIPELINE_VFROOT/.venv"
else
    VENV="${HOME:-/tmp}/.docsrag-venv"
fi
STAMP="$VENV/.installed"

# Ubuntu ships ensurepip in a separate python3-venv package and installing it
# needs root we do not have, so `python3 -m venv` can create the environment and
# then fail wiring pip into it. The venv module itself works, so fall back to
# building without pip and bootstrapping it over the network. The pip check in
# the guard matters: a half-built venv from such a failure leaves bin/python
# behind, and testing only for that would skip the repair on the next run.
if [ ! -x "$VENV/bin/python" ] || [ ! -x "$VENV/bin/pip" ]; then
    echo "[bootstrap] creating venv at $VENV"
    rm -rf "$VENV"
    if ! python3 -m venv "$VENV"; then
        echo "[bootstrap] ensurepip unavailable — building venv without pip"
        rm -rf "$VENV"
        python3 -m venv --without-pip "$VENV"
        curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
        "$VENV/bin/python" /tmp/get-pip.py --quiet
        rm -f /tmp/get-pip.py
    fi
fi

# Reinstall when the stamp is missing or older than the requirements file, so
# editing requirements.txt takes effect without anyone remembering to clear it.
if [ ! -f "$STAMP" ] || [ requirements.txt -nt "$STAMP" ]; then
    echo "[bootstrap] installing dependencies"
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install --quiet -r requirements.txt

    if ! command -v pandoc >/dev/null 2>&1 && [ ! -x "$VENV/bin/pandoc" ]; then
        echo "[bootstrap] installing static pandoc ${PANDOC_VERSION}"
        TARBALL="pandoc-${PANDOC_VERSION}-linux-amd64.tar.gz"
        curl -fsSL -o "/tmp/$TARBALL" \
            "https://github.com/jgm/pandoc/releases/download/${PANDOC_VERSION}/${TARBALL}"
        tar -xzf "/tmp/$TARBALL" -C /tmp
        cp "/tmp/pandoc-${PANDOC_VERSION}/bin/pandoc" "$VENV/bin/pandoc"
        rm -rf "/tmp/$TARBALL" "/tmp/pandoc-${PANDOC_VERSION}"
    fi

    touch "$STAMP"
fi

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[bootstrap] running: $*"
exec "$VENV/bin/python" "$@"
