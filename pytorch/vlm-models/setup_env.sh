#!/bin/bash

# --- 1. CLI의 첫 번째 인자를 VFOLDER_NAME 변수에 할당 ---
if [ -z "$1" ]; then
    echo "❌ Error: Please provide the base path for your vfolder as the first argument."
    echo "Usage: ./setup_env.sh /path/to/your/vfolder"
    exit 1
fi

VFOLDER_NAME=$1
# --------------------------------------------------------

# Inside a Backend.AI pipeline job the venv belongs on the shared vfroot so every
# task sees it. Anywhere else /pipeline does not exist, so fall back to the
# vfolder path the caller already had to supply.
if [ -n "$BACKENDAI_PIPELINE_JOB_ID" ]; then
    VENV_PATH="/pipeline/vfroot/.venv"
else
    VENV_PATH="${VFOLDER_NAME}/.venv"
fi

# Some images ship only python3.
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
else
    echo "❌ Error: Neither python3 nor python was found on PATH."
    exit 1
fi

REQUIREMENTS_PATH="${VFOLDER_NAME}/backend.ai-fasttrack-examples/pytorch/vlm-models/pipeline-code/requirements.txt"

echo "Using base path: $VFOLDER_NAME"

# 2. 가상 환경이 존재하는지 확인하고, 없으면 새로 생성합니다.
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found. Creating a new one at $VENV_PATH..."
    if ! "$PYTHON_BIN" -m venv "$VENV_PATH"; then
        echo "❌ Error: Failed to create a virtual environment at $VENV_PATH"
        exit 1
    fi
else
    echo "Existing virtual environment found at $VENV_PATH."
fi

# 3. 공통 작업: 항상 패키지를 설치하거나 확인합니다.
echo "Installing or verifying packages from $REQUIREMENTS_PATH..."
if ! "$VENV_PATH/bin/pip" install -r "$REQUIREMENTS_PATH"; then
    echo "❌ Error: Failed to install packages from $REQUIREMENTS_PATH"
    exit 1
fi

# -----------------------------

# 4. 사용자에게 다음 행동을 안내합니다.
echo ""
echo "✅ Virtual environment setup complete."
echo "To activate the environment, run the following command in your terminal:"
echo "source $VENV_PATH/bin/activate"