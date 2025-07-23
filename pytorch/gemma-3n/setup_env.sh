#!/bin/bash

# --- 1. CLI의 첫 번째 인자를 VFOLDER_NAME 변수에 할당 ---
# 인자가 제공되지 않았을 경우, 에러 메시지를 출력하고 스크립트를 종료합니다.
if [ -z "$1" ]; then
    echo "❌ Error: Please provide the base path for your vfolder as the first argument."
    echo "Usage: ./setup_venv.sh /path/to/your/vfolder"
    exit 1
fi

VFOLDER_NAME=$1
# --------------------------------------------------------

VENV_PATH="/pipeline/vfroot/.venv"
REQUIREMENTS_PATH="/${VFOLDER_NAME}/backend.ai-fasttrack-examples/pytorch/gemma-3n/pipeline-code/requirements.txt"

echo "Using base path: $VFOLDER_NAME"

# 가상 환경이 존재하는지 확인
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found. Creating a new one at $VENV_PATH"
    python -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    pip install -r "$REQUIREMENTS_PATH"
else
    echo "Existing virtual environment found at $VENV_PATH. Skipping creation and installation."
fi
# 가상 환경 활성화 및 패키지 설치
echo "✅ Virtual environment setup complete."