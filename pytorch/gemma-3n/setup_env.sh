#!/bin/bash
# Gemma-3n 파인튜닝 환경 설정 스크립트

echo "🚀 Gemma-3n Fine-tuning Environment Setup"
echo "========================================="

# 현재 위치 확인
if [[ ! -d "pipeline-code" ]]; then
    echo "❌ Error: pipeline-code 폴더를 찾을 수 없습니다."
    echo "   gemma-3n 프로젝트 루트 디렉토리에서 실행해주세요."
    exit 1
fi

# 가상환경 디렉토리 확인
if [[ ! -d "pipeline-code/.gemma3n" ]]; then
    echo "❌ Error: .gemma3n 가상환경을 찾을 수 없습니다."
    echo "   먼저 가상환경을 생성해주세요:"
    echo "   cd pipeline-code && python -m venv .gemma3n"
    exit 1
fi

# 가상환경 활성화
echo "🔧 가상환경 활성화 중..."
source pipeline-code/.gemma3n/bin/activate

# 의존성 설치 확인
echo "📦 의존성 설치 확인 중..."
cd pipeline-code
pip install -r requirements.txt

echo "✅ 환경 설정 완료!"
echo ""
echo "🎯 사용 가능한 명령어:"
echo "   python scripts/cli.py --help"
echo ""
echo "🔄 파이프라인 실행 예시:"
echo "   python scripts/cli.py download-dataset"
echo "   python scripts/cli.py preprocess-dataset"
echo "   python scripts/cli.py format-dataset"
echo ""
echo "💡 종료할 때는 'deactivate' 명령어를 사용하세요."

# 새로운 셸 시작
exec bash
