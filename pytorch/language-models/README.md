# General language model Fine-tuning Pipeline

이 프로젝트는 General language model에 대해 Hugging Face 데이터셋으로 파인튜닝하고 평가하는 파이프라인을 제공합니다.

## 🚀 Quick Start

### 환경 설정

**1. 자동 환경 설정 (권장)**
```bash
# gemma-3n 프로젝트 루트에서 실행
bash setup_env.sh
```
이미 가상환경이 존재한다면 건너뛰어도 됩니다.

**2. 수동 환경 설정**
```bash
cd pipeline-code
python -m venv .gemma3n
source .gemma3n/bin/activate
pip install -r requirements.txt
```

**3. 환경 변수 설정**
`pipeline-code/.env` 파일에서 다음 값들을 확인하세요:
- `HF_TOKEN`: Hugging Face 토큰
- `MODEL_ID`: 사용할 Hugging Face의 model repository 이름. 기본값은 `google/gemma-3n-e2b-it`
- `DATASET`: 사용할 Hugging Face의 dataset repository 이름. 기본값은 `TheFinAI/Fino1_Reasoning_Path_FinQA`
- `WANDB_API_KEY`: Weights & Biases API 키 (선택사항)

### 📚 사용자 정의 학습 데이터셋 설정 가이드

다양한 데이터셋을 파이프라인에 적용하는 방법은 [`pipeline-code/configs/README.md`](pipeline-code/configs/README.md)를 참조하세요.

### 파이프라인 실행

#### 방법 1: 전체 파이프라인 한 번에 실행
```bash
cd pipeline-code
python scripts/cli.py pipeline \
    --train_config_path train_config.yaml \
    --peft_config_path peft_config.yaml
```

#### 방법 2: 개별 태스크 실행 (권장)
```bash
# Task 1a: 데이터셋 다운로드
python scripts/cli.py download-dataset

# Task 1b: 데이터셋 전처리  
python scripts/cli.py preprocess-dataset

# Task 1b (사용자 정의 설정 사용):
python scripts/cli.py preprocess-dataset --config my_messages_format.yaml

# Task 1c: 데이터셋 포맷팅
python scripts/cli.py format-dataset

# Task 2: 베이스 모델 평가
python scripts/cli.py eval-base

# Task 3: 모델 파인튜닝
python scripts/cli.py train \
    --train_config_path train_config.yaml \
    --peft_config_path peft_config.yaml

# Task 4: 파인튜닝된 모델 평가
python scripts/cli.py eval-finetuned
```

## 📁 파일 구조 및 데이터 경로

### 기본 경로 설정
- **데이터셋 저장 경로**: `{프로젝트_루트}/dataset/`
- **PEFT 어댑터 저장 경로**: `{프로젝트_루트}/results/model/`
- **배포용 모델 저장 경로**: `{프로젝트_루트}/results/deployment_model/`
- **평가 결과 저장**: `{프로젝트_루트}/results/evaluation/`

- (FastTrack pipeline용 폴더 설계) 확장 예정

### 파일 구조
```
gemma-3n/
├── README.md
├── scripts/
│   └── cli.py              # 파이프라인 실행 CLI
└── pipeline-code
    ├── configs/
    │   ├── settings.py          # 전역 설정
    │   ├── train_config.yaml    # 훈련 인자 설정
    │   ├── peft_config.yaml     # PEFT(LoRA) 설정
    │   └── messages_format.yaml # 데이터셋 전처리 설정
    ├── data/                    # 처리된 데이터셋 저장소
    ├── results/
    │   ├── model/              # PEFT 어댑터 저장소
    │   ├── deployment_model/   # LoRA 가중치가 병합된 배포용 모델 저장소
    │   └── evaluation/         # 평가 결과 저장소
    ├── logs/                   # 훈련 로그 저장소
    ├── src/
    │   ├── data/
    │   │   ├── download_dataset.py # Task 1a: 데이터셋 다운로드
    │   │   ├── preprocess_dataset.py # Task 1b: 데이터셋 전처리
    │   │   ├── format_dataset.py   # Task 1c: 데이터셋 포맷팅
    │   │   └── dataset.py          # 통합 데이터셋 파이프라인 (호환성 유지)
    │   ├── evaluation/
    │   │   └── evaluation.py       # 모델 평가
    │   ├── models/
    │   │   └── model.py           # 모델 로딩
    │   └── training/
    │       └── trainer.py         # 모델 파인튜닝
    ├── .env                    # 환경 변수
    └── requirements.txt        # 의존성

```

## 🔧 태스크 상세 설명

### Task 1: Dataset Pipeline (세분화된 태스크)

#### Task 1a: Dataset Download
- **파일**: `src/data/download_dataset.py`
- **입력**: Hugging Face 데이터셋 (`TheFinAI/Fino1_Reasoning_Path_FinQA`)
- **처리**: 데이터셋을 train/validation/test로 분할
- **출력**: `dataset/raw/` 폴더에 원본 데이터셋 저장

#### Task 1b: Dataset Preprocessing  
- **파일**: `src/data/preprocess_dataset.py`
- **입력**: `dataset/raw/` 폴더의 원본 데이터셋
- **설정**: `configs/messages_format.yaml` (사용자 정의 가능)
- **처리**: 
  - 설정 파일 기반 데이터 전처리 (다양한 데이터셋 지원)
  - 컬럼 매핑을 통한 일반화된 전처리
  - 데이터 품질 필터링, 정제
  - messages 구조로 변환 (train/validation: 학습용, test: 평가용)
  - system prompt, user, assistant content 구조화
- **출력**: `dataset/preprocessed/` 폴더에 messages 구조의 데이터셋 저장

#### ⚙️ 사용자 정의 데이터셋 설정
`configs/messages_format.yaml` 파일을 통해 다양한 데이터셋에 적용 가능:
다양한 데이터셋을 파이프라인에 적용하는 자세한 방법은 [`pipeline-code/configs/README.md`](pipeline-code/configs/README.md)를 참조하세요.

```yaml
# 컬럼 매핑 (key: 변수명, value: 실제 데이터셋 컬럼명)
training_columns:
  question: "Open-ended Verifiable Question"
  cot: "Complex_CoT" 
  response: "Response"

# messages 포맷 템플릿
messages_format:
  system_prompt: |
    Below is an instruction that describes a task...
  messages:
    - role: "system"
      content: "{system_prompt}"
    - role: "user"
      content: "{question}"
    - role: "assistant"
      content: "<think>\n{cot}\n</think>\n{response}"

# 평가용 컬럼 매핑
evaluate_columns:
  query: "Open-ended Verifiable Question"
  response: "Response"
```

#### Task 1c: Dataset Formatting
- **파일**: `src/data/format_dataset.py`
- **입력**: `dataset/preprocessed/` 폴더의 messages 구조 데이터셋
- **처리**: 
  - processor.apply_chat_template 적용
  - train/validation: 학습용 텍스트 생성
  - test: 평가용 프롬프트와 reference 생성
- **출력**: `dataset/formatted/` 폴더에 최종 포맷팅된 데이터셋 저장

### Task 2: 학습 전 Base Model Evaluation
- **입력**: 원본 Gemma-3n 모델, 전처리된 테스트 데이터
- **처리**: ROUGE, BERTScore 등의 지표로 베이스 모델 성능 평가
- **출력**: `base_model_evaluation.json`

### Task 3: Model Fine-tuning
- **입력**: 베이스 모델, 포맷팅된 훈련/검증 데이터, 설정 파일들
- **처리**: LoRA를 사용한 파라미터 효율적 파인튜닝
- **출력**: 
  - PEFT 어댑터: `results/model/` 폴더
  - 배포용 완전한 모델: `results/deployment_model/` 폴더 (LoRA 가중치 병합됨)

### Task 4: Fine-tuned Model Evaluation
- **입력**: 파인튜닝된 모델, 전처리된 테스트 데이터
- **처리**: 파인튜닝된 모델의 성능 평가
- **출력**: `finetuned_model_evaluation.json`

## 📊 설정 커스터마이징

### 훈련 인자 수정 (`configs/train_config.yaml`)
- 배치 크기, 학습률, 에포크 수 등 조정 가능
- Weights & Biases 로깅 설정

### PEFT 설정 수정 (`configs/peft_config.yaml`)
- LoRA rank, alpha, dropout 등 조정 가능
- 타겟 모듈 선택

## 🚨 주의사항

1. **GPU 메모리**: Gemma-3n은 대용량 모델이므로 충분한 GPU 메모리가 필요합니다.
2. **Hugging Face 토큰**: 모델 접근을 위해 유효한 HF 토큰이 필요합니다.
3. **데이터 저장 공간**: 데이터셋과 모델 저장을 위한 충분한 디스크 공간을 확보하세요.

## 📈 성능 모니터링

- Weights & Biases를 통한 실시간 훈련 모니터링
- 평가 결과는 JSON 파일로 저장되어 성능 비교 가능

## 🚀 배포 및 사용

### 파인튜닝된 모델 사용 방법

파인튜닝 완료 후, 병합된 모델을 다음과 같이 사용할 수 있습니다:

```python
from transformers import AutoModelForCausalLM, AutoProcessor

# 1. 로컬에서 배포용 모델 로드
model = AutoModelForCausalLM.from_pretrained("./results/deployment_model/")
processor = AutoProcessor.from_pretrained("./results/deployment_model/")

# 2. 추론 실행
messages = [{"role": "user", "content": "What is the impact of inflation on stock prices?"}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
response = processor.decode(outputs[0], skip_special_tokens=True)
```

### Hugging Face Hub에 배포

```bash
# Hugging Face CLI 설치 및 로그인
pip install huggingface_hub
huggingface-cli login

# 모델 업로드
huggingface-cli upload your-username/gemma-3n-financial-qa ./results/deployment_model/
```

### 모델 구조

- **PEFT 어댑터**: `results/model/` - LoRA 가중치만 포함, 작은 파일 크기
- **배포용 모델**: `results/deployment_model/` - LoRA 가중치가 병합된 완전한 모델, 독립적으로 사용 가능
