# Gemma-3n Financial Q&A Fine-tuning Pipeline

이 프로젝트는 Gemma-3n 모델을 financial Q&A 데이터셋으로 파인튜닝하고 평가하는 완전한 파이프라인을 제공합니다.

## 🚀 Quick Start

### 환경 설정

1. 의존성 설치:
```bash
pip install -r requirements.txt
```

2. 환경 변수 설정:
`.env` 파일에서 다음 값들을 확인하세요:
- `HF_TOKEN`: Hugging Face 토큰
- `MODEL_ID`: 기본값은 `google/gemma-3n-e2b-it`
- `DATASET`: 기본값은 `TheFinAI/Fino1_Reasoning_Path_FinQA`
- `WANDB_API_KEY`: Weights & Biases API 키 (선택사항)

### 파이프라인 실행

#### 방법 1: 전체 파이프라인 한 번에 실행
```bash
python scripts/cli.py pipeline \
    --training_args_path configs/training_args.yaml \
    --peft_config_path configs/peft_config.yaml
```

#### 방법 2: 개별 태스크 실행
```bash
# 1. 데이터셋 다운로드 및 전처리
python scripts/cli.py dataset

# 2. 베이스 모델 평가
python scripts/cli.py eval-base

# 3. 모델 파인튜닝
python scripts/cli.py train \
    --training_args_path training_args.yaml \
    --peft_config_path peft_config.yaml

# 4. 파인튜닝된 모델 평가
python scripts/cli.py eval-finetuned
```

## 📁 파일 구조 및 데이터 경로

### 기본 경로 설정
- **데이터셋 저장 경로**: `{프로젝트_루트}/dataset/`
- **PEFT 어댑터 저장 경로**: `{프로젝트_루트}/results/model/`
- **병합된 모델 저장 경로**: `{프로젝트_루트}/results/merged_model/`
- **평가 결과 저장**: `{프로젝트_루트}/results/evaluation/`

### 파일 구조
```
gemma-3n/
├── README.md
├── scripts/
│   └── cli.py              # 파이프라인 실행 CLI
└── pipeline-code
    ├── configs/
    │   ├── settings.py          # 전역 설정
    │   ├── training_args.yaml   # 훈련 인자 설정
    │   └── peft_config.yaml     # PEFT(LoRA) 설정
    ├── data/                    # 처리된 데이터셋 저장소
    ├── results/
    │   ├── model/              # PEFT 어댑터 저장소
    │   ├── merged_model/       # 병합된 모델 저장소 (배포용)
    │   └── evaluation/         # 평가 결과 저장소
    ├── logs/                   # 훈련 로그 저장소
    ├── src/
    │   ├── data/
    │   │   └── dataset.py      # 데이터셋 로딩 및 전처리
    │   ├── evaluation/
    │   │   └── evaluation.py   # 모델 평가
    │   ├── models/
    │   │   └── model.py        # 모델 로딩
    │   └── training/
    │       └── trainer.py      # 모델 파인튜닝
    ├── .env                    # 환경 변수
    └── requirements.txt        # 의존성

```

## 🔧 태스크 상세 설명

### Task 1: Dataset Download & Preprocessing
- **입력**: Hugging Face 데이터셋 (`TheFinAI/Fino1_Reasoning_Path_FinQA`)
- **처리**: 데이터셋을 train/validation/test로 분할하고 채팅 템플릿 적용
- **출력**: `dataset/` 폴더에 전처리된 데이터셋 저장

### Task 2: Base Model Evaluation
- **입력**: 원본 Gemma-3n 모델, 전처리된 테스트 데이터
- **처리**: ROUGE, BERTScore 등의 지표로 베이스 모델 성능 평가
- **출력**: `base_model_evaluation.json`

### Task 3: Model Fine-tuning
- **입력**: 베이스 모델, 전처리된 훈련/검증 데이터, 설정 파일들
- **처리**: LoRA를 사용한 파라미터 효율적 파인튜닝
- **출력**: 
  - PEFT 어댑터: `results/model/` 폴더
  - 병합된 완전한 모델: `results/merged_model/` 폴더 (배포용)

### Task 4: Fine-tuned Model Evaluation
- **입력**: 파인튜닝된 모델, 전처리된 테스트 데이터
- **처리**: 파인튜닝된 모델의 성능 평가
- **출력**: `finetuned_model_evaluation.json`

## 📊 설정 커스터마이징

### 훈련 인자 수정 (`configs/training_args.yaml`)
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

# 1. 로컬에서 병합된 모델 로드
model = AutoModelForCausalLM.from_pretrained("./results/merged_model/")
processor = AutoProcessor.from_pretrained("./results/merged_model/")

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
huggingface-cli upload your-username/gemma-3n-financial-qa ./results/merged_model/
```

### 모델 구조

- **PEFT 어댑터**: `results/model/` - LoRA 가중치만 포함, 작은 파일 크기
- **병합된 모델**: `results/merged_model/` - 완전한 모델, 독립적으로 사용 가능
