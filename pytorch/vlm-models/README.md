# VLM Fine-tuning Pipeline

이 프로젝트는 Vision-Language Model (VLM)에 대해 Hugging Face 데이터셋으로 파인튜닝하고 평가하는 파이프라인을 제공합니다.

**주요 변경사항**: 기존 Language Model 파이프라인을 VLM 파이프라인으로 전환하였습니다.

## 🚀 Quick Start

### 환경 설정

**1. 자동 환경 설정 (권장)**

```bash
# vlm 프로젝트 루트에서 실행
bash setup_env.sh
```

이미 가상환경이 존재한다면 건너뛰어도 됩니다.

**2. 수동 환경 설정**

```bash
cd pipeline-code
python -m venv .vlm
source .vlm/bin/activate  # Windows: .vlm\Scripts\activate
pip install -r requirements.txt
```

**3. 환경 변수 설정**
`pipeline-code/.env` 파일에서 다음 값들을 확인하세요:

-   `HF_TOKEN`: Hugging Face 토큰
-   `MODEL_ID`: 사용할 VLM 모델 repository 이름. 기본값은 `Qwen/Qwen2-VL-2B-Instruct`
-   `DATASET`: 사용할 VQA 데이터셋 repository 이름. 기본값은 `HuggingFaceM4/VQAv2`
-   `VLM_MODEL_CONFIG`: VLM 모델 설정 파일 (기본값: `vlm_model_config.yaml`)
-   `VLM_COLLATOR_CONFIG`: VLM 데이터 콜레이터 설정 파일 (기본값: `vlm_collator_config.yaml`)
-   `WANDB_API_KEY`: Weights & Biases API 키 (선택사항)

### 📚 VLM 모델 및 데이터셋 설정 가이드

#### 지원되는 VLM 모델

현재 설정된 VLM 모델들:

-   **Qwen2-VL**: `Qwen/Qwen2-VL-2B-Instruct`, `Qwen/Qwen2-VL-7B-Instruct`
-   **LLaVA**: `llava-hf/llava-1.5-7b-hf`, `llava-hf/llava-1.5-13b-hf`
-   **InternVL**: `OpenGVLab/InternVL2-2B`
-   **PaliGemma**: `google/paligemma-3b-pt-448`
-   **Phi-3-Vision**: `microsoft/Phi-3-vision-128k-instruct`

#### 모델별 클래스 설정

`configs/vlm_model_config.yaml`에서 모델별 클래스를 설정할 수 있습니다:

```yaml
model_classes:
    "Qwen/Qwen2-VL-2B-Instruct":
        model_class: "Qwen2VLForConditionalGeneration"
        processor_class: "Qwen2VLProcessor"
        import_path: "transformers"
```

#### 데이터 콜레이터 설정

`configs/vlm_collator_config.yaml`에서 다양한 VQA 데이터셋에 맞게 설정할 수 있습니다:

```yaml
dataset_columns:
    image_column: "image"
    question_column: "question"
    answer_column: "answer"

message_format:
    system_prompt: "Answer briefly."
    training_messages:
        - role: "user"
          content:
              - type: "text"
                text: "{system_prompt}"
              - type: "image"
              - type: "text"
                text: "{question}"
        - role: "assistant"
          content:
              - type: "text"
                text: "{answer}"
```

### VLM 파이프라인 실행

#### 방법 1: 전체 파이프라인 한 번에 실행

```bash
cd pipeline-code
python scripts/vlm_cli.py pipeline \
    --train_config_path train_config.yaml \
    --peft_config_path peft_config.yaml \
    --vlm_model_config vlm_model_config.yaml \
    --vlm_collator_config vlm_collator_config.yaml
```

#### 방법 2: 개별 태스크 실행 (권장)

```bash
# Task 1: VQA 데이터셋 다운로드
python scripts/vlm_cli.py download-dataset

# Task 2: 베이스 VLM 모델 평가
python scripts/vlm_cli.py eval-base

# Task 3: VLM 모델 파인튜닝
python scripts/vlm_cli.py train \
    --train_config_path train_config.yaml \
    --peft_config_path peft_config.yaml \
    --vlm_model_config vlm_model_config.yaml \
    --vlm_collator_config vlm_collator_config.yaml

# Task 4: 파인튜닝된 VLM 모델 평가
python scripts/vlm_cli.py eval-finetuned
```

## 📁 파일 구조 및 데이터 경로

### 기본 경로 설정

-   **데이터셋 저장 경로**: `{프로젝트_루트}/dataset/`
-   **PEFT 어댑터 저장 경로**: `{프로젝트_루트}/results/model/`
-   **배포용 모델 저장 경로**: `{프로젝트_루트}/results/deployment_model/`
-   **평가 결과 저장**: `{프로젝트_루트}/results/evaluation/`

### 파일 구조

```
vlm-models/
├── README.md
├── scripts/
│   └── vlm_cli.py              # VLM 파이프라인 실행 CLI
└── pipeline-code/
    ├── configs/
    │   ├── vlm_model_config.yaml       # VLM 모델 클래스 설정
    │   ├── vlm_collator_config.yaml    # VLM 데이터 콜레이터 설정
    │   ├── train_config.yaml           # 학습 설정
    │   ├── peft_config.yaml            # PEFT 설정
    │   └── settings.py                 # 중앙 설정 관리
    ├── src/
    │   ├── data/
    │   │   ├── download_dataset.py     # Task 1: 데이터셋 다운로드
    │   │   └── collate_fn.py           # VLM 데이터 콜레이터
    │   ├── models/
    │   │   └── model.py                # VLM 모델 로더
    │   ├── training/
    │   │   └── vlm_trainer.py          # Task 3: VLM 트레이너
    │   └── evaluation/
    │       └── evaluation.py           # Task 2,4: VLM 평가
    └── requirements.txt                # VLM 의존성
```

## 🔧 태스크 상세 설명

### Task 1: VQA Dataset Download

-   **파일**: `src/data/download_dataset.py`
-   **입력**: Hugging Face VQA 데이터셋 (예: `HuggingFaceM4/VQAv2`)
-   **처리**: 데이터셋을 train/validation/test로 분할
-   **출력**: `dataset/raw/` 폴더에 원본 데이터셋 저장 (이미지 포함)

### Task 2: 학습 전 Base VLM Model Evaluation

-   **입력**: 원본 VLM 모델, VQA 테스트 데이터
-   **처리**: VQA 정확도, BLEU, ROUGE 등의 지표로 베이스 모델 성능 평가
-   **출력**: `base_vlm_model_evaluation.json`

### Task 3: VLM Model Fine-tuning

-   **파일**: `src/training/vlm_trainer.py`
-   **입력**: 베이스 VLM 모델, 원본 VQA 데이터 (이미지+텍스트), 설정 파일들
-   **처리**:
    -   VLM 전용 데이터 콜레이터를 통해 이미지와 텍스트 동시 처리
    -   LoRA를 사용한 파라미터 효율적 파인튜닝
    -   모델별 최적화된 클래스 사용
-   **출력**:
    -   PEFT 어댑터: `results/model/` 폴더
    -   배포용 완전한 모델: `results/deployment_model/` 폴더

### Task 4: Fine-tuned VLM Model Evaluation

-   **입력**: 파인튜닝된 VLM 모델, VQA 테스트 데이터
-   **처리**: VQA 정확도, BLEU, ROUGE 등의 지표로 파인튜닝된 모델 성능 평가
-   **출력**: `finetuned_vlm_model_evaluation.json`

## 🆕 VLM 특화 기능

### 1. 모델별 클래스 자동 선택

-   `vlm_model_config.yaml`을 통해 모델별 최적 클래스 자동 선택
-   fallback 시스템으로 호환성 보장

### 2. VLM 데이터 콜레이터

-   이미지와 텍스트를 동시에 처리하는 커스텀 콜레이터
-   다양한 VQA 데이터셋 형식 지원
-   설정 파일을 통한 유연한 커스터마이징

### 3. 메모리 최적화

-   VLM의 높은 메모리 사용량을 고려한 배치 크기 조정
-   Gradient checkpointing 및 mixed precision 지원

### 4. 이미지 전처리

-   PIL 이미지 자동 변환 및 RGB 변환
-   다양한 이미지 형식 지원

## 🔄 Language Model에서 VLM으로의 주요 변경사항

1. **모델 로더**: 다양한 VLM 모델 클래스 지원
2. **데이터 처리**: 이미지+텍스트 동시 처리
3. **콜레이터**: 기존 text-only에서 multimodal 콜레이터로 변경
4. **파이프라인**: 4단계 VLM 파이프라인 (기존 6단계에서 단순화)
5. **설정**: VLM 특화 설정 파일 추가

## 🚨 주의사항

-   VLM 모델은 Language Model 대비 더 많은 GPU 메모리 필요
-   이미지가 포함된 데이터셋은 용량이 클 수 있음
-   일부 VLM 모델은 특정 라이센스 동의 필요할 수 있음
