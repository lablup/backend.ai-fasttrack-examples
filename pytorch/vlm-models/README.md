# VLM Fine-tuning Pipeline

이 프로젝트는 Vision-Language Model (VLM)에 대해 Hugging Face 데이터셋으로 파인튜닝하고 평가하는 파이프라인을 제공합니다.

**주요 특징**:

-   🤖 **자동 특수 토큰 감지**: 모델의 tokenizer에서 모든 특수 토큰을 자동으로 감지하고 설정
-   🔄 **범용 VLM 지원**: 다양한 VLM 모델 아키텍처를 자동으로 지원
-   🛡️ **안전한 설정**: 사용자의 수동 토큰 설정 오류를 방지하는 자동화 시스템
-   📊 **포괄적 평가**: VLM 태스크에 특화된 평가 지표 제공

**주요 변경사항**: 기존 Language Model 파이프라인을 VLM 파이프라인으로 전환하였습니다.

## ✨ 새로운 자동 특수 토큰 감지 시스템

**더 이상 특수 토큰을 수동으로 설정할 필요가 없습니다!**

### 🔍 자동 감지 기능

-   **4단계 자동 감지**: Core tokens → Additional tokens → Visual tokens → Template compatibility
-   **모델별 적응**: 각 VLM 모델의 특수 토큰을 자동으로 파악
-   **안전한 처리**: 잘못된 토큰 설정으로 인한 학습 오류 방지

### 🎯 감지되는 토큰 유형

-   **기본 토큰**: pad, eos, bos, unk, sep, cls, mask
-   **VLM 토큰**: `<image>`, `<video>`, `<|vision_start|>`, `<|vision_end|>` 등
-   **모델별 토큰**: Qwen2VL, LLaVA, PaliGemma 등의 모델 특화 토큰
-   **템플릿 토큰**: apply_chat_template에서 사용되는 토큰들

## 🚀 Quick Start

### 환경 설정

**1. 자동 환경 설정 (권장)**

```bash
# vlm 프로젝트 루트(pytorch/vlm-models/)에서 실행
# 첫 번째 인자로 vFolder base path를 반드시 넘겨야 합니다.
bash setup_env.sh /home/work/<your-vfolder>
```

가상환경은 Backend.AI 파이프라인 job 안에서는 `/pipeline/vfroot/.venv`에,
그 외의 환경에서는 첫 번째 인자로 넘긴 경로 아래(`<vfolder>/.venv`)에 생성됩니다.
이미 가상환경이 존재한다면 건너뛰어도 됩니다.

**2. 수동 환경 설정**

```bash
cd pipeline-code
python3 -m venv .vlm
source .vlm/bin/activate  # Windows: .vlm\Scripts\activate
pip install -r requirements.txt
```

⚠️ V100 등 Ampere 이전 GPU에는 native bfloat16이 없습니다. 설정 파일의
`bf16: true`는 그대로 두어도 됩니다 — 파이프라인이 compute capability를 확인해
자동으로 fp16으로 전환하고 그 이유를 로그에 남깁니다.

**3. 환경 변수 설정**
`pipeline-code/.env` 파일에서 다음 값들을 확인하세요:

-   `HF_TOKEN`: Hugging Face 토큰
-   `MODEL_ID`: 사용할 VLM 모델 repository 이름. 기본값은 `Qwen/Qwen2-VL-2B-Instruct`
-   `DATASET`: 사용할 데이터셋 repository 이름. 기본값은 `philschmid/amazon-product-descriptions-vlm`
-   `VLM_MODEL_CONFIG`: VLM 모델 설정 파일 (기본값: `vlm_model_config.yaml`)
-   `VLM_COLLATOR_CONFIG`: VLM 데이터 콜레이터 설정 파일 (기본값: `vlm_collator_config.yaml`)
-   `WANDB_API_KEY`: Weights & Biases API 키 (선택사항)

### 📚 VLM 모델 및 데이터셋 설정 가이드

#### 지원되는 VLM 모델

`configs/vlm_model_config.yaml`에 클래스 매핑이 포함된 모델들:

-   **Qwen2-VL**: `Qwen/Qwen2-VL-2B-Instruct`, `Qwen/Qwen2-VL-7B-Instruct`
-   **SmolVLM**: `HuggingFaceTB/SmolVLM-Base`

그 외의 VLM은 `default_fallback` 설정에 따라 `AutoModelForImageTextToText` + `AutoProcessor`로
로드됩니다. 전용 클래스가 필요한 모델은 아래 방법으로 직접 매핑을 추가하세요.

#### 모델별 클래스 설정

`configs/vlm_model_config.yaml`에서 모델별 클래스를 설정할 수 있습니다:
**주의!! 이때, 각 모델에서 사용하는 processor는 apply_chat_template을 사용할 수 있어야 합니다.**
```yaml
model_classes:
    "Qwen/Qwen2-VL-2B-Instruct":
        model_class: "Qwen2VLForConditionalGeneration"
        # processor_class: "Qwen2VLProcessor" # 생략하면 AutoProcessor 사용
        import_path: "transformers"
```

⚠️ `loading_params.trust_remote_code`와 `processor_params.trust_remote_code`는 기본값이
`false`입니다. 모델 저장소의 임의 Python 코드를 실행하므로, 직접 검토한 모델에 한해서만
`true`로 바꾸세요.

#### 데이터 콜레이터 설정

`configs/vlm_collator_config.yaml`에서 다양한 이미지-텍스트 데이터셋에 맞게 설정할 수 있습니다:
`dataset_columns`의 key 값은 message_format 안에서 매핑될 변수의 이름이 됩니다(별칭으로 뒤 _columns는 제외해도 됩니다.)
예를 들어, 'question_column'이 `dataset_columns`의 key 값이라면 해당 값이 message_format에서 'question' 혹은 'question_column'으로 변수가 매핑되어야 합니다.
**주의!! role : "assistant"에게 매핑되는 데이터의 key 값은 반드시 answer_column으로 해야 합니다.**

```yaml
dataset_columns:
    image_column: "image"
    product_name_column: "Product Name"
    category_column: "Category"
    answer_column: "description"

message_format:
    system_prompt: "You are an expert product description writer for Amazon."
    training_messages:
        - role: "system"
          content:
              - type: "text"
                text: "{system_prompt}"
        - role: "user"
          content:
              - type: "image"
              - type: "text"
                text: "##PRODUCT NAME##: {product_name} \n##CATEGORY##: {category}"
        - role: "assistant"
          content:
              - type: "text"
                text: "{answer}"
```

위 설정 중에서 -type: "image"에 해당하는 설정의 순서는 message_format에서 바꿔도 파인튜닝 과정 중에서 `role: "system"`과 `role: "user"` 사이로 순서가 고정됩니다.

### VLM 파이프라인 실행

#### 방법 1: 전체 파이프라인 한 번에 실행

```bash
# pytorch/vlm-models/ 에서 실행 (scripts/는 pipeline-code/의 형제 디렉터리입니다)
python scripts/vlm_cli.py pipeline \
    --train_config_path train_config.yaml \
    --peft_config_path peft_config.yaml \
    --vlm_model_config vlm_model_config.yaml \
    --vlm_collator_config vlm_collator_config.yaml
```

#### 방법 2: 개별 태스크 실행 (권장)

```bash
# pytorch/vlm-models/ 에서 실행
# Task 1: 데이터셋 다운로드
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

모든 출력 경로는 `settings.py`의 `base_path`, 즉 **`pipeline-code/`** 하위에 생성됩니다
(`vlm-models/` 가 아닙니다).

-   **데이터셋 저장 경로**: `pipeline-code/dataset/`
-   **PEFT 어댑터 저장 경로**: `pipeline-code/results/models/`
-   **배포용 모델 저장 경로**: `pipeline-code/results/deployment_model/`
-   **평가 결과 저장**: `pipeline-code/results/evaluation/`

각 경로는 `.env`의 `SAVE_DATASET_PATH`, `SAVE_MODEL_PATH`, `DEPLOYMENT_MODEL_PATH`,
`EVALUATION_OUTPUT_PATH`로 변경할 수 있습니다.

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

### Task 1: Dataset Download

-   **파일**: `src/data/download_dataset.py`
-   **입력**: 이미지 컬럼을 가진 Hugging Face 데이터셋
    (기본값 `philschmid/amazon-product-descriptions-vlm`: `image` / `Product Name` /
    `Category` / `description`)
-   **처리**: 데이터셋을 train/validation/test로 분할
-   **출력**: `dataset/raw/` 폴더에 원본 데이터셋 저장 (이미지 포함)

### Task 2: 학습 전 Base VLM Model Evaluation

-   **입력**: 원본 VLM 모델, 테스트 데이터
-   **처리**: ROUGE, BLEU, BERTScore 지표로 베이스 모델 성능 평가
-   **출력**: `results/evaluation/base_model_evaluation.json`

### Task 3: VLM Model Fine-tuning

-   **파일**: `src/training/vlm_trainer.py`
-   **입력**: 베이스 VLM 모델, 원본 데이터 (이미지+텍스트), 설정 파일들
-   **처리**:
    -   VLM 전용 데이터 콜레이터를 통해 이미지와 텍스트 동시 처리
    -   LoRA를 사용한 파라미터 효율적 파인튜닝
    -   모델별 최적화된 클래스 사용
-   **출력**:
    -   PEFT 어댑터: `results/models/` 폴더
    -   배포용 완전한 모델: `results/deployment_model/` 폴더

### Task 4: Fine-tuned VLM Model Evaluation

-   **입력**: 파인튜닝된 VLM 모델, 테스트 데이터
-   **처리**: ROUGE, BLEU, BERTScore 지표로 파인튜닝된 모델 성능 평가
-   **출력**: `results/evaluation/finetuned_model_evaluation.json`

## 🆕 VLM 특화 기능

### 1. 모델별 클래스 자동 선택

-   `vlm_model_config.yaml`을 통해 모델별 최적 클래스 자동 선택
-   fallback 시스템으로 호환성 보장

### 2. VLM 데이터 콜레이터

-   이미지와 텍스트를 동시에 처리하는 커스텀 콜레이터
-   다양한 이미지-텍스트 데이터셋 형식 지원
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
