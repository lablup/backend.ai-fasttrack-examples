# 📚 VLM Configuration 가이드

이 폴더에는 Vision-Language Model (VLM) 파인튜닝 파이프라인을 위한 설정 파일들이 포함되어 있습니다.

## 🎯 VLM 설정 파일 개요

### 핵심 VLM 설정 파일들

-   **`vlm_model_config.yaml`**: VLM 모델별 클래스 매핑 및 로딩 설정
-   **`vlm_collator_config.yaml`**: VLM 데이터 처리 및 콜레이터 설정
-   **`train_config.yaml`**: 훈련 파라미터 설정
-   **`peft_config.yaml`**: LoRA 등 PEFT 설정

---

## 🚀 빠른 시작 가이드

### 1️⃣ 모델 설정 (`vlm_model_config.yaml`)

```yaml
# 사용하고 싶은 모델이 이미 설정되어 있는지 확인
model_classes:
    "Qwen/Qwen2-VL-2B-Instruct":
        model_class: "Qwen2VLForConditionalGeneration"
        processor_class: "Qwen2VLProcessor"
        import_path: "transformers"
```

### 2️⃣ 데이터 처리 설정 (`vlm_collator_config.yaml`)

```yaml
# 데이터셋 컬럼명 매핑
dataset_columns:
    image_column: "image" # 실제 데이터셋의 이미지 컬럼명
    question_column: "question" # 실제 데이터셋의 질문 컬럼명
    answer_column: "answer" # 실제 데이터셋의 답변 컬럼명

# 이미지/비디오 처리 활성화 설정
data_processing:
    image_data: true # 이미지 처리 활성화
    video_data: false # 비디오 처리 비활성화 (기본값)
```

---

## 🔧 상세 설정 가이드

## 📱 VLM 모델 설정 (`vlm_model_config.yaml`)

### 지원되는 VLM 모델들

현재 사전 설정된 모델들:

```yaml
model_classes:
    # Qwen2-VL 시리즈
    "Qwen/Qwen2-VL-2B-Instruct":
        model_class: "Qwen2VLForConditionalGeneration"
        processor_class: "Qwen2VLProcessor"
        import_path: "transformers"

    # LLaVA 시리즈
    "llava-hf/llava-1.5-7b-hf":
        model_class: "LlavaForConditionalGeneration"
        processor_class: "LlavaProcessor"
        import_path: "transformers"

    # InternVL 시리즈
    "OpenGVLab/InternVL2-2B":
        model_class: "InternVLChatModel"
        processor_class: "InternVLChatProcessor"
        import_path: "transformers"

    # PaliGemma 시리즈
    "google/paligemma-3b-pt-448":
        model_class: "PaliGemmaForConditionalGeneration"
        processor_class: "PaliGemmaProcessor"
        import_path: "transformers"

    # Phi-3-Vision 시리즈
    "microsoft/Phi-3-vision-128k-instruct":
        model_class: "Phi3VForCausalLM"
        processor_class: "Phi3VProcessor"
        import_path: "transformers"
```

### 새로운 VLM 모델 추가하기

**Step 1: 모델 클래스 정보 확인**

```python
# Hugging Face에서 모델 페이지를 확인하여 클래스명 찾기
from transformers import AutoModelForCausalLM, AutoProcessor
```

**Step 2: 설정 추가**

```yaml
model_classes:
    "your-org/your-vlm-model":
        model_class: "YourVLMModelClass" # 모델 클래스명
        processor_class: "YourVLMProcessorClass" # 프로세서 클래스명
        import_path: "transformers" # 임포트 경로
```

### 로딩 파라미터 커스터마이징

```yaml
# 공통 로딩 파라미터
loading_params:
    torch_dtype: "torch.bfloat16" # 데이터 타입: torch.float16, torch.bfloat16
    device_map: "auto" # 디바이스 매핑: auto, cuda, cpu
    trust_remote_code: true # 원격 코드 신뢰 여부

# 프로세서 공통 설정
processor_params:
    trust_remote_code: true # 프로세서 원격 코드 신뢰 여부
```

---

## 🖼️ VLM 데이터 콜레이터 설정 (`vlm_collator_config.yaml`)

### 데이터셋 컬럼 매핑

**실제 데이터셋 구조 확인 방법:**

```python
from datasets import load_dataset
dataset = load_dataset("your-dataset-name")
print("Available columns:", dataset["train"].column_names)
print("Sample data:", dataset["train"][0])
```

**컬럼 매핑 설정:**
**주의 : dataset_columns의 key 값인 image_column, question_column, answer_column 등은 아래 message format에서 사용되는 변수의 이름과 동일하게 설정해야 합니다.**

```yaml
dataset_columns:
    image_column: "image" # 데이터셋의 이미지 컬럼명
    question_column: "question" # 데이터셋의 질문 컬럼명
    answer_column: "answer" # 데이터셋의 답변 컬럼명
    video_column: "video" # 데이터셋의 비디오 컬럼명 (선택사항)
    context_column: null # 추가 컨텍스트 컬럼 (선택사항)
```

### 이미지/비디오 처리 제어

**🆕 새로운 플래그 기반 처리 제어:**

```yaml
data_processing:
    image_data: true # 이미지 데이터 처리 활성화 (기본값: true)
    video_data: false # 비디오 데이터 처리 활성화 (기본값: false)
```

**사용 시나리오:**

1. **이미지만 처리** (대부분의 VLM 모델):

```yaml
data_processing:
    image_data: true
    video_data: false
```

2. **비디오만 처리** (비디오 지원 모델):

```yaml
data_processing:
    image_data: false
    video_data: true
```

3. **둘 다 활성화** (비디오 우선 처리):

```yaml
data_processing:
    image_data: true
    video_data: true
# ⚠️ 주의: 대부분의 프로세서는 하나의 타입만 지원하므로 비디오가 우선 처리됩니다
```

### 메시지 포맷 설정

**사용하는 변수의 이름은 dataset_columns에서 정의한 key 값과 매핑되어야 합니다.**

```yaml
message_format:
    system_prompt: "Answer briefly." # 시스템 프롬프트

    # 학습용 메시지 구조
    training_messages:
        - role: "user"
          content:
              - type: "text"
                text: "{system_prompt}"
              - type: "image" # 이미지 플레이스홀더
              - type: "text"
                text: "{question_column}"
        - role: "assistant"
          content:
              - type: "text"
                text: "{answer_column}"
```

### 이미지 전처리 설정

```yaml
image_processing:
    convert_to_rgb: true # RGB 변환 (권장: true)
    resize_mode: "keep_aspect" # 리사이즈 모드: keep_aspect, crop, stretch
    max_size: null # 최대 크기 제한 (픽셀, null=제한없음)
```

### 비디오 처리 설정

```yaml
video_processing:
    enabled: true # 비디오 처리 기능 활성화

    # 프레임 추출 설정
    frame_extraction:
        library: "decord" # 비디오 라이브러리: decord (권장), cv2
        sampling_strategy: "uniform" # 샘플링 전략: uniform (균등 샘플링)
        num_frames: 8 # 추출할 프레임 수

    # 비디오 파일 처리
    file_processing:
        convert_to_rgb: true # RGB 변환
        max_duration: null # 최대 비디오 길이 (초)
```

### 텍스트 처리 설정

```yaml
text_processing:
    padding: true # 패딩 적용
    truncation: true # 자르기 적용
    max_length: 2048 # 최대 토큰 길이
    add_generation_prompt: false # generation prompt 추가 (학습시 false)
```

### 레이블 마스킹 설정

```yaml
label_masking:
    mask_pad_token: true # 패딩 토큰 마스킹
    mask_image_token: true # 이미지 토큰 마스킹
    mask_input_tokens: false # 입력 토큰 마스킹 (instruction tuning시 false)
    ignore_index: -100 # 마스킹된 토큰의 라벨 값
```

---

## ⚙️ 훈련 설정 (`train_config.yaml`)

### 기본 훈련 파라미터

```yaml
# 배치 크기 설정
per_device_train_batch_size: 8 # 훈련 배치 크기 (GPU 메모리에 따라 조정)
per_device_eval_batch_size: 8 # 평가 배치 크기
gradient_accumulation_steps: 2 # 그래디언트 누적 스텝

# 학습률 및 에포크
learning_rate: 1.0e-05 # 학습률 (VLM은 보통 낮은 학습률 사용)
num_train_epochs: 1.0 # 훈련 에포크 수

# 로깅 및 저장
logging_steps: 0.1 # 로깅 주기 (0.1 = 10%마다)
eval_steps: 0.1 # 평가 주기
save_steps: 0.1 # 모델 저장 주기
save_total_limit: 2 # 저장할 체크포인트 개수

# 메모리 최적화
gradient_checkpointing: true # 그래디언트 체크포인팅 (메모리 절약)
bf16: true # bfloat16 사용 (권장)
fp16: false # float16 사용 (bf16과 배타적)

# 데이터 처리
group_by_length: true # 길이별 그룹화 (효율성 향상)
remove_unused_columns: false # VLM에서는 false로 설정
dataloader_pin_memory: false # 메모리 핀닝
```

### WandB 설정

```yaml
# Weights & Biases 로깅
report_to: ["wandb"] # 로깅 서비스
run_name: "qwen2-vl-2b-vqa-finetuning" # 실험 이름
```

---

## 🎯 PEFT 설정 (`peft_config.yaml`)

### LoRA 설정

```yaml
# PEFT (LoRA) Configuration
task_type: "CAUSAL_LM" # 태스크 타입
r: 64 # LoRA rank (높을수록 더 많은 파라미터)
target_modules: "all-linear" # 타겟 모듈 (all-linear 권장)
lora_alpha: 16 # LoRA alpha (일반적으로 r의 1/4)
lora_dropout: 0.05 # LoRA 드롭아웃
bias: "none" # 바이어스 설정
use_rslora: false # RS-LoRA 사용 여부
use_dora: false # DoRA 사용 여부
```

---

## 🛠️ 실전 사용 예시

### 시나리오 1: 새로운 VQA 데이터셋 사용하기

**Step 1: 데이터셋 구조 확인**

```python
from datasets import load_dataset
dataset = load_dataset("your-vqa-dataset")
print(dataset["train"].column_names)
# 출력 예: ['image_path', 'query', 'response', 'metadata']
```

**Step 2: `vlm_collator_config.yaml` 수정**

```yaml
dataset_columns:
    image_column: "image_path" # 실제 이미지 컬럼명으로 변경
    question_column: "query" # 실제 질문 컬럼명으로 변경
    answer_column: "response" # 실제 답변 컬럼명으로 변경
```

### 시나리오 2: 비디오 VLM 모델 사용하기

**Step 1: `vlm_collator_config.yaml`에서 비디오 활성화**

```yaml
data_processing:
    image_data: false # 이미지 비활성화
    video_data: true # 비디오 활성화

dataset_columns:
    video_column: "video_path" # 비디오 컬럼명 설정
```

**Step 2: 비디오 처리 파라미터 조정**

```yaml
video_processing:
    frame_extraction:
        num_frames: 16 # 더 많은 프레임 추출
```

### 시나리오 3: 메모리 부족 시 최적화

**Step 1: `train_config.yaml` 수정**

```yaml
per_device_train_batch_size: 4 # 배치 크기 줄이기
gradient_accumulation_steps: 4 # 누적 스텝 늘리기
gradient_checkpointing: true # 체크포인팅 활성화
```

**Step 2: `peft_config.yaml`에서 LoRA rank 줄이기**

```yaml
r: 32 # rank 줄이기 (64 → 32)
lora_alpha: 8 # alpha도 비례해서 줄이기
```

---

## 🚨 주의사항 및 팁

### ⚠️ 주의사항

1. **이미지/비디오 동시 활성화**: 대부분의 VLM processor는 한 가지 타입만 지원
2. **메모리 관리**: VLM은 메모리 사용량이 크므로 배치 크기 조심
3. **컬럼명 확인**: 데이터셋의 실제 컬럼명과 설정이 일치해야 함
4. **모델 호환성**: 모든 VLM이 동일한 인터페이스를 제공하지 않음

### 💡 최적화 팁

1. **성능 최적화**: `bf16=true`, `gradient_checkpointing=true` 사용
2. **실험 관리**: WandB로 실험 추적 및 비교
3. **점진적 테스트**: 작은 데이터셋으로 먼저 테스트
4. **에러 디버깅**: 각 단계별로 개별 실행해서 문제 파악

---

## 📞 문제 해결

### 자주 발생하는 문제들

**Q: 새로운 모델이 인식되지 않아요**

```yaml
# vlm_model_config.yaml에 모델 추가 필요
model_classes:
    "your-model-id":
        model_class: "YourModelClass"
        processor_class: "YourProcessorClass"
        import_path: "transformers"
```

**Q: 이미지가 로드되지 않아요**

```yaml
# 컬럼명 확인 및 수정
dataset_columns:
    image_column: "실제_이미지_컬럼명" # 데이터셋에 맞게 수정
```

**Q: 메모리 부족 에러가 발생해요**

```yaml
# 배치 크기 및 LoRA rank 줄이기
per_device_train_batch_size: 2 # 더 작게
gradient_accumulation_steps: 8 # 더 크게
r: 16 # LoRA rank 줄이기
```

이 가이드를 통해 VLM 파인튜닝 파이프라인을 효과적으로 설정하고 사용할 수 있습니다! 🚀
