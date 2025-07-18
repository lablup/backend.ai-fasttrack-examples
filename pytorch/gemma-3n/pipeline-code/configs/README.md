# 📚 Dataset Configuration 완전 가이드

이 폴더에는 다양한 데이터셋에 대한 설정 예시와 상세한 튜토리얼이 포함되어 있습니다.

## 🎯 기본 설정 파일

- **`dataset_config.yaml`**: TheFinAI/Fino1_Reasoning_Path_FinQA 데이터셋용 기본 설정
- **`my_dataset_config.yaml`**: 사용자 정의 데이터셋 예시 (문서 요약용)

## 🔧 설정 파일 구조 완전 이해

### 1. **training_columns** (템플릿 변수 매핑)

```yaml
training_columns:
  question: "Open-ended Verifiable Question"  # key → value 매핑
  cot: "Complex_CoT"
  response: "Response"
```

**변수 설:**
- **key (예: `question`, `cot`, `response`)**: `messages_format`에서 사용할 **템플릿 변수명**
- **value (예: `"Open-ended Verifiable Question"`)**: HuggingFace 데이터셋의 **실제 컬럼명**

**⚠️ 중요한 규칙:**
1. `training_columns`의 **key**는 반드시 `messages_format`의 `content`에서 `{key}` 형태로 사용되어야 함
2. `training_columns`의 **value**는 실제 데이터셋에 존재하는 컬럼명이어야 함

### 2. **messages_format** (메시지 템플릿 구조)

```yaml
messages_format:
  system_prompt: |
    Below is an instruction that describes a task...
  
  messages:
    - role: "system"
      content: "{system_prompt}"
    - role: "user" 
      content: "{question}"          # ← training_columns의 key와 일치!
    - role: "assistant"
      content: "<think>\n{cot}\n</think>\n{response}"  # ← 여러 변수 조합 가능
```

**변수 설:**
- **system_prompt**: 모든 대화에 사용될 시스템 프롬프트 (선택사항)
- **messages**: 실제 대화 구조를 정의하는 리스트
  - **role**: `"system"`, `"user"`, `"assistant"` 중 하나
  - **content**: 실제 내용 (중괄호 `{}` 안에 변수명 사용)

**⚠️ 변수 사용 규칙:**
- `{question}` → `training_columns`에서 `question` key에 매핑된 컬럼의 데이터
- `{system_prompt}` → `messages_format`에서 정의한 system_prompt
- `role: "system"`은 사용하는 모델에 따라 없을 수도 있으니 주의가 필요합니다.
- 여러 변수 조합 가능: `"<think>\n{cot}\n</think>\n{response}"`

### 3. **evaluate_columns** (평가용 컬럼 매핑)

```yaml
evaluate_columns:
  query: "Open-ended Verifiable Question"    # 질문 컬럼
  response: "Response"                       # 정답 컬럼
```

**변수 설명:**
- **query**: 평가 시 모델에게 입력할 질문이 담긴 컬럼명
- **response**: 평가 시 정답으로 사용할 응답이 담긴 컬럼명
- **고정 key**: 반드시 `query`와 `response` key를 사용해야 함

### 4. **data_filtering** (데이터 품질 필터링)

```yaml
data_filtering:
  required_columns:
    - "Open-ended Verifiable Question"
    - "Complex_CoT"
    - "Response"
  min_text_length: 1
```

**변수 설명**
- **required_columns**: None 값이나 빈 문자열을 가진 샘플을 제거할 컬럼들
- **min_text_length**: 각 컬럼의 최소 텍스트 길이

## 🚀 단계별 사용자 정의 설정 가이드

### **Step 1: 데이터셋 구조 파악**

```python
from datasets import load_dataset

# 사용하고자 하는 데이터셋 로드
dataset = load_dataset("your_dataset_name")

# 컬럼 구조 확인
print("Available columns:", dataset["train"].column_names)

# 샘플 데이터 확인
print("Sample data:")
for i in range(2):
    print(f"Sample {i}:")
    for col in dataset["train"].column_names:
        print(f"  {col}: {dataset['train'][i][col]}")
```

### **Step 2: 변수명과 컬럼 매핑 설계**

**예시 1: Q&A 데이터셋**
```yaml
training_columns:
  question: "question"      # 데이터셋의 'question' 컬럼
  answer: "answer"          # 데이터셋의 'answer' 컬럼

messages_format:
  system_prompt: |
    You are a helpful assistant. Answer the question accurately.
  messages:
    - role: "user"
      content: "{question}"   # ← training_columns의 'question' key 사용
    - role: "assistant"  
      content: "{answer}"     # ← training_columns의 'answer' key 사용
```

**예시 2: Chain-of-Thought 데이터셋**
```yaml
training_columns:
  problem: "math_problem"
  reasoning: "step_by_step_solution"  
  final_answer: "final_answer"

messages_format:
  messages:
    - role: "user"
      content: "Solve this problem: {problem}"
    - role: "assistant"
      content: "Let me think step by step:\n{reasoning}\n\nFinal Answer: {final_answer}"
```

### **Step 3: 설정 파일 생성**

```yaml
# configs/my_custom_config.yaml
training_columns:
  input_text: "your_input_column_name"
  output_text: "your_output_column_name"

messages_format:
  system_prompt: |
    Your custom system prompt here...
  messages:
    - role: "user"
      content: "{input_text}"
    - role: "assistant"
      content: "{output_text}"

evaluate_columns: #evaluate_columns의 key는 'query'와 'response'가 고정입니다.
  query: "your_input_column_name"
  response: "your_output_column_name"

data_filtering:
  required_columns:
    - "your_input_column_name"
    - "your_output_column_name"
  min_text_length: 5
```

### **Step 4: 전처리 실행**

```bash
cd pipeline-code
python src/data/preprocess_dataset.py --config configs/my_custom_config.yaml
```

## FAQ

### **실수 1: 변수명 불일치**
```yaml
# ❌ 잘못된 예시
training_columns:
  question: "input_text"
messages:
  - content: "{query}"  # ← 'question'이 아닌 'query' 사용

# ✅ 올바른 예시  
training_columns:
  question: "input_text"
messages:
  - content: "{question}"  # ← training_columns의 key와 일치
```

### **실수 2: 존재하지 않는 컬럼 참조**
```yaml
# ❌ 데이터셋에 'reasoning' 컬럼이 없는 경우
training_columns:
  reasoning: "step_by_step"  # ← 실제 컬럼명이 'reasoning_steps'라면 오류

# ✅ 실제 컬럼명 확인 후 사용
training_columns:
  reasoning: "reasoning_steps"
```

### **실수 3: evaluate_columns 구조 오류**
```yaml
# ❌ 잘못된 key 이름
evaluate_columns:
  input: "question"    # ← 'query'여야 함
  output: "answer"     # ← 'response'여야 함

# ✅ 올바른 구조
evaluate_columns:
  query: "question"
  response: "answer"
```

## 📚 추가 리소스

- **HuggingFace Datasets 문서**: https://huggingface.co/docs/datasets
- **Chat Template 가이드**: https://huggingface.co/docs/transformers/chat_templating
- **YAML 문법 가이드**: https://yaml.org/

---

💡 **도움이 필요하신가요?** 이 가이드대로 따라해도 문제가 있다면, 설정 파일과 에러 메시지를 함께 공유해 주세요!
