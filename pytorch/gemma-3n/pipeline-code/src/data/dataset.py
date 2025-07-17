import os
import argparse
from datasets import load_dataset, get_dataset_split_names, DatasetDict
from pathlib import Path
from transformers import AutoTokenizer, AutoProcessor
from configs.settings import settings

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Loader")
    parser.add_argument('--model-id', type=str, default=os.getenv('MODEL_ID'),
                        help='ID of the model to load from Hugging Face Hub')
    parser.add_argument('--dataset-name', type=str, default=os.getenv('DATASET'),
                        help='Name of the dataset to load from Hugging Face Hub')
    parser.add_argument('--trust_remote_code', action='store_true', default=False,
                    help='Trust remote code for datasets that require it.')
    parser.add_argument('--HF_TOKEN', type=str, default=os.getenv('HF_TOKEN'),
                        help='Hugging Face API token for authentication.')

    return parser.parse_args()

def load_and_split_dataset(dataset_name, hf_token, trust_remote_code=False) -> DatasetDict:
    """허깅페이스 데이터셋을 불러와 'train', 'validation', 'test' 스플릿을 보장하는 함수."""
    print(f"Loading dataset: {dataset_name}")

    hub_kwargs = {
        "token": hf_token,
        "trust_remote_code": trust_remote_code
    }
    
    # trust_remote_code가 False이면 딕셔너리에서 해당 키를 제거
    if not trust_remote_code:
        del hub_kwargs["trust_remote_code"]
    
    available_splits = set(get_dataset_split_names(dataset_name, **hub_kwargs))
    print(f"Available splits: {list(available_splits)}")

    if 'train' not in available_splits:
        raise ValueError(f"Dataset '{dataset_name}' must have a 'train' split.")

    # 모든 스플릿 로드
    dataset = load_dataset(dataset_name, token=hf_token, trust_remote_code=trust_remote_code)
    
    # 각 시나리오를 if/elif로 명확하게 분리
    has_train = 'train' in dataset
    has_val = 'validation' in dataset
    has_test = 'test' in dataset

    # Case 1: 모든 스플릿이 존재하는 이상적인 경우
    if has_train and has_val and has_test:
        print("Found 'train', 'validation', and 'test' splits. No modification needed.")
        return dataset
    
    # Case 2: train 스플릿만 존재하는 경우 -> 80/10/10으로 분할
    elif has_train and not has_val and not has_test:
        print("Only 'train' split found. Creating 'validation' (10%) and 'test' (10%) splits.")
        # 1. train -> train (80%) / temp (20%)
        train_test_split = dataset['train'].train_test_split(test_size=0.2, shuffle=True, seed=42)
        # 2. temp (20%) -> validation (10%) / test (10%)
        val_test_split = train_test_split['test'].train_test_split(test_size=0.5, shuffle=True, seed=42)
        
        return DatasetDict({
            'train': train_test_split['train'],
            'validation': val_test_split['train'],
            'test': val_test_split['test']
        })
        
    # Case 3: train과 validation 스플릿만 존재하는 경우 -> validation을 50/50으로 분할
    elif has_train and has_val and not has_test:
        print("Found 'train' and 'validation'. Creating 'test' split from 'validation'.")
        val_test_split = dataset['validation'].train_test_split(test_size=0.5, shuffle=True, seed=42)
        dataset['test'] = val_test_split['test']
        dataset['validation'] = val_test_split['train']
        return dataset

    # Case 4: train과 test 스플릿만 존재하는 경우 -> test를 50/50으로 분할
    elif has_train and not has_val and has_test:
        print("Found 'train' and 'test'. Creating 'validation' split from 'test'.")
        val_test_split = dataset['test'].train_test_split(test_size=0.5, shuffle=True, seed=42)
        dataset['validation'] = val_test_split['train']
        dataset['test'] = val_test_split['test']
        return dataset
        
    return dataset

def formatting_prompts(examples, processor):
    """학습용 데이터셋을 포맷팅하는 함수"""
    questions = examples["Open-ended Verifiable Question"]
    complex_cots = examples["Complex_CoT"]
    responses = examples["Response"]
    
    system_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response."""

    texts = []
    for question, cot, response in zip(questions, complex_cots, responses):
        assistant_content = f"<think>\n{cot}\n</think>\n{response}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content}
        ]
        formatted_prompt = processor.apply_chat_template(messages, tokenize=False)
        if not formatted_prompt.endswith(processor.tokenizer.eos_token):
            formatted_prompt += processor.tokenizer.eos_token
        texts.append(formatted_prompt)
        
    return {"text": texts}



class DatasetLoader:
    def __init__(self, model_id: str, dataset_name: str, trust_remote_code: bool = False, hf_token: str = None):
        
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.trust_remote_code = trust_remote_code
        self.hf_token = hf_token
        # 프로세서는 '모델 ID'를 기준으로 불러와야 합니다.
        self.processor = AutoProcessor.from_pretrained(self.model_id, token=self.hf_token)
        self.processor.tokenizer.padding_side = "left"


    def load_and_process(self) -> DatasetDict:
        """
        데이터셋 로딩부터 전처리까지의 전체 과정을 실행하고 최종 데이터셋을 반환합니다.
        """
        # 1. 원본 데이터셋 로드 및 스플릿 보장
        raw_dataset = load_and_split_dataset(self.dataset_name, self.hf_token, self.trust_remote_code)

        # 2. train/validation 스플릿만 포맷팅
        print("Formatting 'train' and 'validation' splits...")
        train_val_to_format = DatasetDict({
            'train': raw_dataset['train'],
            'validation': raw_dataset['validation']
        })
        
        formatted_train_val = train_val_to_format.map(
            formatting_prompts,
            batched=True,
            fn_kwargs={'processor': self.processor},
            desc="Formatting training and validation data" # 진행률 표시줄 설명 추가
        )
        
        # 3. 포맷팅된 스플릿과 원본 test 스플릿을 합쳐 최종 데이터셋 구성
        final_dataset = DatasetDict({
            'train': formatted_train_val['train'],
            'validation': formatted_train_val['validation'],
            'test': raw_dataset['test']  # test 스플릿은 원본을 그대로 유지
        })
        
        print("\n--- Dataset processing complete ---")
        print("The number of samples in each split:")
        print(f"Train: {len(final_dataset['train'])}, Validation: {len(final_dataset['validation'])}, Test: {len(final_dataset['test'])}")
        print("Preview of formatted train data's first sample:")
        print(final_dataset['train'][0]['text'])
        print("\nPreview of raw test data's first sample:")
        print(final_dataset['test'][0])
        
        return final_dataset

def main():
    args = parse_args()
    dataset_name = args.dataset_name

    if not dataset_name:
        print("Dataset name is not provided. Please set the DATASET environment variable.")
        return

    dataset_loader = DatasetLoader(
        model_id=args.model_id,
        dataset_name=dataset_name,
        trust_remote_code=args.trust_remote_code,
        hf_token=args.HF_TOKEN
        )
    
    # 2. 데이터 처리 실행
    processed_dataset = dataset_loader.load_and_process()

    save_dataset_path = settings.save_dataset_path
    print(f"Saving processed dataset to: {save_dataset_path}")
    if not save_dataset_path.exists():
        save_dataset_path.mkdir(parents=True, exist_ok=True)

    processed_dataset.save_to_disk(settings.save_dataset_path)

if __name__ == "__main__":
    main()