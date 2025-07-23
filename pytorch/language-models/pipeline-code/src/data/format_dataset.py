#!/usr/bin/env python3
"""
Task 1c: Dataset Formatting
전처리된 messages 구조의 데이터셋에 채팅 템플릿을 적용하여 모델 학습용 포맷으로 변환하는 태스크
"""

import os
import argparse
from datasets import load_from_disk, DatasetDict
from pathlib import Path
import sys

from src.models.model import ModelLoader
from configs.settings import settings

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Formatter")
    parser.add_argument('--model-id', type=str, default=os.getenv('MODEL_ID'),
                        help='ID of the model to load processor from Hugging Face Hub')
    parser.add_argument('--HF_TOKEN', type=str, default=os.getenv('HF_TOKEN'),
                        help='Hugging Face API token for authentication.')
    return parser.parse_args()

def format_training_messages(examples, tokenizer):
    """학습용 messages를 tokenizer.apply_chat_template로 포맷팅하는 함수"""
    messages_list = examples["messages"]
    references = examples["reference"]
    
    texts = []
    for messages in messages_list:
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        if not formatted_prompt.endswith(tokenizer.eos_token):
            formatted_prompt += tokenizer.eos_token
        texts.append(formatted_prompt)
        
    return {
        "text": texts,
        "reference": references
    }

def format_evaluation_messages(examples, tokenizer):
    """평가용 messages를 tokenizer.apply_chat_template로 포맷팅하는 함수"""
    messages_list = examples["messages"]
    references = examples["reference"]
    questions = examples["question"]
    
    prompts = []
    for messages in messages_list:
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts.append(formatted_prompt)
        
    return {
        "prompt": prompts,
        "reference": references,
        "question": questions
    }

def format_dataset(dataset: DatasetDict, tokenizer) -> DatasetDict:
    """
    messages 구조의 데이터셋에 채팅 템플릿을 적용하는 함수
    - train/validation: 학습용 텍스트 생성
    - test: 평가용 프롬프트와 reference 생성
    """
    print("Starting dataset formatting...")
    
    formatted_dataset = {}
    
    for split_name, split_data in dataset.items():
        print(f"Formatting {split_name} split...")
        
        if split_name in ['train', 'validation']:
            # 학습용 포맷팅
            formatted_split = split_data.map(
                format_training_messages,
                batched=True,
                fn_kwargs={'tokenizer': tokenizer},
                desc=f"Formatting training data for {split_name}"
            )
        else:  # test split
            # 평가용 포맷팅
            formatted_split = split_data.map(
                format_evaluation_messages,
                batched=True,
                fn_kwargs={'tokenizer': tokenizer},
                desc=f"Formatting evaluation data for {split_name}"
            )
        
        formatted_dataset[split_name] = formatted_split
    
    return DatasetDict(formatted_dataset)

def main():
    args = parse_args()
    
    print("=== Task 1c: Dataset Formatting ===")
    
    # 모델 로더를 통해 tokenizer 로드
    if not args.model_id:
        raise ValueError("Model ID is required for loading tokenizer")
    
    print(f"Loading model and tokenizer: {args.model_id}")
    model_loader = ModelLoader(args.model_id)
    
    if not model_loader.tokenizer:
        raise ValueError(f"Failed to load tokenizer for model: {args.model_id}")
    
    tokenizer = model_loader.tokenizer
    
    # 입력 경로 설정 - 파이프라인 환경에서는 이전 task의 output을 input1에서 읽음
    if settings.is_pipeline_env:
        readonly_input_path = settings.pipeline_input_path
        # 읽기 전용 경로를 쓰기 가능한 임시 경로로 복사
        input_path = settings.copy_readonly_to_writable(readonly_input_path, 'format_dataset')
    else:
        input_path = settings.save_dataset_path_preprocessed
    
    # 전처리된 데이터셋 로드
    if not input_path.exists():
        raise FileNotFoundError(f"Preprocessed dataset not found at: {input_path}")
    
    print(f"Loading preprocessed dataset from: {input_path}")
    preprocessed_dataset = load_from_disk(input_path)
    
    # 데이터셋 포맷팅
    formatted_dataset = format_dataset(preprocessed_dataset, tokenizer)
    
    # 출력 경로 설정 (settings에서 중앙 관리)
    output_path = settings.save_dataset_path_formatted
    
    print(f"Saving formatted dataset to: {output_path}")
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # 포맷팅된 데이터셋 저장
    formatted_dataset.save_to_disk(output_path)
    
    print("\n--- Dataset formatting complete ---")
    print("Number of samples in each split:")
    for split_name, split_data in formatted_dataset.items():
        print(f"{split_name}: {len(split_data)}")
    
    print("\nPreview of formatted data:")
    if 'train' in formatted_dataset:
        print("Train split sample (formatted text):")
        print(formatted_dataset['train'][0]['text'][:500] + "...")
    
    if 'test' in formatted_dataset:
        print("\nTest split sample (evaluation format):")
        print(f"Prompt: {formatted_dataset['test'][0]['prompt'][:200]}...")
        print(f"Reference: {formatted_dataset['test'][0]['reference'][:100]}...")
        print(f"Question: {formatted_dataset['test'][0]['question'][:100]}...")
    
    print(f"✅ Formatted dataset saved to: {output_path}")

if __name__ == "__main__":
    main()
