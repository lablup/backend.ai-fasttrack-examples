#!/usr/bin/env python3
"""
Task 1b: Dataset Preprocessing
원본 데이터셋을 불러와서 messages 구조로 변환하는 전처리 태스크
- train/validation: 학습용 messages 구조 생성
- test: 평가용 messages 구조 생성

일반화된 데이터셋 전처리로 사용자가 configs/messages_format.yaml을 통해
다양한 데이터셋에 적용할 수 있습니다.
"""

import os
import argparse
import yaml
from datasets import load_from_disk, DatasetDict
from pathlib import Path
from configs.settings import settings

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Preprocessor")
    parser.add_argument('--max_samples_per_split', type=int, default=None,
                        help='Maximum number of samples per split for testing (optional)')
    parser.add_argument('--config', type=str, default='messages_format.yaml',
                        help='Path to dataset configuration YAML file')
    return parser.parse_args()

def load_dataset_config(config_path: str) -> dict:
    """
    데이터셋 설정 YAML 파일을 로드합니다.
    """
    config_file = settings.config_path / config_path
    print(f"Loading dataset config from: {config_file}")
    if not config_file.exists():
        raise FileNotFoundError(f"Dataset config file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 필수 설정 확인
    required_keys = ['training_columns', 'messages_format', 'evaluate_columns']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    return config

def create_training_messages(examples, training_columns: dict, messages_format: dict):
    """
    학습용 messages 구조를 생성합니다.
    
    Args:
        examples: HuggingFace dataset examples (batched format)
        training_columns: 컬럼 매핑 정보 (key: 템플릿_변수명, value: 실제_데이터셋_컬럼명)
        messages_format: messages 포맷 템플릿
    
    Returns:
        dict: {"messages": List[List[dict]], "reference": List[str]}
    """
    messages_list = []
    references_list = []
    
    # 첫 번째 컬럼으로 데이터 개수 확인 (모든 컬럼은 같은 길이를 가짐)
    first_column_name = list(training_columns.values())[0]
    batch_size = len(examples[first_column_name])
    
    # 배치의 각 샘플에 대해 처리
    for idx in range(batch_size):
        # 템플릿 변수에 실제 데이터 값 매핑
        template_variables = {}
        for template_var, dataset_column in training_columns.items():
            template_variables[template_var] = examples[dataset_column][idx]
        
        # system_prompt 추가
        template_variables['system_prompt'] = messages_format['system_prompt']
        
        # messages 구조 생성 (role과 content로 구성)
        messages = []
        for message_template in messages_format['messages']:
            content = message_template['content'].format(**template_variables)
            messages.append({
                "role": message_template['role'],
                "content": content
            })
        
        messages_list.append(messages)
        # response를 reference로 저장 (평가용)
        references_list.append(template_variables.get('response', ''))
    
    return {
        "messages": messages_list,
        "reference": references_list
    }

def create_evaluation_messages(examples, evaluate_columns: dict):
    """
    평가용 messages 구조를 생성합니다.
    
    Args:
        examples: HuggingFace dataset examples (batched format)
        evaluate_columns: 평가용 컬럼 매핑 정보 (query, response 키 필수)
    
    Returns:
        dict: {"messages": List[List[dict]], "reference": List[str], "question": List[str]}
    """
    messages_list = []
    references_list = []
    questions_list = []
    
    query_column_name = evaluate_columns['query']
    response_column_name = evaluate_columns['response']
    
    # 배치의 각 샘플에 대해 처리
    for query, response in zip(examples[query_column_name], examples[response_column_name]):
        messages = [{"role": "user", "content": query}]
        messages_list.append(messages)
        references_list.append(response)
        questions_list.append(query)
    
    return {
        "messages": messages_list,
        "reference": references_list,
        "question": questions_list
    }

def preprocess_dataset(dataset: DatasetDict, config: dict, max_samples_per_split=None) -> DatasetDict:
    """
    데이터셋을 messages 구조로 변환하는 전처리 함수
    - train/validation: 학습용 messages (system, user, assistant)
    - test: 평가용 messages (user만 포함, reference 별도 저장)
    
    Args:
        dataset: 원본 데이터셋
        config: 데이터셋 설정 정보
        max_samples_per_split: 최대 샘플 수 제한
    """
    print("Starting dataset preprocessing...")
    
    training_columns = config['training_columns']
    messages_format = config['messages_format']
    evaluate_columns = config['evaluate_columns']
    data_filtering = config.get('data_filtering', {})
    
    # 필수 컬럼 확인
    required_columns = data_filtering.get('required_columns', [])
    if required_columns:
        for split_name, split_data in dataset.items():
            missing_columns = [col for col in required_columns if col not in split_data.column_names]
            if missing_columns:
                raise ValueError(f"Missing required columns in {split_name} split: {missing_columns}")
    
    preprocessed_dataset = {}
    min_text_length = data_filtering.get('min_text_length', 1)
    
    for split_name, split_data in dataset.items():
        print(f"Preprocessing {split_name} split...")
        
        # 데이터 품질 필터링
        if required_columns:
            def filter_data(example):
                for col in required_columns:
                    if (example[col] is None or 
                        len(str(example[col]).strip()) < min_text_length):
                        return False
                return True
            
            filtered_data = split_data.filter(
                filter_data,
                desc=f"Filtering {split_name} data"
            )
        else:
            filtered_data = split_data
        
        # 샘플 수 제한 (테스트용)
        if max_samples_per_split and len(filtered_data) > max_samples_per_split:
            print(f"Limiting {split_name} split to {max_samples_per_split} samples")
            filtered_data = filtered_data.select(range(max_samples_per_split))
        
        # messages 구조로 변환
        if split_name in ['train', 'validation']:
            # 학습용: system, user, assistant messages
            processed_data = filtered_data.map(
                create_training_messages,
                batched=True,
                fn_kwargs={
                    'training_columns': training_columns,
                    'messages_format': messages_format
                },
                desc=f"Creating training messages for {split_name}"
            )
            
        else:  # test split
            # 평가용: user message만 포함, reference는 별도 저장
            processed_data = filtered_data.map(
                create_evaluation_messages,
                batched=True,
                fn_kwargs={
                    'evaluate_columns': evaluate_columns
                },
                desc=f"Creating evaluation messages for {split_name}"
            )
        
        preprocessed_dataset[split_name] = processed_data
        print(f"{split_name}: {len(dataset[split_name])} -> {len(processed_data)} samples")
    
    return DatasetDict(preprocessed_dataset)

def main():
    args = parse_args()
    
    print("=== Task 1b: Dataset Preprocessing ===")
    
    # 데이터셋 설정 파일 로드
    print(f"Loading dataset config from: {args.config}")
    config = load_dataset_config(args.config)
    
    # 입력 경로 설정 - 파이프라인 환경에서는 이전 task의 output을 input1에서 읽음
    if settings.is_pipeline_env:
        readonly_input_path = settings.pipeline_input_path
        # 읽기 전용 경로를 쓰기 가능한 임시 경로로 복사
        input_path = settings.copy_readonly_to_writable(readonly_input_path, 'preprocess_dataset')
    else:
        input_path = settings.save_dataset_path_raw
    
    # 원본 데이터셋 로드
    if not input_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {input_path}")
    
    print(f"Loading raw dataset from: {input_path}")
    raw_dataset = load_from_disk(input_path)
    
    # 데이터 전처리
    preprocessed_dataset = preprocess_dataset(
        raw_dataset, 
        config,
        max_samples_per_split=args.max_samples_per_split
    )
    
    # 출력 경로 설정
    output_path = settings.save_dataset_path_preprocessed
    
    print(f"Saving preprocessed dataset to: {output_path}")
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # 전처리된 데이터셋 저장
    preprocessed_dataset.save_to_disk(output_path)
    
    print("\n--- Dataset preprocessing complete ---")
    print("Number of samples in each split after preprocessing:")
    for split_name, split_data in preprocessed_dataset.items():
        print(f"{split_name}: {len(split_data)}")
    
    print("\nSample of preprocessed data:")
    print("Train split sample (messages structure):")
    if 'train' in preprocessed_dataset:
        print(f"Messages: {preprocessed_dataset['train'][0]['messages']}")
        print(f"Reference: {preprocessed_dataset['train'][0]['reference']}")
    
    print("\nTest split sample (evaluation structure):")
    if 'test' in preprocessed_dataset:
        print(f"Messages: {preprocessed_dataset['test'][0]['messages']}")
        print(f"Reference: {preprocessed_dataset['test'][0]['reference']}")
        print(f"Question: {preprocessed_dataset['test'][0]['question']}")
    
    print(f"\n✅ Preprocessed dataset saved to: {output_path}")
    print(f"📄 Configuration used: {args.config}")

if __name__ == "__main__":
    main()
