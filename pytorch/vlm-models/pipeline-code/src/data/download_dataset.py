#!/usr/bin/env python3
"""
Task 1a: Dataset Download
HuggingFace에서 데이터셋을 다운로드하고 train/validation/test 스플릿을 보장하는 태스크
"""

import os
import argparse
from datasets import load_dataset, get_dataset_split_names, DatasetDict, concatenate_datasets
from pathlib import Path
from configs.settings import settings

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Downloader")
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

    # 최소 하나의 스플릿은 있어야 함
    if not available_splits:
        raise ValueError(f"Dataset '{dataset_name}' has no available splits.")

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
    
    # Case 5: validation 스플릿만 존재하는 경우 -> 80/10/10으로 분할
    elif not has_train and has_val and not has_test:
        print("Only 'validation' split found. Creating 'train' (80%), 'validation' (10%) and 'test' (10%) splits.")
        # 1. validation -> train (80%) / temp (20%)
        train_temp_split = dataset['validation'].train_test_split(test_size=0.2, shuffle=True, seed=42)
        # 2. temp (20%) -> validation (10%) / test (10%)
        val_test_split = train_temp_split['test'].train_test_split(test_size=0.5, shuffle=True, seed=42)
        
        return DatasetDict({
            'train': train_temp_split['train'],
            'validation': val_test_split['train'],
            'test': val_test_split['test']
        })
    
    # Case 6: test 스플릿만 존재하는 경우 -> 80/10/10으로 분할
    elif not has_train and not has_val and has_test:
        print("Only 'test' split found. Creating 'train' (80%), 'validation' (10%) and 'test' (10%) splits.")
        # 1. test -> train (80%) / temp (20%)
        train_temp_split = dataset['test'].train_test_split(test_size=0.2, shuffle=True, seed=42)
        # 2. temp (20%) -> validation (10%) / test (10%)
        val_test_split = train_temp_split['test'].train_test_split(test_size=0.5, shuffle=True, seed=42)
        
        return DatasetDict({
            'train': train_temp_split['train'],
            'validation': val_test_split['train'],
            'test': val_test_split['test']
        })
    
    # Case 7: validation과 test 스플릿만 존재하는 경우 -> 전체를 8:1:1로 재구성
    elif not has_train and has_val and has_test:
        print("Found 'validation' and 'test'. Combining and redistributing to achieve 8:1:1 ratio.")
        
        # validation과 test를 합쳐서 전체 데이터로 만들기
        combined_dataset = concatenate_datasets([dataset['validation'], dataset['test']])
        
        # 전체를 80/20으로 먼저 분할 (train 80% / temp 20%)
        train_temp_split = combined_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
        # temp 20%를 다시 50/50으로 분할 (validation 10% / test 10%)
        val_test_split = train_temp_split['test'].train_test_split(test_size=0.5, shuffle=True, seed=42)
        
        return DatasetDict({
            'train': train_temp_split['train'],
            'validation': val_test_split['train'],
            'test': val_test_split['test']
        })
    
    # Case 8: 알 수 없는 다른 스플릿들이 있는 경우 - 첫 번째 스플릿을 사용
    else:
        available_split_names = list(dataset.keys())
        if available_split_names:
            first_split = available_split_names[0]
            print(f"Unknown split configuration. Using '{first_split}' split to create train/validation/test splits.")
            # 첫 번째 스플릿을 80/10/10으로 분할
            train_temp_split = dataset[first_split].train_test_split(test_size=0.2, shuffle=True, seed=42)
            val_test_split = train_temp_split['test'].train_test_split(test_size=0.5, shuffle=True, seed=42)
            
            return DatasetDict({
                'train': train_temp_split['train'],
                'validation': val_test_split['train'],
                'test': val_test_split['test']
            })
        else:
            raise ValueError(f"No valid splits found in dataset '{dataset_name}'")
        
    return dataset

def main():
    args = parse_args()
    dataset_name = args.dataset_name

    if not dataset_name:
        print("Dataset name is not provided. Please set the DATASET environment variable.")
        return

    print("=== Task 1a: Dataset Download & Split ===")
    
    # 데이터셋 다운로드 및 스플릿 생성
    raw_dataset = load_and_split_dataset(dataset_name, args.HF_TOKEN, args.trust_remote_code)
    
    # 저장 경로 설정
    output_path = settings.save_dataset_path_raw
    
    print(f"Saving raw dataset to: {output_path}")
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # 원본 데이터셋 저장
    raw_dataset.save_to_disk(output_path)
    
    print("\n--- Dataset download complete ---")
    print("Number of samples in each split:")
    for split_name, split_data in raw_dataset.items():
        print(f"{split_name}: {len(split_data)}")
    
    print(f"✅ Raw dataset saved to: {output_path}")

if __name__ == "__main__":
    main()
