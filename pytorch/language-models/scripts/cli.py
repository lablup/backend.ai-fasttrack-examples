#!/usr/bin/env python3
"""
Gemma-3n Fine-tuning Pipeline CLI
각 태스크를 개별적으로 실행할 수 있는 CLI 인터페이스
"""

import argparse
import sys
import os
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent / 'pipeline-code'
sys.path.insert(0, str(project_root))

from src.data.download_dataset import main as download_dataset_main
from src.data.preprocess_dataset import main as preprocess_dataset_main  
from src.data.format_dataset import main as format_dataset_main
from src.data.dataset import main as dataset_main  # Legacy compatibility
from src.training.trainer import main as trainer_main
from src.evaluation.evaluation import main as evaluation_main

def download_dataset():
    """Task 1a: 데이터셋 다운로드"""
    print("=== Task 1a: Dataset Download ===")
    try:
        sys.argv = ['src/data/download_dataset.py']
        download_dataset_main()
        print("✅ Dataset download task completed successfully")
    except Exception as e:
        print(f"❌ Dataset download task failed: {e}")
        return False
    return True

def preprocess_dataset(config_path=None):
    """Task 1b: 데이터셋 전처리"""
    print("=== Task 1b: Dataset Preprocessing ===")
    try:
        # CLI 인자 설정
        config_file = config_path if config_path else 'messages_format.yaml'
        print(f"Using dataset config: {config_file}")
        sys.argv = [
            'src/data/preprocess_dataset.py',
            '--config', config_file
            ]
        
        preprocess_dataset_main()
        print("✅ Dataset preprocessing task completed successfully")
    except Exception as e:
        print(f"❌ Dataset preprocessing task failed: {e}")
        return False
    return True

def format_dataset():
    """Task 1c: 데이터셋 포맷팅"""
    print("=== Task 1c: Dataset Formatting ===")
    try:
        sys.argv = ['src/data/format_dataset.py']
        format_dataset_main()
        print("✅ Dataset formatting task completed successfully")
    except Exception as e:
        print(f"❌ Dataset formatting task failed: {e}")
        return False
    return True

def evaluate_base_model():
    """Task 2: 베이스 모델 평가"""
    print("=== Task 2: Base Model Evaluation ===")
    try:
        # 베이스 모델 평가를 위한 인자 설정
        sys.argv = [
            'src/evaluation/evaluation.py',
            '--model_name_or_path', os.getenv('MODEL_ID'),
            '--output_path', 'base_model_evaluation.json',
        ]
        evaluation_main()
        print("✅ Base model evaluation completed successfully")
    except Exception as e:
        print(f"❌ Base model evaluation failed: {e}")
        return False
    return True

def fine_tune_model(train_config_path, peft_config_path):
    """Task 3: 모델 파인튜닝"""
    print("=== Task 3: Model Fine-tuning ===")
    try:
        # 파인튜닝을 위한 인자 설정
        sys.argv = [
            'src/training/trainer.py',
            '--train_config_path', train_config_path,
            '--peft_config_path', peft_config_path
        ]
        trainer_main()
        print("✅ Fine-tuning completed successfully")
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        return False
    return True

def evaluate_finetuned_model():
    """Task 4: 파인튜닝된 모델 평가"""
    print("=== Task 4: Fine-tuned Model Evaluation ===")
    try:
        # 파인튜닝된 모델 평가를 위한 인자 설정
        sys.argv = [
            'src/evaluation/evaluation.py',
            '--model_name_or_path', os.getenv('MODEL_ID'),
            '--output_path', 'finetuned_model_evaluation.json',
            '--use_adapter'
        ]
        evaluation_main()
        print("✅ Fine-tuned model evaluation completed successfully")
    except Exception as e:
        print(f"❌ Fine-tuned model evaluation failed: {e}")
        return False
    return True

def run_full_pipeline(train_config_path, peft_config_path, config_path=None):
    """전체 파이프라인 실행"""
    print("🚀 Starting Full Pipeline Execution")
    
    tasks = [
        ("Dataset Download", download_dataset),
        ("Dataset Preprocessing", preprocess_dataset(config_path)), 
        ("Dataset Formatting", format_dataset),
        ("Base Model Evaluation", evaluate_base_model),
        ("Model Fine-tuning", lambda: fine_tune_model(train_config_path, peft_config_path)),
        ("Fine-tuned Model Evaluation", evaluate_finetuned_model)
    ]
    
    for task_name, task_func in tasks:
        print(f"\n📋 Starting: {task_name}")
        success = task_func()
        if not success:
            print(f"💥 Pipeline failed at: {task_name}")
            return False
    
    print("\n🎉 Full pipeline completed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(description="General language model Fine-tuning Pipeline")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 세분화된 데이터셋 태스크들
    subparsers.add_parser('download-dataset', help='Task 1a: Download dataset from HuggingFace')

    preprocess_dataset_parser = subparsers.add_parser('preprocess-dataset', help='Task 1b: Preprocess raw dataset')
    preprocess_dataset_parser.add_argument('--config', type=str, help='Path to dataset preprocessing config file (messages_format.yaml)')
    
    subparsers.add_parser('format-dataset', help='Task 1c: Format dataset with chat template')
    
    # 평가 및 학습 태스크들
    subparsers.add_parser('eval-base', help='Task 2: Evaluate base model')
    
    train_parser = subparsers.add_parser('train', help='Task 3: Fine-tune model')
    train_parser.add_argument('--train_config_path', type=str, required=True,
                             help='Path to training arguments YAML file (train_config.yaml)')
    train_parser.add_argument('--peft_config_path', type=str, required=True,
                             help='Path to PEFT config YAML file (peft_config.yaml)')
    
    subparsers.add_parser('eval-finetuned', help='Task 4: Evaluate fine-tuned model')
    
    # 전체 파이프라인 실행
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
    pipeline_parser.add_argument('--train_config_path', type=str, required=True,
                                help='Path to training arguments YAML file (train_config.yaml)')
    pipeline_parser.add_argument('--peft_config_path', type=str, required=True,
                                help='Path to PEFT config YAML file (peft_config.yaml)')
    pipeline_parser.add_argument('--config', type=str, help='Path to dataset preprocessing config file (messages_format.yaml)')

    
    args = parser.parse_args()
    
    if args.command == 'download-dataset':
        download_dataset()
    elif args.command == 'preprocess-dataset':
        preprocess_dataset(config_path=args.config)
    elif args.command == 'format-dataset':
        format_dataset()
    elif args.command == 'eval-base':
        evaluate_base_model()
    elif args.command == 'train':
        fine_tune_model(args.train_config_path, args.peft_config_path)
    elif args.command == 'eval-finetuned':
        evaluate_finetuned_model()
    elif args.command == 'pipeline':
        run_full_pipeline(args.train_config_path, args.peft_config_path, config_path=args.config)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()