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

from src.data.dataset import main as dataset_main
from src.training.trainer import main as trainer_main
from src.evaluation.evaluation import main as evaluation_main

def setup_dataset():
    """Task 1: 데이터셋 다운로드 및 전처리"""
    print("=== Task 1: Dataset Download & Preprocessing ===")
    try:
        sys.argv = ['src/data/dataset.py',
                    ] 

        dataset_main()
        print("✅ Dataset task completed successfully")
    except Exception as e:
        print(f"❌ Dataset task failed: {e}")
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

def fine_tune_model(training_args_path, peft_config_path):
    """Task 3: 모델 파인튜닝"""
    print("=== Task 3: Model Fine-tuning ===")
    try:
        # 파인튜닝을 위한 인자 설정
        sys.argv = [
            'src/training/trainer.py',
            '--training_args_path', training_args_path,
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

def run_full_pipeline(training_args_path, peft_config_path):
    """전체 파이프라인 실행"""
    print("🚀 Starting Full Pipeline Execution")
    
    tasks = [
        ("Dataset Setup", setup_dataset),
        ("Base Model Evaluation", evaluate_base_model),
        ("Model Fine-tuning", lambda: fine_tune_model(training_args_path, peft_config_path)),
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

    parser = argparse.ArgumentParser(description="Gemma-3n Fine-tuning Pipeline")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 개별 태스크 실행
    subparsers.add_parser('dataset', help='Download and preprocess dataset')
    subparsers.add_parser('eval-base', help='Evaluate base model')
    
    train_parser = subparsers.add_parser('train', help='Fine-tune model')
    train_parser.add_argument('--training_args_path', type=str, required=True,
                             help='Path to training arguments YAML file')
    train_parser.add_argument('--peft_config_path', type=str, required=True,
                             help='Path to PEFT config YAML file')
    
    subparsers.add_parser('eval-finetuned', help='Evaluate fine-tuned model')
    
    # 전체 파이프라인 실행
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
    pipeline_parser.add_argument('--training_args_path', type=str, required=True,
                                help='Path to training arguments YAML file')
    pipeline_parser.add_argument('--peft_config_path', type=str, required=True,
                                help='Path to PEFT config YAML file')
    
    args = parser.parse_args()
    
    if args.command == 'dataset':
        setup_dataset()
    elif args.command == 'eval-base':
        evaluate_base_model()
    elif args.command == 'train':
        fine_tune_model(args.training_args_path, args.peft_config_path)
    elif args.command == 'eval-finetuned':
        evaluate_finetuned_model()
    elif args.command == 'pipeline':
        run_full_pipeline(args.training_args_path, args.peft_config_path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()