#!/usr/bin/env python3
"""
VLM Fine-tuning Pipeline CLI
각 태스크를 개별적으로 실행할 수 있는 CLI 인터페이스
VLM 모델용으로 수정됨
"""

import os
import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent / 'pipeline-code'
sys.path.insert(0, str(project_root))

from src.data.download_dataset import main as download_dataset_main
from src.training.vlm_trainer import main as vlm_trainer_main
from src.evaluation.evaluation import main as evaluation_main

def download_dataset():
    """Task 1: VQA 데이터셋 다운로드"""
    print("=== Task 1: VQA Dataset Download ===")
    try:
        # 환경 변수에서 인자 대신 직접 호출
        original_sys_argv = sys.argv.copy()
        sys.argv = ['src/data/download_dataset.py']  # 스크립트명만 유지
        download_dataset_main()
        sys.argv = original_sys_argv
        print("✅ Dataset download completed successfully")
    except Exception as e:
        print(f"❌ Dataset download failed: {e}")
        sys.argv = original_sys_argv
    return True

def evaluate_base_model():
    """Task 2: 베이스 VLM 모델 평가"""
    print("=== Task 2: Base VLM Model Evaluation ===")
    try:
        original_sys_argv = sys.argv.copy()
        sys.argv = [
            'src/evaluation/evaluation.py',
            '--model_name_or_path', os.getenv('MODEL_ID'),
            '--output_path', 'base_model_evaluation.json',
            ]
        evaluation_main()
        sys.argv = original_sys_argv
        print("✅ Base model evaluation completed successfully")
    except Exception as e:
        print(f"❌ Base model evaluation failed: {e}")
        sys.argv = original_sys_argv
    return True

def fine_tune_vlm_model(train_config_path, peft_config_path, vlm_model_config, vlm_collator_config):
    """Task 3: VLM 모델 파인튜닝"""
    print("=== Task 3: VLM Model Fine-tuning ===")
    try:
        original_sys_argv = sys.argv.copy()
        sys.argv = [
            'src/training/vlm_trainer.py',
            '--train_config_path', train_config_path,
            '--peft_config_path', peft_config_path,
            '--vlm_model_config', vlm_model_config,
            '--vlm_collator_config', vlm_collator_config
        ]
        vlm_trainer_main()
        sys.argv = original_sys_argv
        print("✅ VLM fine-tuning completed successfully")
    except Exception as e:
        print(f"❌ VLM fine-tuning failed: {e}")
        sys.argv = original_sys_argv
    return True

def evaluate_finetuned_vlm_model():
    """Task 4: 파인튜닝된 VLM 모델 평가"""
    print("=== Task 4: Fine-tuned VLM Model Evaluation ===")
    try:
        original_sys_argv = sys.argv.copy()
        sys.argv = [
            'src/evaluation/evaluation.py',
            '--model_name_or_path', os.getenv('MODEL_ID'),
            '--output_path', 'finetuned_model_evaluation.json',
            '--use_adapter',
            ]
        evaluation_main()
        sys.argv = original_sys_argv
        print("✅ Fine-tuned model evaluation completed successfully")
    except Exception as e:
        print(f"❌ Fine-tuned model evaluation failed: {e}")
        sys.argv = original_sys_argv
    return True

def run_full_vlm_pipeline(train_config_path, peft_config_path, vlm_model_config='vlm_model_config.yaml', vlm_collator_config='vlm_collator_config.yaml'):
    """전체 VLM 파이프라인 실행"""
    print("🚀 Starting Full VLM Pipeline Execution")
    
    tasks = [
        ("VQA Dataset Download", download_dataset),
        ("Base VLM Model Evaluation", evaluate_base_model),
        ("VLM Model Fine-tuning", lambda: fine_tune_vlm_model(train_config_path, peft_config_path, vlm_model_config, vlm_collator_config)),
        ("Fine-tuned VLM Model Evaluation", evaluate_finetuned_vlm_model)
    ]
    
    for task_name, task_func in tasks:
        print(f"\n🔄 Starting {task_name}...")
        success = task_func()
        if not success:
            print(f"❌ {task_name} failed. Stopping pipeline.")
            return False
        print(f"✅ {task_name} completed successfully")
    
    print("\n🎉 Full VLM pipeline completed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(description="VLM Fine-tuning Pipeline")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 개별 태스크들
    subparsers.add_parser('download-dataset', help='Task 1: Download VQA dataset from HuggingFace')
    subparsers.add_parser('eval-base', help='Task 2: Evaluate base VLM model')
    
    train_parser = subparsers.add_parser('train', help='Task 3: Fine-tune VLM model')
    train_parser.add_argument('--train_config_path', type=str, required=True,
                             help='Path to training arguments YAML file (train_config.yaml)')
    train_parser.add_argument('--peft_config_path', type=str, required=True,
                             help='Path to PEFT config YAML file (peft_config.yaml)')
    train_parser.add_argument('--vlm_model_config', type=str, required=True,
                             help='Path to VLM model config YAML file')
    train_parser.add_argument('--vlm_collator_config', type=str, required=True,
                             help='Path to VLM collator config YAML file')
    
    subparsers.add_parser('eval-finetuned', help='Task 4: Evaluate fine-tuned VLM model')
    
    # 전체 파이프라인 실행
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full VLM pipeline')
    pipeline_parser.add_argument('--train_config_path', type=str, required=True,
                                help='Path to training arguments YAML file (train_config.yaml)')
    pipeline_parser.add_argument('--peft_config_path', type=str, required=True,
                                help='Path to PEFT config YAML file (peft_config.yaml)')
    pipeline_parser.add_argument('--vlm_model_config', type=str, required=True,
                                help='Path to VLM model config YAML file')
    pipeline_parser.add_argument('--vlm_collator_config', type=str, required=True,
                                help='Path to VLM collator config YAML file')
    
    args = parser.parse_args()
    
    if args.command == 'download-dataset':
        download_dataset()
    elif args.command == 'eval-base':
        evaluate_base_model()
    elif args.command == 'train':
        fine_tune_vlm_model(args.train_config_path, args.peft_config_path, args.vlm_model_config, args.vlm_collator_config)
    elif args.command == 'eval-finetuned':
        evaluate_finetuned_vlm_model()
    elif args.command == 'pipeline':
        run_full_vlm_pipeline(args.train_config_path, args.peft_config_path, args.vlm_model_config, args.vlm_collator_config)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
