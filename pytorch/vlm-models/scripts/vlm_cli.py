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
from configs.settings import settings

def download_dataset():
    """Task 1: 데이터셋 다운로드"""
    print("=== Task 1: Dataset Download ===")
    original_sys_argv = sys.argv.copy()
    try:
        # 환경 변수에서 인자 대신 직접 호출
        sys.argv = ['src/data/download_dataset.py']  # 스크립트명만 유지
        download_dataset_main()
        print("✅ Dataset download completed successfully")
        return True
    except Exception as e:
        print(f"❌ Dataset download failed: {e}")
        return False
    finally:
        sys.argv = original_sys_argv

def evaluate_base_model(vlm_model_config, vlm_collator_config):
    """Task 2: 베이스 VLM 모델 평가"""
    print("=== Task 2: Base VLM Model Evaluation ===")
    original_sys_argv = sys.argv.copy()
    try:
        sys.argv = [
            'src/evaluation/evaluation.py',
            '--model_name_or_path', os.getenv('MODEL_ID'),
            '--output_path', 'base_model_evaluation.json',
            '--vlm_model_config', vlm_model_config,
            '--vlm_collator_config', vlm_collator_config,
            ]
        evaluation_main()
        print("✅ Base model evaluation completed successfully")
        return True
    except Exception as e:
        print(f"❌ Base model evaluation failed: {e}")
        return False
    finally:
        sys.argv = original_sys_argv

def fine_tune_vlm_model(train_config_path, peft_config_path, vlm_model_config, vlm_collator_config):
    """Task 3: VLM 모델 파인튜닝"""
    print("=== Task 3: VLM Model Fine-tuning ===")
    original_sys_argv = sys.argv.copy()
    try:
        sys.argv = [
            'src/training/vlm_trainer.py',
            '--train_config_path', train_config_path,
            '--peft_config_path', peft_config_path,
            '--vlm_model_config', vlm_model_config,
            '--vlm_collator_config', vlm_collator_config
        ]
        vlm_trainer_main()
        print("✅ VLM fine-tuning completed successfully")
        return True
    except Exception as e:
        print(f"❌ VLM fine-tuning failed: {e}")
        return False
    finally:
        sys.argv = original_sys_argv

def evaluate_finetuned_vlm_model(vlm_model_config, vlm_collator_config):
    """Task 4: 파인튜닝된 VLM 모델 평가"""
    print("=== Task 4: Fine-tuned VLM Model Evaluation ===")
    original_sys_argv = sys.argv.copy()
    try:
        sys.argv = [
            'src/evaluation/evaluation.py',
            '--model_name_or_path', os.getenv('MODEL_ID'),
            '--output_path', 'finetuned_model_evaluation.json',
            '--vlm_model_config', vlm_model_config,
            '--vlm_collator_config', vlm_collator_config,
            '--use_adapter',
            ]
        evaluation_main()
        print("✅ Fine-tuned model evaluation completed successfully")
        return True
    except Exception as e:
        print(f"❌ Fine-tuned model evaluation failed: {e}")
        return False
    finally:
        sys.argv = original_sys_argv

def run_full_vlm_pipeline(train_config_path, peft_config_path, vlm_model_config, vlm_collator_config):
    """전체 VLM 파이프라인 실행"""
    print("🚀 Starting Full VLM Pipeline Execution")
    
    tasks = [
        ("Dataset Download", download_dataset),
        ("Base VLM Model Evaluation", lambda: evaluate_base_model(vlm_model_config, vlm_collator_config)),
        ("VLM Model Fine-tuning", lambda: fine_tune_vlm_model(train_config_path, peft_config_path, vlm_model_config, vlm_collator_config)),
        ("Fine-tuned VLM Model Evaluation", lambda: evaluate_finetuned_vlm_model(vlm_model_config, vlm_collator_config))
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

def add_vlm_config_args(parser, required):
    """VLM 모델/콜레이터 설정 파일 인자를 서브파서에 추가합니다."""
    parser.add_argument('--vlm_model_config', type=str, required=required,
                        default=None if required else settings.vlm_model_config,
                        help='Path to VLM model config YAML file')
    parser.add_argument('--vlm_collator_config', type=str, required=required,
                        default=None if required else settings.vlm_collator_config,
                        help='Path to VLM collator config YAML file')

def main():
    parser = argparse.ArgumentParser(description="VLM Fine-tuning Pipeline")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 개별 태스크들
    subparsers.add_parser('download-dataset', help='Task 1: Download dataset from HuggingFace')

    eval_base_parser = subparsers.add_parser('eval-base', help='Task 2: Evaluate base VLM model')
    add_vlm_config_args(eval_base_parser, required=False)
    
    train_parser = subparsers.add_parser('train', help='Task 3: Fine-tune VLM model')
    train_parser.add_argument('--train_config_path', type=str, required=True,
                             help='Path to training arguments YAML file (train_config.yaml)')
    train_parser.add_argument('--peft_config_path', type=str, required=True,
                             help='Path to PEFT config YAML file (peft_config.yaml)')
    add_vlm_config_args(train_parser, required=True)
    
    eval_finetuned_parser = subparsers.add_parser('eval-finetuned', help='Task 4: Evaluate fine-tuned VLM model')
    add_vlm_config_args(eval_finetuned_parser, required=False)
    
    # 전체 파이프라인 실행
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full VLM pipeline')
    pipeline_parser.add_argument('--train_config_path', type=str, required=True,
                                help='Path to training arguments YAML file (train_config.yaml)')
    pipeline_parser.add_argument('--peft_config_path', type=str, required=True,
                                help='Path to PEFT config YAML file (peft_config.yaml)')
    add_vlm_config_args(pipeline_parser, required=True)
    
    args = parser.parse_args()
    
    if args.command == 'download-dataset':
        success = download_dataset()
    elif args.command == 'eval-base':
        success = evaluate_base_model(args.vlm_model_config, args.vlm_collator_config)
    elif args.command == 'train':
        success = fine_tune_vlm_model(args.train_config_path, args.peft_config_path, args.vlm_model_config, args.vlm_collator_config)
    elif args.command == 'eval-finetuned':
        success = evaluate_finetuned_vlm_model(args.vlm_model_config, args.vlm_collator_config)
    elif args.command == 'pipeline':
        success = run_full_vlm_pipeline(args.train_config_path, args.peft_config_path, args.vlm_model_config, args.vlm_collator_config)
    else:
        parser.print_help()
        success = False

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
