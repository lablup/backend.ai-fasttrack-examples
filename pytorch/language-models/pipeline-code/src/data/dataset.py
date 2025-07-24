#!/usr/bin/env python3
"""
Dataset Pipeline - Legacy Compatibility Mode
기존 전체 데이터셋 파이프라인을 하나의 파일에서 실행 (하위 호환성 유지)
새로운 세분화된 태스크는 download_dataset.py, preprocess_dataset.py, format_dataset.py 사용
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path
from configs.settings import settings

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Pipeline - Legacy Mode")
    parser.add_argument('--model-id', type=str, default=os.getenv('MODEL_ID'),
                        help='ID of the model to load from Hugging Face Hub')
    parser.add_argument('--dataset-name', type=str, default=os.getenv('DATASET'),
                        help='Name of the dataset to load from Hugging Face Hub')
    parser.add_argument('--trust_remote_code', action='store_true', default=False,
                        help='Trust remote code for datasets that require it.')
    parser.add_argument('--HF_TOKEN', type=str, default=os.getenv('HF_TOKEN'),
                        help='Hugging Face API token for authentication.')
    parser.add_argument('--use_subtasks', action='store_true', default=True,
                        help='Use separated subtasks (recommended). Set to False for legacy behavior.')
    return parser.parse_args()

def run_subtask(script_name, args_list):
    """개별 서브태스크를 실행하는 함수"""
    script_path = Path(__file__).parent / script_name
    cmd = [sys.executable, str(script_path)] + args_list
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running {script_name}:")
        print(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)
    else:
        print(result.stdout)

# Legacy functions and classes kept for reference (deprecated)
# Use download_dataset.py, preprocess_dataset.py, format_dataset.py instead

def main():
    args = parse_args()
    
    if args.use_subtasks:
        print("=== Dataset Pipeline - Using Separated Subtasks ===")
        
        # Task 1a: Dataset Download
        download_args = [
            '--dataset-name', args.dataset_name or '',
            '--HF_TOKEN', args.HF_TOKEN or ''
        ]
        if args.trust_remote_code:
            download_args.append('--trust_remote_code')
        
        print("\n--- Running Task 1a: Dataset Download ---")
        run_subtask('download_dataset.py', download_args)
        
        # Task 1b: Dataset Preprocessing  
        print("\n--- Running Task 1b: Dataset Preprocessing ---")
        run_subtask('preprocess_dataset.py', [])
        
        # Task 1c: Dataset Formatting
        format_args = [
            '--model-id', args.model_id or '',
            '--HF_TOKEN', args.HF_TOKEN or ''
        ]
        
        print("\n--- Running Task 1c: Dataset Formatting ---")
        run_subtask('format_dataset.py', format_args)
        
        print("\n✅ All dataset subtasks completed successfully!")
        
    else:
        # Legacy behavior - import and run original code
        print("=== Dataset Pipeline - Legacy Mode ===")
        print("Warning: Legacy mode is deprecated. Consider using --use_subtasks flag.")
        
        # 여기에 기존 코드를 유지할 수 있지만, 새로운 서브태스크 방식을 권장
        raise NotImplementedError("Legacy mode has been deprecated. Use separated subtasks instead.")

if __name__ == "__main__":
    main()