#!/usr/bin/env python3
"""
VLM Fine-tuning Trainer
VLM 모델을 위한 파인튜닝 트레이너
사용자 정의 데이터 콜레이터를 사용하여 이미지와 텍스트를 함께 처리
"""

import os
import json
import argparse
import torch
import sys
import yaml
import wandb
from pathlib import Path
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig

from src.models.model import ModelLoader
from src.data.collate_fn import create_vlm_collator
from configs.settings import settings

def parse_args():
    parser = argparse.ArgumentParser(description="VLM Trainer Configuration")
    parser.add_argument('--model_id', type=str, default=os.getenv('MODEL_ID'),
                        help='ID of the VLM model to load from Hugging Face Hub')
    parser.add_argument('--dataset-name', type=str, default=os.getenv('DATASET'),
                        help='Name of the dataset to load from Hugging Face Hub')
    parser.add_argument('--train_config_path', type=str, required=True,
                        help='Path to YAML file for SFTConfig (e.g., train_config.yaml)')
    parser.add_argument('--peft_config_path', type=str, required=True,
                        help='Path to YAML file for PEFT config')
    parser.add_argument('--vlm_model_config', type=str, required=True,
                        help='Path to VLM model configuration file')
    parser.add_argument('--vlm_collator_config', type=str, required=True,
                        help='Path to VLM collator configuration file')
    parser.add_argument('--wandb_token', type=str, default=os.getenv('WANDB_API_KEY'),
                        help='Weights & Biases API token for logging.')
    parser.add_argument('--wandb_project', type=str, default="vlm-finetuning",
                        help='Weights & Biases project name.')
    return parser.parse_args()

class VLMTrainer:
    """VLM 모델을 위한 커스텀 트레이너 클래스"""
    
    def __init__(self, model_loader, output_dir, dataset_path, vlm_collator_config='vlm_collator_config.yaml'):
        self.model_loader = model_loader
        self.model = self.model_loader.model
        self.processor = self.model_loader.processor
        self.tokenizer = self.model_loader.tokenizer
        self.output_dir = output_dir
        self.vlm_collator_config = vlm_collator_config
        
        # VLM 데이터 콜레이터 생성
        if self.processor:
            print(f"Creating VLM data collator with config: {vlm_collator_config}")
            self.data_collator = create_vlm_collator(
                processor=self.processor,
                config_path=vlm_collator_config
            )
            print("✅ VLM data collator created successfully")
        else:
            print("❌ Cannot create VLM data collator: processor not available")
            self.data_collator = None
        
        # 데이터셋 로드
        try:
            print(f"Loading dataset from: {dataset_path}")
            self.dataset = load_from_disk(dataset_path)
            print(f"✅ Dataset loaded successfully with splits: {list(self.dataset.keys())}")
        except FileNotFoundError:
            print(f"❌ Dataset not found at: {dataset_path}")
            self.dataset = None

    def train(self, train_config_dict=None, peft_config_dict=None, logging_dir=None):
        """VLM 모델 파인튜닝 실행"""
        if not self.model or not self.processor or not self.data_collator:
            print("❌ Model, processor, or data collator is not loaded. Cannot proceed with training.")
            return
        
        if not self.dataset:
            print("❌ Dataset is not loaded. Cannot proceed with training.")
            return
        
        # 학습 설정 (self. 직접 사용으로 일관성 유지)
        if train_config_dict:
            print("Using custom VLM training arguments from YAML file...")
            train_config = SFTConfig(
                output_dir=self.output_dir,
                logging_dir=logging_dir,
                **train_config_dict
            )
        else:
            print("Using default VLM training arguments...")
            train_config = SFTConfig(
                output_dir=self.output_dir,
                logging_dir=logging_dir,
                per_device_train_batch_size=4,  # VLM은 메모리 사용량이 크므로 작게 설정
                per_device_eval_batch_size=4,
                gradient_accumulation_steps=4,  # 배치 크기 보상
                gradient_checkpointing=True,
                learning_rate=1e-5,
                num_train_epochs=1,
                logging_steps=0.1,
                eval_steps=0.1,
                save_steps=0.1,
                bf16=True,
                remove_unused_columns=False,  # VLM에서는 이미지 데이터 보존 필요
                dataloader_pin_memory=False,
            )
        
        # PEFT 설정
        if peft_config_dict:
            print("Using custom PEFT configuration...")
            peft_config = LoraConfig(**peft_config_dict)
        else:
            print("Using default PEFT configuration for VLM...")
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                r=32,  # VLM은 더 복잡하므로 r 값을 조금 높게
                target_modules="all-linear",
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none"
            )
        
        # SFT 트레이너 생성 (VLM processor 사용)
        # 중요: Qwen2VLProcessor는 직접 eos_token 속성이 없으므로 tokenizer를 직접 전달
        trainer = SFTTrainer(
            model=self.model,  # self. 직접 사용
            args=train_config,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset.get('validation'),
            tokenizer=self.tokenizer,  # tokenizer를 직접 전달 (processor 대신)
            peft_config=peft_config,
            data_collator=self.data_collator,  # VLM 전용 데이터 콜레이터 사용
        )
        
        print("🚀 Starting VLM fine-tuning...")
        train_result = trainer.train()
        trainer_stats = train_result.metrics
        
        # 학습 통계 저장
        if logging_dir:
            log_file_path = logging_dir / "training_stats.json"
            print(f"Saving training stats to {log_file_path}")
            with open(log_file_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(trainer_stats, f, ensure_ascii=False, indent=4)
        
        print("💾 Saving VLM PEFT adapter...")
        trainer.save_model()  # PEFT 어댑터만 저장
        
        # 병합된 배포용 모델 저장을 위한 경로 설정 (settings에서 관리)
        deployment_model_path = settings.deployment_model_path
        deployment_model_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🔄 Merging LoRA weights and saving deployment-ready VLM model to {deployment_model_path}")
        try:
            # PEFT 모델에서 병합된 모델 생성
            merged_model = trainer.model.merge_and_unload()
            
            # 병합된 모델과 프로세서 저장
            merged_model.save_pretrained(
                deployment_model_path,
                safe_serialization=True,  # 안전한 텐서 포맷으로 저장
                max_shard_size="5GB"  # 큰 모델을 위한 샤딩
            )
            
            # VLM 모델의 경우 processor와 tokenizer 모두 저장
            if self.processor:
                self.processor.save_pretrained(deployment_model_path)
                print(f"✅ VLM processor saved to {deployment_model_path}")
                
            if self.tokenizer:
                self.tokenizer.save_pretrained(deployment_model_path)
                print(f"✅ VLM tokenizer saved to {deployment_model_path}")
            
            print(f"✅ Successfully saved deployment-ready VLM model to {deployment_model_path}")
            print(f"✅ VLM model is ready for deployment or distribution")
            
        except Exception as e:
            print(f"❌ Error during VLM model merging: {e}")
            print("Falling back to adapter-only save")

        # GPU 메모리 정리 (self. 직접 사용으로 일관성 유지)
        print("🧹 Clearing GPU cache to free up memory...")
        del trainer
        del self.model  # 일관성을 위해 self.model 직접 참조
        import torch
        torch.cuda.empty_cache()
        
        print(f"✅ VLM fine-tuning completed successfully!")
        print(f"📂 PEFT adapter saved to: {self.output_dir}")
        print(f"📂 Deployment-ready model saved to: {deployment_model_path}")

def main():
    args = parse_args()
    
    # VLM 모델 로더 생성 (VLM 설정 포함)
    model_loader = ModelLoader(args.model_id, vlm_config_path=args.vlm_model_config)

    # --- WandB 조건부 설정 ---
    if args.wandb_token:
        wandb.login(key=args.wandb_token)
        os.environ["WANDB_PROJECT"] = args.wandb_project
        print(f"WandB initialized for project: {args.wandb_project}")
        report_to = ["wandb"]
    else:
        print("WandB token not provided. Skipping WandB initialization.")
        report_to = []
        
    # --- 설정 파일 및 경로 로드 (settings.py 사용) ---
    print("Loading configurations from YAML files...")

    train_config_dict = None
    peft_config_dict = None

    if args.train_config_path:
        train_config_path = settings.config_path / args.train_config_path
        with open(train_config_path, 'r') as f:
            train_config_dict = yaml.safe_load(f)
        print(f"✅ Loaded training configuration from: {train_config_path}")
    
    if args.peft_config_path:
        peft_config_path = settings.config_path / args.peft_config_path
        with open(peft_config_path, 'r') as f:
            peft_config_dict = yaml.safe_load(f)
        print(f"✅ Loaded PEFT configuration from: {peft_config_path}")
    
    # CLI 조건에 따라 report_to 값을 덮어쓰기
    if train_config_dict:
        train_config_dict['report_to'] = report_to
    
    if not model_loader.model or not model_loader.processor:
        print("❌ Failed to load VLM model or processor. Cannot proceed with training.")
        return

    print("Output directory is not specified. Using default settings.")
    output_dir = settings.save_model_path
    output_dir.mkdir(parents=True, exist_ok=True)

    logging_dir = settings.logging_dir
    logging_dir.mkdir(parents=True, exist_ok=True)

    # 데이터셋 경로 설정 - 파이프라인 환경에서는 이전 task의 output을 input1에서 읽음
    if settings.is_pipeline_env:
        dataset_path = settings.pipeline_input_path
    else:
        dataset_path = settings.save_dataset_path_raw  # VLM은 원본 데이터셋 사용 (이미지 포함)

    # VLM 트레이너 생성 및 실행
    trainer = VLMTrainer(
        model_loader, 
        output_dir, 
        dataset_path, 
        vlm_collator_config=args.vlm_collator_config
    )
    trainer.train(
        train_config_dict=train_config_dict, 
        peft_config_dict=peft_config_dict, 
        logging_dir=logging_dir
    )

if __name__ == "__main__":
    main()
