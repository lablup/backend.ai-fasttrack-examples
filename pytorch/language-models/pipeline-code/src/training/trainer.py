import os
import json
import argparse
import torch
import sys
from pathlib import Path
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForLanguageModeling, TrainingArguments
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig

# # 프로젝트 루트를 sys.path에 추가
# project_root = Path(__file__).parent.parent.parent
# sys.path.insert(0, str(project_root))

from src.models.model import ModelLoader
from configs.settings import settings
import yaml
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Trainer Configuration")
    parser.add_argument('--model_id', type=str, default=os.getenv('MODEL_ID'),
                        help='ID of the model to load from Hugging Face Hub')
    parser.add_argument('--dataset-name', type=str, default=os.getenv('DATASET'),
                        help='Name of the dataset to load from Hugging Face Hub')
    parser.add_argument('--train_config_path', type=str, required=True,
                        help='Path to YAML file for SFTConfig (e.g., train_config.yaml)')
    parser.add_argument('--peft_config_path', type=str, required=True,
                        help='Path to YAML file for PEFT config')
    parser.add_argument('--wandb_token', type=str, default=os.getenv('WANDB_API_KEY'),
                        help='Weights & Biases API token for logging.')
    parser.add_argument('--wandb_project', type=str, default="my-llm-finetuning",
                        help='Weights & Biases project name.')
    return parser.parse_args()

class CustomTrainer:
    def __init__(self, model_loader, output_dir, dataset_path):
        self.model_loader = model_loader

        self.model = self.model_loader.model
        self.tokenizer = self.model_loader.tokenizer

        self.output_dir = output_dir
        
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # we're doing causal LM, not masked LM
        )
        try:
            self.dataset = load_from_disk(dataset_path)
            print(f"Successfully loaded dataset from {dataset_path}")
        except FileNotFoundError:
            self.dataset = None
            print(f"Failed to load dataset. Directory not found: {dataset_path}")

    def train(self, train_config_dict=None, peft_config_dict=None, logging_dir=None):
        if not self.model or not self.tokenizer:
            print("Model or dataset is not loaded. Cannot proceed with training.")
            return
        
        model = self.model
        tokenizer = self.tokenizer

        if train_config_dict:
            # 사용자가 원하는 세팅의 configuration을 불러옵니다.
            print("Using custom training arguments from YAML file...")
            train_config = SFTConfig(
                    output_dir=self.output_dir / self.model_loader.model_id,
                    **train_config_dict
                )
        else:
            train_config = SFTConfig(
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                gradient_accumulation_steps=2,
                gradient_checkpointing=True,
                learning_rate=1e-05,
                num_train_epochs=1.0,
                logging_steps=0.2,
                logging_strategy="steps",
                # save_steps=1,
                load_best_model_at_end = True,
                metric_for_best_model = "eval_loss",  # 또는 사용하는 평가 메트릭
                greater_is_better = False,  # loss의 경우 false, accuracy 등은 true
                report_to=["wandb"],
                run_name='gemma-3n-E2B-it-trl-sft',
                fp16=False,
                bf16=False,
                group_by_length=True,
            )

        if peft_config_dict:
            # 사용자가 원하는 세팅의 configuration을 불러옵니다.
            print("Using custom PEFT configuration from YAML file...")
            peft_config = LoraConfig(**peft_config_dict)
        else:
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                r=64,
                target_modules="all-linear",
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
            )

        trainer = SFTTrainer(
            model=model,
            args=train_config,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['validation'],
            processing_class=tokenizer,
            peft_config=peft_config,
            # data_collator=self.data_collator,
        )
        print("Starting training...")
        train_result = trainer.train()
        trainer_stats = train_result.metrics

        log_file_path = logging_dir / "training_stats.json"
        print(f"Saving training stats to {log_file_path}")
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(trainer_stats, f, ensure_ascii=False, indent=4)

        print(f"Training finished. Saving PEFT adapter to {self.output_dir}")
        trainer.save_model()  # PEFT 어댑터만 저장
        
        # 병합된 배포용 모델 저장을 위한 경로 설정 (settings에서 관리)
        deployment_model_path = settings.deployment_model_path
        deployment_model_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Merging LoRA weights and saving deployment-ready model to {deployment_model_path}")
        try:
            # PEFT 모델에서 병합된 모델 생성
            merged_model = trainer.model.merge_and_unload()
            
            # 병합된 모델과 토크나이저 저장
            merged_model.save_pretrained(
                deployment_model_path,
                safe_serialization=True,  # 안전한 텐서 포맷으로 저장
                max_shard_size="5GB"  # 큰 모델을 위한 샤딩
            )
            tokenizer.save_pretrained(deployment_model_path)
            
            
            print(f"✅ Successfully saved deployment-ready model and tokenizer to {deployment_model_path}")
            print(f"✅ Model is ready for deployment or distribution")
            
        except Exception as e:
            print(f"❌ Error during model merging: {e}")
            print("Falling back to adapter-only save")

        
        print("Clearing GPU cache to free up memory...")
        del trainer
        del model
        torch.cuda.empty_cache()
        
        print(f"Model training completed. Adapter saved to {self.output_dir}, Deployment-ready model saved to {deployment_model_path}")
def main():
    args = parse_args()
    model_loader = ModelLoader(args.model_id)

    # --- WandB 조건부 설정 ---
    if args.wandb_token:
        print("W&B token provided. Logging in to Weights & Biases...")
        try:
            wandb.login(key=args.wandb_token)
            wandb.init(
                project=args.wandb_project,
                name=f"{args.model_id}-trl-sft-{args.dataset_name}",
            )
            os.environ['WANDB_PROJECT'] = args.wandb_project
            report_to = "wandb"
        except Exception as e:
            print(f"Failed to login to W&B: {e}")
            report_to = "none" # 로그인 실패 시 로깅 비활성화
    else:
        print("No W&B token provided. Disabling Weights & Biases logging.")
        os.environ['WANDB_DISABLED'] = 'true'
        report_to = "none"
        
    # --- 설정 파일 및 경로 로드 (settings.py 사용) ---
    print("Loading configurations from YAML files...")

    train_config_dict = None
    peft_config_dict = None

    if args.train_config_path:
        with open(settings.config_path / args.train_config_path, 'r') as f:
            train_config_dict = yaml.safe_load(f)
    
    if args.peft_config_path:
        with open(settings.config_path / args.peft_config_path, 'r') as f:
            peft_config_dict = yaml.safe_load(f)
    
    # CLI 조건에 따라 report_to 값을 덮어쓰기
    train_config_dict['report_to'] = report_to
    
    if not model_loader.model or not model_loader.tokenizer:
        print("Failed to load model or tokenizer. Cannot proceed with training.")
        return

    print("Output directory is not specified. Using default settings.")
    output_dir = settings.save_model_path
    output_dir.mkdir(parents=True, exist_ok=True)

    logging_dir = settings.logging_dir
    logging_dir.mkdir(parents=True, exist_ok=True)

    # 데이터셋 경로 설정 - 파이프라인 환경에서는 이전 task의 output을 input1에서 읽음
    if settings.is_pipeline_env:
        readonly_dataset_path = settings.pipeline_input_path
        # 읽기 전용 경로를 쓰기 가능한 임시 경로로 복사
        dataset_path = settings.copy_readonly_to_writable(readonly_dataset_path, 'training')
    else:
        dataset_path = settings.save_dataset_path_formatted

    trainer = CustomTrainer(model_loader, output_dir, dataset_path)
    trainer.train(train_config_dict=train_config_dict, peft_config_dict=peft_config_dict, logging_dir=logging_dir)

if __name__ == "__main__":
    main()