import os
import json
import argparse
import torch
from pathlib import Path
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForLanguageModeling, TrainingArguments
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig
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
    parser.add_argument('--training_args_path', type=str, required=True,
                        help='Path to YAML file for SFTConfig')
    parser.add_argument('--peft_config_path', type=str, required=True,
                        help='Path to YAML file for PEFT config')
    parser.add_argument('--wandb_token', type=str, default=os.getenv('WANDB_API_KEY'),
                        help='Weights & Biases API token for logging.')
    parser.add_argument('--wandb_project', type=str, default="my-llm-finetuning",
                        help='Weights & Biases project name.')
    return parser.parse_args()

class CustomTrainer:
    def __init__(self, model_id, processor, output_dir, dataset_path):
        self.model_id = model_id
        self.output_dir = output_dir
        self.processor = processor
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.processor.tokenizer,
            mlm=False  # we're doing causal LM, not masked LM
        )
        try:
            self.dataset = load_from_disk(dataset_path)
            print(f"Successfully loaded dataset from {dataset_path}")
        except FileNotFoundError:
            self.dataset = None
            print(f"Failed to load dataset. Directory not found: {dataset_path}")

    def train(self, model_loader=None, training_args_dict=None, peft_config_dict=None, logging_dir=None):
        if not model_loader.model:
            print("Model or dataset is not loaded. Cannot proceed with training.")
            return
        
        model = model_loader.model
        processor = self.processor

        if training_args_dict:
            # 사용자가 원하는 세팅의 configuration을 불러옵니다.
            print("Using custom training arguments from YAML file...")
            training_args = SFTConfig(
                    output_dir=self.output_dir / self.model_id,
                    **training_args_dict
                )
        else:
            training_args = SFTConfig(
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
            args=training_args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['validation'],
            processing_class=processor.tokenizer,
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
        
        # 병합된 모델 저장을 위한 경로 설정
        merged_model_path = self.output_dir.parent / "merged_model"
        merged_model_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Merging LoRA weights and saving full model to {merged_model_path}")
        try:
            # PEFT 모델에서 병합된 모델 생성
            merged_model = trainer.model.merge_and_unload()
            
            # 병합된 모델과 토크나이저 저장
            merged_model.save_pretrained(
                merged_model_path,
                safe_serialization=True,  # 안전한 텐서 포맷으로 저장
                max_shard_size="5GB"  # 큰 모델을 위한 샤딩
            )
            processor.tokenizer.save_pretrained(merged_model_path)
            
            
            print(f"✅ Successfully saved merged model and tokenizer to {merged_model_path}")
            print(f"✅ Model is ready for deployment or distribution")
            
        except Exception as e:
            print(f"❌ Error during model merging: {e}")
            print("Falling back to adapter-only save")

        print("Clearing GPU cache to free up memory...")
        del trainer
        del model
        torch.cuda.empty_cache()
        
        print(f"Model training completed. Adapter saved to {self.output_dir}, Merged model saved to {merged_model_path}")


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

    training_args_dict = None
    peft_config_dict = None

    if args.training_args_path:
        with open(settings.config_path / args.training_args_path, 'r') as f:
            training_args_dict = yaml.safe_load(f)
    
    if args.peft_config_path:
        with open(settings.config_path / args.peft_config_path, 'r') as f:
            peft_config_dict = yaml.safe_load(f)
    
    # CLI 조건에 따라 report_to 값을 덮어쓰기
    training_args_dict['report_to'] = report_to
    
    if not model_loader.model or not model_loader.processor:
        print("Failed to load model or processor. Cannot proceed with training.")
        return

    print("Output directory is not specified. Using default settings.")
    output_dir = settings.save_model_path
    output_dir.mkdir(parents=True, exist_ok=True)

    logging_dir = settings.logging_dir
    logging_dir.mkdir(parents=True, exist_ok=True)

    trainer = CustomTrainer(args.model_id, model_loader.processor, output_dir, settings.save_dataset_path)
    trainer.train(model_loader = model_loader, training_args_dict = training_args_dict, peft_config_dict = peft_config_dict, logging_dir=logging_dir)

if __name__ == "__main__":
    main()