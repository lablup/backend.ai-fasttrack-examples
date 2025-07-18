import os
import argparse
import json
import torch
import logging
import sys
from pathlib import Path
from datasets import load_from_disk  # 'load_dataset' 대신 'load_from_disk'를 사용
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    pipeline,
    Gemma3nForConditionalGeneration,
)
from peft import PeftModel
from tqdm import tqdm
import evaluate

# # 프로젝트 루트를 sys.path에 추가
# project_root = Path(__file__).parent.parent.parent
# sys.path.insert(0, str(project_root))

from src.models.model import ModelLoader
from configs.settings import settings

# tqdm을 통한 진행률 표시를 위해 로깅 레벨을 설정합니다.
logging.getLogger("transformers.pipelines.pt_utils").setLevel(logging.ERROR)

def parse_args():
    """CLI 인자를 파싱하는 함수"""
    parser = argparse.ArgumentParser(description="Evaluate a model on a pre-processed local dataset.")
    parser.add_argument('--model_name_or_path', type=str, default=os.getenv('MODEL_ID'),required=True, help='Base model identifier')
    # '--dataset_name'을 '--dataset_path'로 변경하여 로컬 경로를 받습니다.
    parser.add_argument('--output_path', type=str, default="output.json", help='Path to save the evaluation results.')
    parser.add_argument('--max_samples', type=int, default=None, help='Optional: Maximum number of samples for quick testing.')
    parser.add_argument('--use_adapter', action='store_true', 
                        help='Use a PEFT adapter if this flag is set. If not, evaluates the base model.')
    return parser.parse_args()

def load_model_for_evaluation(model_name_or_path, use_finetuned=False):
    """베이스 모델 또는 파인튜닝된 모델을 로드하는 함수"""
    
    if use_finetuned:
        # 1. 먼저 병합된 모델이 있는지 확인
        merged_model_path = settings.merged_model_path
        if merged_model_path.exists() and (merged_model_path / "config.json").exists():
            print(f"Found merged fine-tuned model at: {merged_model_path}")
            try:
                if 'gemma-3' in model_name_or_path.lower():
                    model = Gemma3nForConditionalGeneration.from_pretrained(
                        str(merged_model_path),
                        device_map="auto", 
                        torch_dtype=torch.float16,
                        local_files_only=True  # 로컬 파일만 사용
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        str(merged_model_path),
                        device_map="auto", 
                        torch_dtype=torch.float16,
                        local_files_only=True
                    )
                print(f"✅ Successfully loaded merged fine-tuned model from {merged_model_path}")
                return model
            except Exception as e:
                print(f"⚠️ Failed to load merged model: {e}")
                print("Falling back to adapter-based loading...")
        
        # 2. 병합된 모델이 없으면 어댑터 방식으로 로딩
        adapter_path = settings.save_model_path
        if adapter_path.exists():
            print(f"Loading base model and applying PEFT adapter from: {adapter_path}")
            
            # 베이스 모델 로드
            if 'gemma-3' in model_name_or_path.lower():
                base_model = Gemma3nForConditionalGeneration.from_pretrained(
                    model_name_or_path, 
                    device_map="auto", 
                    torch_dtype=torch.float16, 
                    token=os.getenv('HF_TOKEN')
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path, 
                    device_map="auto", 
                    torch_dtype=torch.float16, 
                    token=os.getenv('HF_TOKEN')
                )
            
            # PEFT 어댑터 적용
            try:
                model = PeftModel.from_pretrained(
                    base_model, 
                    str(adapter_path), 
                    device_map='auto', 
                    torch_dtype=torch.float16
                )
                print(f"✅ Successfully loaded model with PEFT adapter from {adapter_path}")
                return model
            except Exception as e:
                print(f"❌ Failed to load PEFT adapter: {e}")
                print("Using base model instead")
                return base_model
        else:
            print(f"❌ No fine-tuned model found at {adapter_path}")
            print("Using base model instead")
    
    # 베이스 모델 로딩
    print(f"Loading base model: {model_name_or_path}")
    if 'gemma-3' in model_name_or_path.lower():
        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            device_map="auto", 
            torch_dtype=torch.float16, 
            token=os.getenv('HF_TOKEN')
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            device_map="auto", 
            torch_dtype=torch.float16, 
            token=os.getenv('HF_TOKEN')
        )
    
    print(f"✅ Successfully loaded base model: {model_name_or_path}")
    return model

def main():
    args = parse_args()
    print("--- Starting Evaluation Script on Local Dataset ---")
    
    # 1. 평가 지표 로드
    rouge_metric = evaluate.load("rouge")
    bertscore_metric = evaluate.load("bertscore")
    
    # 2. 모델 로더를 통해 tokenizer 로드
    processor_path = args.model_name_or_path
    if args.use_adapter and settings.merged_model_path.exists() and (settings.merged_model_path / "tokenizer_config.json").exists():
        processor_path = str(settings.merged_model_path)
        print(f"Loading tokenizer from fine-tuned model: {processor_path}")
    else:
        print(f"Loading tokenizer from base model: {processor_path}")
    
    # ModelLoader 사용 (모델은 따로 로드할 예정이므로 tokenizer만 필요)
    model_loader = ModelLoader(processor_path)
    
    if not model_loader.tokenizer:
        print(f"❌ Failed to load tokenizer from {processor_path}")
        return
    
    tokenizer = model_loader.tokenizer
        
    dataset_path = settings.save_dataset_path_formatted
    try:
        # load_from_disk를 사용하여 저장된 데이터셋 전체를 불러옵니다.
        full_dataset = load_from_disk(dataset_path)
        # 평가에는 'test' 스플릿만 사용합니다.
        test_dataset = full_dataset['test']
        print(f"Successfully loaded test split from {dataset_path}")
        print(f"Test dataset columns: {test_dataset.column_names}")
    except FileNotFoundError:
        print(f"Error: Dataset directory not found at {dataset_path}")
        return
    except Exception as e:
        print(f"Failed to load dataset from disk: {e}")
        return

    # 3. 평가할 모델 로드 - 병합된 모델 우선 사용
    model = load_model_for_evaluation(args.model_name_or_path, use_finetuned=args.use_adapter)
    # 4. 파이프라인 설정
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.float16, # 데이터 타입을 명시적으로 지정
                    max_new_tokens=256,
                    device_map="auto"
                    )

    # 5. 평가 실행: 전처리된 데이터 직접 사용
    if args.max_samples:
        test_dataset = test_dataset.select(range(args.max_samples))
    
    # 배치 크기 설정 (GPU 메모리에 따라 조절)
    batch_size = 16
    
    all_predictions = []
    all_references = []
    all_prompts = []
    
    print(f"Evaluating on {len(test_dataset)} samples with batch size {batch_size}...")
    
    # tqdm으로 전체 데이터셋에 대한 진행률을 표시합니다.
    for i in tqdm(range(0, len(test_dataset), batch_size)):
        # 1. 현재 배치에 해당하는 데이터(딕셔너리)를 가져옵니다.
        batch = test_dataset[i : i + batch_size]
        
        # 2. 전처리된 데이터에서 prompt와 reference 직접 추출
        batch_prompts = batch.get("prompt", [])
        batch_references = batch.get("reference", [])

        # 3. 파이프라인으로 현재 배치를 한 번에 처리
        generated_outputs = pipe(batch_prompts, batch_size=len(batch_prompts), eos_token_id=tokenizer.eos_token_id)
        
        # 4. 생성된 결과에서 예측 텍스트만 추출
        batch_predictions = [
            out[0]['generated_text'].replace(prompt, '').strip()
            for out, prompt in zip(generated_outputs, batch_prompts)
        ]
        
        all_predictions.extend(batch_predictions)
        all_references.extend(batch_references)
        all_prompts.extend(batch_prompts)

    # 5. 정량적 성능 지표 계산 및 결과 저장 (이전과 동일)
    print("\nCalculating quantitative metrics...")
    rouge_scores = rouge_metric.compute(predictions=all_predictions, references=all_references)
    bert_scores = bertscore_metric.compute(predictions=all_predictions, references=all_references, lang="en")
    avg_bert_f1 = sum(bert_scores['f1']) / len(bert_scores['f1']) if bert_scores['f1'] else 0.0
    
    sample_results = [
        {"prompt": p, "ground_truth": r, "model_prediction": pred} 
        for p, r, pred in zip(all_prompts, all_references, all_predictions)
    ]

    final_results = {
        "summary_metrics": {"rouge": rouge_scores, "bertscore_f1_avg": avg_bert_f1},
        "sample_results": sample_results
    }
    
    print("\n--- Evaluation Metrics Summary ---")
    print(f"ROUGE Scores: {rouge_scores}")
    print(f"BERTScore (F1 Average): {avg_bert_f1:.4f}")

    output_path = settings.evaluation_output_path / args.output_path

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
        
    print(f"\nEvaluation complete. Results saved to {output_path}")
    
    print("\nClearing GPU memory...")
    del model
    del pipe
    torch.cuda.empty_cache()
    print("GPU memory cleared.")

if __name__ == "__main__":
    main()