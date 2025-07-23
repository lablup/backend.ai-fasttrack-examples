import os
import argparse
import json
import torch
import logging
import sys
import re
from pathlib import Path
from datasets import load_from_disk  # 'load_dataset' 대신 'load_from_disk'를 사용
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    pipeline,
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

def detect_language(text_sample):
    """
    텍스트 샘플을 분석하여 언어를 감지하는 함수 (한국어/영어/기타로 간소화)
    
    Args:
        text_sample (str): 분석할 텍스트 샘플
        
    Returns:
        dict: {"language": str, "confidence": str}
    """
    if not text_sample or len(text_sample.strip()) == 0:
        return {"language": "other", "confidence": "low"}
    
    text = text_sample[:500]  # 처음 500자만 분석
    
    # 한글 문자 감지
    korean_chars = len(re.findall(r'[\uac00-\ud7af]', text))
    
    # 영어 문자 감지 (라틴 문자 + 영어 단어 패턴)
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    english_words = len(re.findall(r'\b(the|and|or|but|in|on|at|to|for|of|with|by|is|are|was|were|a|an|this|that|will|have|has|can|could|would|should)\b', text.lower()))
    
    total_chars = len(re.findall(r'[^\s\d\W]', text))
    total_words = len(re.findall(r'\b\w+\b', text))
    
    if total_chars == 0:
        return {"language": "other", "confidence": "low"}
    
    # 한국어 비율 계산
    korean_ratio = korean_chars / total_chars
    
    # 영어 비율 계산 (라틴 문자 비율 + 영어 단어 비율)
    latin_ratio = latin_chars / total_chars
    english_word_ratio = english_words / total_words if total_words > 0 else 0
    
    # 언어 결정 로직
    if korean_ratio > 0.3:  # 한국어 문자가 30% 이상
        return {"language": "korean", "confidence": "high" if korean_ratio > 0.7 else "medium"}
    
    elif latin_ratio > 0.5 and english_word_ratio > 0.05:  # 라틴 문자 50% + 영어 단어 5% 이상
        confidence = "high" if english_word_ratio > 0.15 else "medium"
        return {"language": "english", "confidence": confidence}
    
    else:  # 기타 언어 (중국어, 일본어, 아랍어 등)
        return {"language": "other", "confidence": "medium"}

def get_bertscore_model_config(language_info):
    """
    언어 정보를 바탕으로 최적의 BERTScore 모델 설정을 반환 (간소화된 버전)
    
    Args:
        language_info (dict): detect_language 함수의 반환값
        
    Returns:
        dict: {"model_type": str, "lang": str, "description": str}
    """
    language = language_info["language"]
    confidence = language_info["confidence"]
    
    # 한국어: 다국어 BERT 사용 (한국어에 최적화)
    if language == "korean":
        return {
            "model_type": "bert-base-multilingual-cased",
            "lang": "ko",
            "description": "Multilingual BERT for Korean"
        }
    
    # 영어: DeBERTa-v3 사용 (영어 성능 우수)
    elif language == "english" and confidence == "high":
        return {
            "model_type": "microsoft/deberta-v3-large",
            "lang": "en", 
            "description": "DeBERTa-v3 for English"
        }
    
    # 기타 모든 언어 또는 낮은 신뢰도: 다국어 BERT 사용
    else:
        return {
            "model_type": "bert-base-multilingual-cased",
            "lang": None,  # 자동 언어 감지
            "description": "Multilingual BERT for other languages"
        }

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
    """베이스 모델 또는 파인튜닝된 모델을 로드하는 함수 (AutoModelForCausalLM 사용)"""
    
    if use_finetuned:
        # 1. 먼저 배포용 모델이 있는지 확인
        deployment_model_path = settings.deployment_model_path
        if deployment_model_path.exists() and (deployment_model_path / "config.json").exists():
            print(f"Found deployment-ready fine-tuned model at: {deployment_model_path}")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    str(deployment_model_path),
                        device_map="auto", 
                        torch_dtype=torch.float16,
                        local_files_only=True  # 로컬 파일만 사용
                    )
                print(f"✅ Successfully loaded deployment-ready fine-tuned model from {deployment_model_path}")
                return model
            except Exception as e:
                print(f"⚠️ Failed to load deployment model: {e}")
                print("Falling back to adapter-based loading...")
        
        # 2. 병합된 모델이 없으면 어댑터 방식으로 로딩
        adapter_path = settings.save_model_path / model_name_or_path
        if adapter_path.exists():
            print(f"Loading base model and applying PEFT adapter from: {adapter_path}")
            
            # 베이스 모델 로드
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
    
    # 1. 평가 지표 로드 - 한국어 텍스트 평가에 적합한 메트릭 사용
    print("Loading evaluation metrics...")
    rouge_metric = evaluate.load("rouge")
    bertscore_metric = evaluate.load("bertscore")
    
    # BLEU 메트릭 추가 (다국어 지원)
    try:
        bleu_metric = evaluate.load("bleu")
        print("✅ BLEU metric loaded successfully")
    except Exception as e:
        print(f"⚠️ Failed to load BLEU metric: {e}")
        bleu_metric = None
    
    # Perplexity 메트릭 추가 (언어 모델 품질 평가)
    try:
        perplexity_metric = evaluate.load("perplexity", module_type="metric")
        print("✅ Perplexity metric loaded successfully")
    except Exception as e:
        print(f"⚠️ Failed to load Perplexity metric: {e}")
        perplexity_metric = None
    
    # 2. 모델 로더를 통해 tokenizer 로드
    processor_path = args.model_name_or_path
    if args.use_adapter and settings.deployment_model_path.exists() and (settings.deployment_model_path / "tokenizer_config.json").exists():
        processor_path = str(settings.deployment_model_path)
        print(f"Loading tokenizer from fine-tuned model: {processor_path}")
    else:
        print(f"Loading tokenizer from base model: {processor_path}")
    
    # ModelLoader 사용 (모델은 따로 로드할 예정이므로 tokenizer만 필요)
    model_loader = ModelLoader(processor_path)
    
    if not model_loader.tokenizer:
        print(f"❌ Failed to load tokenizer from {processor_path}")
        return
    
    tokenizer = model_loader.tokenizer
    
    # 데이터셋 경로 설정 - 파이프라인 환경에서는 이전 task의 output을 input1에서 읽음
    if settings.is_pipeline_env:
        readonly_dataset_path = settings.pipeline_input_path
        # 읽기 전용 경로를 쓰기 가능한 임시 경로로 복사
        dataset_path = settings.copy_readonly_to_writable(readonly_dataset_path, 'evaluation')
    else:
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
    batch_size = int(os.getenv('EVAL_BATCH_SIZE', 8))
    
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
        
        # 빈 값 필터링 및 검증
        valid_indices = []
        filtered_prompts = []
        filtered_references = []
        
        for idx, (prompt, ref) in enumerate(zip(batch_prompts, batch_references)):
            if prompt and ref and len(str(prompt).strip()) > 0 and len(str(ref).strip()) > 0:
                valid_indices.append(idx)
                filtered_prompts.append(str(prompt).strip())
                filtered_references.append(str(ref).strip())
        
        if not filtered_prompts:
            print(f"⚠️ Skipping batch {i//batch_size + 1}: No valid prompt-reference pairs")
            continue

        # 3. 파이프라인으로 현재 배치를 한 번에 처리
        try:
            generated_outputs = pipe(filtered_prompts, batch_size=len(filtered_prompts), eos_token_id=tokenizer.eos_token_id)
        except Exception as e:
            print(f"⚠️ Generation failed for batch {i//batch_size + 1}: {e}")
            continue
        
        # 4. 생성된 결과에서 예측 텍스트만 추출
        batch_predictions = []
        for out, prompt in zip(generated_outputs, filtered_prompts):
            try:
                generated_text = out[0]['generated_text']
                # 프롬프트 제거하여 순수 생성 텍스트만 추출
                prediction = generated_text.replace(prompt, '').strip()
                batch_predictions.append(prediction if prediction else "[EMPTY_GENERATION]")
            except Exception as e:
                print(f"⚠️ Failed to extract prediction: {e}")
                batch_predictions.append("[EXTRACTION_ERROR]")
        
        all_predictions.extend(batch_predictions)
        all_references.extend(filtered_references)
        all_prompts.extend(filtered_prompts)

    # 5. 정량적 성능 지표 계산 및 결과 저장 (개선된 평가 메트릭)
    print("\nCalculating quantitative metrics...")
    
    # 데이터 유효성 검증
    if not all_predictions or not all_references:
        print("❌ No valid predictions or references found. Evaluation cannot proceed.")
        return
    
    if len(all_predictions) != len(all_references):
        print(f"⚠️ Mismatch in prediction/reference counts: {len(all_predictions)} vs {len(all_references)}")
        min_len = min(len(all_predictions), len(all_references))
        all_predictions = all_predictions[:min_len]
        all_references = all_references[:min_len]
        all_prompts = all_prompts[:min_len]
    
    print(f"📊 Valid samples for evaluation: {len(all_predictions)}")
    
    # ROUGE 점수 계산 (텍스트 요약 평가의 표준 메트릭)
    print("Computing ROUGE scores...")
    try:
        rouge_scores = rouge_metric.compute(predictions=all_predictions, references=all_references)
        
        # ROUGE 점수 검증
        for metric_name, score in rouge_scores.items():
            if not (0 <= score <= 1):
                print(f"⚠️ Unusual ROUGE {metric_name} score: {score:.4f}")
        
        print(f"✅ ROUGE scores computed successfully")
        
    except Exception as e:
        print(f"❌ ROUGE calculation failed: {e}")
        rouge_scores = {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "rougeLsum": 0.0,
            "error": str(e)
        }
    
    # BERTScore 계산 - 다국어 지원 및 최적 모델 자동 선택
    print("Computing BERTScore...")
    bert_scores_result = None
    try:
        # 언어 감지를 위한 샘플 텍스트 분석 (더 많은 샘플 사용)
        sample_texts = []
        sample_size = min(10, len(all_references), len(all_predictions))
        for i in range(sample_size):
            if all_references[i] and all_predictions[i]:
                sample_texts.append(all_references[i][:100] + " " + all_predictions[i][:100])
        
        combined_sample = " ".join(sample_texts)[:1000]  # 최대 1000자 분석
        
        # 언어 감지 실행
        language_info = detect_language(combined_sample)
        print(f"� Language detection: {language_info['language']} (confidence: {language_info['confidence']})")
        print(f"📊 Script ratios: {language_info.get('ratios', {})}")
        
        # 최적 BERTScore 모델 설정 가져오기
        model_config = get_bertscore_model_config(language_info)
        print(f"🎯 Selected model: {model_config['description']}")
        
        # BERTScore 계산 실행
        bert_compute_kwargs = {
            "predictions": all_predictions,
            "references": all_references,
            "model_type": model_config["model_type"]
        }
        
        # 언어 코드가 있는 경우에만 추가
        if model_config["lang"]:
            bert_compute_kwargs["lang"] = model_config["lang"]
        
        bert_scores = bertscore_metric.compute(**bert_compute_kwargs)
        
        # BERTScore 결과 처리 및 검증
        if bert_scores and 'f1' in bert_scores and len(bert_scores['f1']) > 0:
            # 유효하지 않은 점수 필터링 (NaN, 무한대 등)
            valid_f1 = [score for score in bert_scores['f1'] if not (torch.isnan(torch.tensor(score)) or torch.isinf(torch.tensor(score)))]
            valid_precision = [score for score in bert_scores['precision'] if not (torch.isnan(torch.tensor(score)) or torch.isinf(torch.tensor(score)))]
            valid_recall = [score for score in bert_scores['recall'] if not (torch.isnan(torch.tensor(score)) or torch.isinf(torch.tensor(score)))]
            
            if len(valid_f1) == 0:
                raise ValueError("All BERTScore F1 scores are invalid (NaN/Inf)")
            
            avg_bert_f1 = sum(valid_f1) / len(valid_f1)
            avg_bert_precision = sum(valid_precision) / len(valid_precision)
            avg_bert_recall = sum(valid_recall) / len(valid_recall)
            
            # 점수 범위 검증 (BERTScore는 보통 0~1 범위)
            if not (0 <= avg_bert_f1 <= 1) or not (0 <= avg_bert_precision <= 1) or not (0 <= avg_bert_recall <= 1):
                print(f"⚠️ Unusual BERTScore values detected: F1={avg_bert_f1:.4f}, P={avg_bert_precision:.4f}, R={avg_bert_recall:.4f}")
            
            bert_scores_result = {
                "f1_avg": avg_bert_f1,
                "precision_avg": avg_bert_precision,
                "recall_avg": avg_bert_recall,
                "f1_scores": valid_f1[:5],  # 처음 5개 샘플의 개별 점수
                "model_used": model_config["model_type"],
                "language_detected": language_info["language"],
                "language_confidence": language_info["confidence"],
                "valid_samples": len(valid_f1),
                "total_samples": len(bert_scores['f1'])
            }
            print(f"✅ BERTScore computed successfully (F1: {avg_bert_f1:.4f}, P: {avg_bert_precision:.4f}, R: {avg_bert_recall:.4f})")
            print(f"📈 Valid/Total samples: {len(valid_f1)}/{len(bert_scores['f1'])}")
        else:
            raise ValueError("Empty or invalid BERTScore results")
            
    except Exception as e:
        print(f"⚠️ Primary BERTScore calculation failed: {e}")
        print("🔄 Trying fallback models...")
        
        # 폴백 시퀀스: 여러 모델 시도
        fallback_models = [
            {"model_type": "bert-base-multilingual-cased", "lang": None, "desc": "Multilingual BERT"},
            {"model_type": "distilbert-base-uncased", "lang": "en", "desc": "DistilBERT English"},
            {"model_type": "distilbert-base-multilingual-cased", "lang": None, "desc": "DistilBERT Multilingual"}
        ]
        
        for fallback in fallback_models:
            try:
                print(f"🔄 Trying {fallback['desc']}...")
                fallback_kwargs = {
                    "predictions": all_predictions,
                    "references": all_references,
                    "model_type": fallback["model_type"]
                }
                if fallback["lang"]:
                    fallback_kwargs["lang"] = fallback["lang"]
                
                bert_scores = bertscore_metric.compute(**fallback_kwargs)
                
                if bert_scores and 'f1' in bert_scores and len(bert_scores['f1']) > 0:
                    avg_bert_f1 = sum(bert_scores['f1']) / len(bert_scores['f1'])
                    avg_bert_precision = sum(bert_scores['precision']) / len(bert_scores['precision'])
                    avg_bert_recall = sum(bert_scores['recall']) / len(bert_scores['recall'])
                    
                    bert_scores_result = {
                        "f1_avg": avg_bert_f1,
                        "precision_avg": avg_bert_precision,
                        "recall_avg": avg_bert_recall,
                        "model_used": fallback["model_type"],
                        "language_detected": "fallback",
                        "language_confidence": "unknown",
                        "fallback_used": True
                    }
                    print(f"✅ Fallback BERTScore computed with {fallback['desc']} (F1: {avg_bert_f1:.4f})")
                    break
                    
            except Exception as fallback_error:
                print(f"❌ {fallback['desc']} failed: {fallback_error}")
                continue
        
        # 모든 폴백 실패
        if bert_scores_result is None:
            print(f"❌ All BERTScore models failed")
            bert_scores_result = {
                "f1_avg": 0.0,
                "precision_avg": 0.0,
                "recall_avg": 0.0,
                "error": "All models failed",
                "model_used": "none",
                "language_detected": "unknown",
                "language_confidence": "unknown"
            }
    
    # BLEU 점수 계산 (기계 번역/생성 태스크의 표준 메트릭)
    bleu_scores = None
    if bleu_metric:
        print("Computing BLEU scores...")
        try:
            # BLEU는 references를 리스트의 리스트로 요구함
            references_for_bleu = [[ref] for ref in all_references]
            bleu_scores = bleu_metric.compute(predictions=all_predictions, references=references_for_bleu)
            
            # BLEU 점수 검증 (0~1 범위여야 함)
            bleu_score = bleu_scores.get('bleu', 0)
            if not (0 <= bleu_score <= 1):
                print(f"⚠️ Unusual BLEU score: {bleu_score:.4f}")
            
            print(f"✅ BLEU score computed: {bleu_score:.4f}")
            
        except Exception as e:
            print(f"⚠️ BLEU calculation failed: {e}")
            bleu_scores = {"bleu": 0.0, "error": str(e)}
    else:
        print("⚠️ BLEU metric not available")
        bleu_scores = {"bleu": 0.0, "error": "BLEU metric not loaded"}
    
    # 추가 평가 메트릭들
    print("Computing additional metrics...")
    
    # 1. 평균 길이 비교 (요약 태스크에서 중요한 메트릭)
    pred_lengths = [len(pred.split()) for pred in all_predictions]
    ref_lengths = [len(ref.split()) for ref in all_references]
    
    avg_pred_length = sum(pred_lengths) / len(pred_lengths)
    avg_ref_length = sum(ref_lengths) / len(ref_lengths)
    length_ratio = avg_pred_length / avg_ref_length if avg_ref_length > 0 else 0.0
    
    # 길이 분포 분석
    import statistics
    pred_length_stats = {
        "mean": avg_pred_length,
        "median": statistics.median(pred_lengths),
        "std": statistics.stdev(pred_lengths) if len(pred_lengths) > 1 else 0,
        "min": min(pred_lengths),
        "max": max(pred_lengths)
    }
    
    ref_length_stats = {
        "mean": avg_ref_length,
        "median": statistics.median(ref_lengths),
        "std": statistics.stdev(ref_lengths) if len(ref_lengths) > 1 else 0,
        "min": min(ref_lengths),
        "max": max(ref_lengths)
    }
    
    # 2. Exact Match 비율 (대소문자 무시, 공백 정규화)
    exact_matches = 0
    normalized_exact_matches = 0
    
    for pred, ref in zip(all_predictions, all_references):
        # 완전 일치
        if pred.strip() == ref.strip():
            exact_matches += 1
        
        # 정규화된 일치 (대소문자 무시, 공백 정규화)
        pred_normalized = re.sub(r'\s+', ' ', pred.strip().lower())
        ref_normalized = re.sub(r'\s+', ' ', ref.strip().lower())
        if pred_normalized == ref_normalized:
            normalized_exact_matches += 1
    
    exact_match_ratio = exact_matches / len(all_predictions)
    normalized_exact_match_ratio = normalized_exact_matches / len(all_predictions)
    
    # 3. 빈 생성 비율 체크
    empty_predictions = sum(1 for pred in all_predictions if len(pred.strip()) == 0)
    empty_prediction_ratio = empty_predictions / len(all_predictions)
    
    # 4. 에러 생성 비율 체크
    error_predictions = sum(1 for pred in all_predictions if pred.strip() in ["[EMPTY_GENERATION]", "[EXTRACTION_ERROR]"])
    error_prediction_ratio = error_predictions / len(all_predictions)
    
    print(f"✅ Additional metrics computed:")
    print(f"  - Empty predictions: {empty_predictions}/{len(all_predictions)} ({empty_prediction_ratio:.4f})")
    print(f"  - Error predictions: {error_predictions}/{len(all_predictions)} ({error_prediction_ratio:.4f})")
    print(f"  - Exact matches: {exact_matches}/{len(all_predictions)} ({exact_match_ratio:.4f})")
    print(f"  - Normalized matches: {normalized_exact_matches}/{len(all_predictions)} ({normalized_exact_match_ratio:.4f})")
    
    # 결과 정리
    summary_metrics = {
        "rouge": rouge_scores,
        "bertscore": bert_scores_result,
        "bleu": bleu_scores,
        "length_analysis": {
            "prediction_stats": pred_length_stats,
            "reference_stats": ref_length_stats,
            "length_ratio": length_ratio
        },
        "exact_match_analysis": {
            "exact_match_ratio": exact_match_ratio,
            "normalized_exact_match_ratio": normalized_exact_match_ratio,
            "exact_matches": exact_matches,
            "normalized_exact_matches": normalized_exact_matches
        },
        "quality_analysis": {
            "empty_prediction_ratio": empty_prediction_ratio,
            "error_prediction_ratio": error_prediction_ratio,
            "empty_predictions": empty_predictions,
            "error_predictions": error_predictions
        },
        "total_samples": len(all_predictions)
    }
    
    sample_results = [
        {
            "prompt": p, 
            "ground_truth": r, 
            "model_prediction": pred,
            "prediction_length": len(pred.split()),
            "reference_length": len(r.split())
        } 
        for p, r, pred in zip(all_prompts, all_references, all_predictions)
    ]

    final_results = {
        "summary_metrics": summary_metrics,
        "sample_results": sample_results
    }
    
    print("\n--- Evaluation Metrics Summary ---")
    print(f"📊 Total Samples Evaluated: {len(all_predictions)}")
    
    print(f"\n🔍 ROUGE Scores:")
    if 'error' in rouge_scores:
        print(f"  - ❌ ROUGE calculation failed: {rouge_scores['error']}")
    else:
        for metric, score in rouge_scores.items():
            print(f"  - {metric.upper()}: {score:.4f}")
    
    print(f"\n🎯 BERTScore:")
    if bert_scores_result:
        print(f"  - Language Detected: {bert_scores_result.get('language_detected', 'unknown')}")
        print(f"  - Language Confidence: {bert_scores_result.get('language_confidence', 'unknown')}")
        print(f"  - Model Used: {bert_scores_result.get('model_used', 'unknown')}")
        print(f"  - F1 Average: {bert_scores_result.get('f1_avg', 0):.4f}")
        print(f"  - Precision Average: {bert_scores_result.get('precision_avg', 0):.4f}")
        print(f"  - Recall Average: {bert_scores_result.get('recall_avg', 0):.4f}")
        if 'valid_samples' in bert_scores_result:
            print(f"  - Valid Samples: {bert_scores_result['valid_samples']}/{bert_scores_result.get('total_samples', 'unknown')}")
        if 'fallback_used' in bert_scores_result:
            print(f"  - ⚠️ Fallback model used")
        if 'error' in bert_scores_result:
            print(f"  - ❌ Error: {bert_scores_result['error']}")
    else:
        print("  - ❌ BERTScore calculation completely failed")
    
    if bleu_scores and 'bleu' in bleu_scores:
        print(f"\n📝 BLEU Score: {bleu_scores['bleu']:.4f}")
        if 'error' in bleu_scores:
            print(f"  - ⚠️ Error: {bleu_scores['error']}")
    else:
        print(f"\n📝 BLEU Score: Failed to compute")
    
    print(f"\n📏 Length Analysis:")
    print(f"  - Average Prediction Length: {pred_length_stats['mean']:.1f} words")
    print(f"  - Average Reference Length: {ref_length_stats['mean']:.1f} words")
    print(f"  - Length Ratio (pred/ref): {length_ratio:.2f}")
    print(f"  - Prediction Length Range: {pred_length_stats['min']}-{pred_length_stats['max']} words")
    print(f"  - Reference Length Range: {ref_length_stats['min']}-{ref_length_stats['max']} words")
    
    print(f"\n✅ Match Analysis:")
    print(f"  - Exact Match Ratio: {exact_match_ratio:.4f} ({exact_matches}/{len(all_predictions)})")
    print(f"  - Normalized Match Ratio: {normalized_exact_match_ratio:.4f} ({normalized_exact_matches}/{len(all_predictions)})")
    
    print(f"\n🔍 Quality Analysis:")
    print(f"  - Empty Predictions: {empty_prediction_ratio:.4f} ({empty_predictions}/{len(all_predictions)})")
    print(f"  - Error Predictions: {error_prediction_ratio:.4f} ({error_predictions}/{len(all_predictions)})")
    
    if empty_prediction_ratio > 0.1:
        print(f"  - ⚠️ High empty prediction ratio detected!")
    if error_prediction_ratio > 0.05:
        print(f"  - ⚠️ High error prediction ratio detected!")
    
    # 출력 경로 설정 - 파이프라인 환경에서는 적절한 경로 사용
    if settings.is_pipeline_env:
        # 파이프라인 환경에서는 evaluation 결과를 vfroot에 저장 (다른 태스크에서 접근 가능)
        output_path = settings.evaluation_output_path / args.output_path
    else:
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