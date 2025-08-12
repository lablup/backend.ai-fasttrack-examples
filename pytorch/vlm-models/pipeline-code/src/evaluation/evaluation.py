#!/usr/bin/env python3
"""Simplified VLM evaluation script.

Keeps original model/dataset loading semantics (adapter vs base) while
reducing metrics to ROUGE, BLEU, BERTScore (fixed model BAAI/bge-m3).
Outputs first 10 sample prompt/reference/prediction triplets.
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from datasets import load_from_disk, Dataset
import evaluate

from src.models.model import ModelLoader
from src.data.collate_fn import create_vlm_collator
from configs.settings import settings


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a (base or fine-tuned) VLM model on a saved dataset")
    p.add_argument('--model_name_or_path', type=str, default=os.getenv('MODEL_ID'), required=True,
                   help='Base model id or local path')
    p.add_argument('--output_path', type=str, default='evaluation_results.json', help='Where to write JSON results')
    p.add_argument('--max_samples', type=int, default=None, help='Limit number of samples (debug)')
    p.add_argument('--batch_size', type=int, default=os.getenv('EVAL_BATCH_SIZE ',8), help='Generation batch size')
    p.add_argument('--max_new_tokens', type=int, default=512, help='Max new tokens to generate')
    p.add_argument('--use_adapter', action='store_true', help='Load fine-tuned adapter / merged model if available')
    return p.parse_args()


def load_model_for_evaluation(model_name_or_path: str, use_finetuned: bool = False):
    """Load base or fine-tuned model.

    If a merged deployment model directory exists we load weights from that path while still
    using model_name_or_path (hub id) for config mapping inside ModelLoader.
    Otherwise optionally apply PEFT adapter.
    """
    merged_path = None
    if use_finetuned:
        dep = settings.deployment_model_path
        if dep and isinstance(dep, Path) and dep.exists() and (dep / 'config.json').exists():
            merged_path = dep
            print(f"Detected merged deployment model at: {merged_path}")

    # Instantiate ModelLoader with optional model_load_path (will still use hub id for class mapping)
    ml = ModelLoader(model_name_or_path, model_load_path=merged_path)  # processor path defaults to model path
    if not ml.model or not ml.processor:
        print("❌ Failed to load base/merged model. Aborting adapter attempt.")
        return None, None

    # If adapter requested but no merged model; try applying adapter on top of base
    if use_finetuned and merged_path is None:
        adapter_path = settings.save_model_path
        if adapter_path and isinstance(adapter_path, Path) and adapter_path.exists():
            try:
                from peft import PeftModel
                print(f"Applying PEFT adapter from {adapter_path}")
                adapted = PeftModel.from_pretrained(ml.model, str(adapter_path))
                return adapted, ml.processor
            except Exception as e:
                print(f"⚠️ Adapter load failed, using base model: {e}")
    return ml.model, ml.processor


def compute_metrics(preds: List[str], refs: List[str]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    # ROUGE
    try:
        rouge = evaluate.load('rouge')
        r = rouge.compute(predictions=preds, references=refs)
        metrics['rouge1'] = r.get('rouge1')
        metrics['rougeL'] = r.get('rougeL')
    except Exception as e:
        metrics['rouge_error'] = str(e)
    # BLEU (ensure references shape [[ref], ...])
    try:
        bleu = evaluate.load('bleu')
        refs_for_bleu = [[r] for r in refs]
        b = bleu.compute(predictions=preds, references=refs_for_bleu)
        metrics['bleu'] = b.get('bleu')
    except Exception as e:
        metrics['bleu_error'] = str(e)
    # BERTScore (fixed model)
    try:
        bs_metric = evaluate.load('bertscore')
        bs = bs_metric.compute(predictions=preds, references=refs, model_type='BAAI/bge-m3')
        if 'f1' in bs and bs['f1']:
            metrics['bertscore_f1'] = float(sum(bs['f1']) / len(bs['f1']))
    except Exception as e:
        metrics['bertscore_error'] = str(e)
    return metrics


def main():  # noqa: C901 (kept simple & linear intentionally)
    args = parse_args()
    print("=== VLM Evaluation (Simplified) ===")

    # Dataset path resolution (unchanged semantics)
    if settings.is_pipeline_env:
        readonly = settings.pipeline_input_path
        dataset_path = settings.copy_readonly_to_writable(readonly, 'evaluation')
    else:
        dataset_path = settings.save_dataset_path_raw

    if not dataset_path or not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    print(f"Loading dataset from: {dataset_path}")
    ds_dict = load_from_disk(str(dataset_path))

    # Split priority
    for cand in ['test', 'validation', 'val', 'eval', 'train']:
        if cand in ds_dict:
            split = cand
            break
    else:  # pragma: no cover - defensive
        raise RuntimeError('No usable split found')

    eval_ds: Dataset = ds_dict[split]
    print(f"Using split '{split}' with {len(eval_ds)} samples")

    if args.max_samples and 0 < args.max_samples < len(eval_ds):
        eval_ds = eval_ds.select(range(args.max_samples))
        print(f"Trimmed to {len(eval_ds)} samples")

    model, processor = load_model_for_evaluation(args.model_name_or_path, use_finetuned=args.use_adapter)
    if model is None:
        raise RuntimeError('Model loading failed')
    model.eval()

    collator = create_vlm_collator(processor, config_path='vlm_collator_config.yaml')
    ans_col = collator.dataset_columns.get('answer_column', 'answer')
    tokenizer = getattr(processor, 'tokenizer', processor)

    device = getattr(model, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device)

    batch_size = max(1, args.batch_size)
    preds: List[str] = []
    refs: List[str] = []
    prompts: List[str] = []

    print('Generating predictions...')
    for start in tqdm(range(0, len(eval_ds), batch_size), desc='Batches'):
        batch_examples = [eval_ds[i] for i in range(start, min(start + batch_size, len(eval_ds)))]
        for ex in batch_examples:
            refs.append(ex.get(ans_col, ''))
        batch_inputs = collator(batch_examples, is_training=False)
        try:
            prompts.extend(tokenizer.batch_decode(batch_inputs['input_ids'], skip_special_tokens=True))
        except Exception:
            prompts.extend([''] * len(batch_examples))
        batch_inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch_inputs.items()}
        with torch.no_grad():
            gen_allowed = {'input_ids', 'attention_mask', 'pixel_values', 'images', 'pixel_values_videos'}
            gen_in = {k: v for k, v in batch_inputs.items() if k in gen_allowed}
            out_ids = model.generate(**gen_in, max_new_tokens=args.max_new_tokens, do_sample=False)
            decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            preds.extend(decoded)

    # Align lengths
    n = min(len(preds), len(refs))
    preds, refs, prompts = preds[:n], refs[:n], prompts[:n]

    print('Computing metrics (ROUGE, BLEU, BERTScore)...')
    metrics = compute_metrics(preds, refs)

    # Quality analysis (reintroduce empty/error prediction statistics)
    empty_predictions = sum(1 for p in preds if not p.strip())
    # Define error markers (could extend later)
    error_markers = {"[PROCESSING_ERROR]", "[EMPTY_GENERATION]"}
    error_predictions = sum(1 for p in preds if p.strip() in error_markers)
    total = len(preds) if preds else 1
    quality_analysis = {
        'empty_prediction_count': empty_predictions,
        'empty_prediction_ratio': empty_predictions / total,
        'error_prediction_count': error_predictions,
        'error_prediction_ratio': error_predictions / total,
    }
    metrics['quality_analysis'] = quality_analysis

    sample_k = 10
    results = {
        'model_name_or_path': args.model_name_or_path,
        'use_adapter': args.use_adapter,
        'split': split,
        'num_samples': n,
        'metrics': metrics,
        'sample_results': {
            'prompts': prompts[:sample_k],
            'references': refs[:sample_k],
            'predictions': preds[:sample_k]
        }
    }

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'Saved evaluation results to {out_path}')


if __name__ == '__main__':
    main()
