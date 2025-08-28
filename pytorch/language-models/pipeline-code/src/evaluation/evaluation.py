#!/usr/bin/env python3
"""Simplified LLM evaluation script.

Aligns with the improved VLM evaluation: concise args, merged/adaptor-aware loading,
generation via model.generate, and metrics ROUGE/BLEU/BERTScore with fallbacks.
Outputs first 10 prompt/reference/prediction triplets.
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
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, GenerationConfig

from configs.settings import settings


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a (base or fine-tuned) LLM on a saved dataset")
    p.add_argument('--model_name_or_path', type=str, default=os.getenv('MODEL_ID'), required=True,
                   help='Base model id or local path')
    p.add_argument('--output_path', type=str, default='evaluation_results.json', help='Where to write JSON results')
    p.add_argument('--max_samples', type=int, default=None, help='Limit number of samples (debug)')
    p.add_argument('--batch_size', type=int, default=int(os.getenv('EVAL_BATCH_SIZE', 8)), help='Generation batch size')
    p.add_argument('--max_new_tokens', type=int, default=512, help='Max new tokens to generate')
    p.add_argument('--use_adapter', action='store_true', help='Load fine-tuned adapter / merged model if available')
    return p.parse_args()


def _load_tokenizer(source: str):
    """Load tokenizer from a model id or local path without loading a model."""
    try:
        proc = AutoProcessor.from_pretrained(source, token=os.getenv('HF_TOKEN'))
        tok = getattr(proc, 'tokenizer', proc)
        return tok
    except Exception:
        pass
    return AutoTokenizer.from_pretrained(source, token=os.getenv('HF_TOKEN'))


def load_model_for_evaluation(model_name_or_path: str, use_finetuned: bool = False):
    """Load base or fine-tuned model (merged preferred, else adapter on base)."""
    if use_finetuned:
        dep = settings.deployment_model_path
        if dep and isinstance(dep, Path) and dep.exists() and (dep / 'config.json').exists():
            print(f"Detected merged deployment model at: {dep}")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    str(dep), device_map='auto', torch_dtype=torch.bfloat16, local_files_only=True
                )
                return model
            except Exception as e:
                print(f"⚠️ Failed to load merged model, falling back to adapter/base: {e}")

    # Base model
    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map='auto', torch_dtype=torch.bfloat16, token=os.getenv('HF_TOKEN')
    )

    if use_finetuned:
        # Try applying adapter from results/models
        try:
            from peft import PeftModel  # optional
            ap = settings.save_model_path
            if ap and ap.exists():
                adapter_candidates = [
                    ap / 'adapter_model.safetensors',
                    ap / 'adapter_config.json',
                    ap / 'pytorch_lora_weights.bin',
                ]
                if any(p.exists() for p in adapter_candidates):
                    print(f"Applying PEFT adapter from {ap}")
                    return PeftModel.from_pretrained(base, str(ap))
        except Exception as e:  # noqa: BLE001 - best-effort
            print(f"⚠️ Adapter load failed; using base model: {e}")
    return base


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
    # BERTScore
    try:
        bs_metric = evaluate.load('bertscore')
        preferred = os.getenv('BERTSCORE_MODEL', 'BAAI/bge-m3')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        used_model = None

        def _run(model_type: str):
            return bs_metric.compute(
                predictions=preds,
                references=refs,
                model_type=model_type,
                device=device,
                rescale_with_baseline=False,
            )

        try:
            bs = _run(preferred)
            used_model = preferred
        except Exception:
            for alt in ('roberta-large', 'microsoft/deberta-large-mnli'):
                try:
                    bs = _run(alt)
                    used_model = alt
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError(f"All BERTScore model attempts failed (preferred={preferred})")

        if 'f1' in bs and bs['f1']:
            metrics['bertscore_f1'] = float(sum(bs['f1']) / len(bs['f1']))
            metrics['bertscore_model_used'] = used_model
    except Exception as e:
        metrics['bertscore_error'] = str(e)
    return metrics


def _pick_column(cols: List[str], candidates: List[str]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def main():  # noqa: C901
    args = parse_args()
    print("=== LLM Evaluation (Simplified) ===")

    # Resolve dataset path
    if settings.is_pipeline_env:
        readonly = settings.pipeline_input_path
        dataset_path = settings.copy_readonly_to_writable(readonly, 'evaluation')
    else:
        dataset_path = settings.save_dataset_path_formatted

    if not dataset_path or not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    print(f"Loading dataset from: {dataset_path}")
    ds_dict = load_from_disk(str(dataset_path))

    for cand in ['test', 'validation', 'val', 'eval', 'train']:
        if cand in ds_dict:
            split = cand
            break
    else:
        raise RuntimeError('No usable split found')

    eval_ds: Dataset = ds_dict[split]
    print(f"Using split '{split}' with {len(eval_ds)} samples")

    if args.max_samples and 0 < args.max_samples < len(eval_ds):
        eval_ds = eval_ds.select(range(args.max_samples))
        print(f"Trimmed to {len(eval_ds)} samples")

    # Tokenizer source: prefer merged tokenizer when using adapter
    if args.use_adapter and settings.deployment_model_path and (settings.deployment_model_path / 'tokenizer_config.json').exists():
        tok_source = str(settings.deployment_model_path)
    else:
        tok_source = args.model_name_or_path
    tokenizer = _load_tokenizer(tok_source)

    model = load_model_for_evaluation(args.model_name_or_path, use_finetuned=args.use_adapter)
    model.eval()

    # Generation config
    try:
        base_cfg = getattr(model, 'generation_config', None)
        gen_cfg = base_cfg.clone() if base_cfg is not None else GenerationConfig.from_model_config(getattr(model, 'config', None))
    except Exception:
        gen_cfg = GenerationConfig()
    gen_cfg.do_sample = False
    gen_cfg.max_new_tokens = int(args.max_new_tokens)
    try:
        if getattr(tokenizer, 'eos_token_id', None) is not None:
            gen_cfg.eos_token_id = tokenizer.eos_token_id
        if getattr(tokenizer, 'pad_token_id', None) is not None:
            gen_cfg.pad_token_id = tokenizer.pad_token_id
        elif getattr(tokenizer, 'eos_token_id', None) is not None:
            gen_cfg.pad_token_id = tokenizer.eos_token_id
        model.generation_config = gen_cfg
    except Exception:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Column mapping
    cols = list(eval_ds.column_names)
    prompt_col = _pick_column(cols, ['prompt', 'input', 'instruction', 'question', 'text'])
    ref_col = _pick_column(cols, ['reference', 'answer', 'output', 'label', 'text_target'])
    if not prompt_col or not ref_col:
        raise RuntimeError(f"Missing required columns. Found={cols}")

    batch_size = max(1, int(args.batch_size))
    preds: List[str] = []
    refs: List[str] = []
    prompts: List[str] = []

    print('Generating predictions...')
    for start in tqdm(range(0, len(eval_ds), batch_size), desc='Batches'):
        batch = [eval_ds[i] for i in range(start, min(start + batch_size, len(eval_ds)))]
        batch_prompts = [str(ex.get(prompt_col, '') or '') for ex in batch]
        batch_refs = [str(ex.get(ref_col, '') or '') for ex in batch]

        # Filter empty
        f_prompts: List[str] = []
        f_refs: List[str] = []
        for p, r in zip(batch_prompts, batch_refs):
            ps, rs = p.strip(), r.strip()
            if ps and rs:
                f_prompts.append(ps)
                f_refs.append(rs)
        if not f_prompts:
            continue

        enc = tokenizer(
            f_prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            # Leave max_length to model defaults to avoid truncating prompts aggressively
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out_ids = model.generate(**enc, generation_config=gen_cfg)
        # Slice to new tokens (decoder-only models return prompt+new)
        try:
            new_tokens = out_ids[:, enc['input_ids'].shape[1]:]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        except Exception:
            decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

        preds.extend([d.strip() if d.strip() else "[EMPTY_GENERATION]" for d in decoded])
        refs.extend(f_refs)
        prompts.extend(f_prompts)

    n = min(len(preds), len(refs))
    preds, refs, prompts = preds[:n], refs[:n], prompts[:n]

    print('Computing metrics (ROUGE, BLEU, BERTScore)...')
    metrics = compute_metrics(preds, refs)

    empty_predictions = sum(1 for p in preds if not p.strip())
    error_markers = {"[PROCESSING_ERROR]", "[EMPTY_GENERATION]"}
    error_predictions = sum(1 for p in preds if p.strip() in error_markers)
    total = len(preds) if preds else 1
    metrics['quality_analysis'] = {
        'empty_prediction_count': empty_predictions,
        'empty_prediction_ratio': empty_predictions / total,
        'error_prediction_count': error_predictions,
        'error_prediction_ratio': error_predictions / total,
    }

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

    out_path = Path(settings.evaluation_output_path / args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'Saved evaluation results to {out_path}')


if __name__ == '__main__':
    main()