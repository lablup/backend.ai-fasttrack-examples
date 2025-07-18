import os
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, Gemma3nForConditionalGeneration

def parse_args():
    parser = argparse.ArgumentParser(description="Model Loader")
    parser.add_argument('--model_id', type=str, default=os.getenv('MODEL_ID'),
                        help='ID of the model to load from Hugging Face Hub')
    
    return parser.parse_args()

def load_model(model_id):
    """
    Load a model from Hugging Face Hub.
    """
    try:
        if 'gemma-3' in model_id.lower():
            model = Gemma3nForConditionalGeneration.from_pretrained(model_id, device_map = "auto", torch_dtype=torch.bfloat16, token=os.getenv('HF_TOKEN'))
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map = "auto", torch_dtype=torch.bfloat16, token=os.getenv('HF_TOKEN'))
        return model
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        return None

def load_processor_and_tokenizer(model_id):
    """
    Load processor and extract tokenizer for both multimodal and text-only models.
    
    Returns:
        tuple: (processor, tokenizer) where tokenizer is always accessible
    """
    try:
        print(f"Loading processor for model: {model_id}")
        processor = AutoProcessor.from_pretrained(model_id, token=os.getenv('HF_TOKEN'))
        
        # 멀티모달 vs 텍스트 전용 모델 구분
        try:
            # 멀티모달 모델인 경우 processor.tokenizer 속성이 존재
            tokenizer = processor.tokenizer
            print("✅ Multimodal model detected. Extracted tokenizer from processor.")
            return processor, tokenizer
        except AttributeError:
            # 텍스트 전용 모델인 경우 processor 자체가 tokenizer
            print("✅ Text-only model detected. Using processor as tokenizer.")
            return processor, processor
            
    except Exception as e:
        print(f"Error loading processor for model {model_id}: {e}")
        return None, None

def load_processor(model_id):
    """
    Load a processor from Hugging Face Hub.
    """
    try:
        processor = AutoProcessor.from_pretrained(model_id, token=os.getenv('HF_TOKEN'))
        return processor
    except Exception as e:
        print(f"Error loading processor for model {model_id}: {e}")
        return None

class ModelLoader:
    def __init__(self, model_id):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.tokenizer = None

        if self.model_id:
            self.model = load_model(self.model_id)
            self.processor, self.tokenizer = load_processor_and_tokenizer(self.model_id)

            if not self.model or not self.processor or not self.tokenizer:
                print(f"Failed to load model, processor, or tokenizer for {self.model_id}")
            else:
                # 기본 토크나이저 설정
                self.tokenizer.padding_side = "left"
                print(f"✅ Successfully loaded model and tokenizer for {self.model_id}")
        else:
            print("Model ID is not provided. Please set the MODEL_ID environment variable.")