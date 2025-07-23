import os
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Model Loader")
    parser.add_argument('--model_id', type=str, default=os.getenv('MODEL_ID'),
                        help='ID of the model to load from Hugging Face Hub')
    
    return parser.parse_args()

def load_model(model_id):
    """
    Load a model from Hugging Face Hub using AutoModelForCausalLM for universal compatibility.
    """
    try:
        print(f"Loading model: {model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            token=os.getenv('HF_TOKEN')
        )
        print(f"✅ Successfully loaded model: {model_id}")
        return model
    except Exception as e:
        print(f"❌ Error loading model {model_id}: {e}")
        return None

def load_processor_and_tokenizer(model_id):
    """
    Load processor and extract tokenizer for both multimodal and text-only models.
    Supports fallback to AutoTokenizer for deployment models without preprocessor_config.json
    
    Returns:
        tuple: (processor, tokenizer) where tokenizer is always accessible
    """
    # 1차 시도: AutoProcessor (멀티모달 모델 지원)
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
        print(f"⚠️ AutoProcessor loading failed: {e}")
        print("🔄 Falling back to AutoTokenizer...")
        
        # 2차 시도: AutoTokenizer (deployment 모델 등에서 사용)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.getenv('HF_TOKEN'))
            print("✅ Successfully loaded tokenizer using AutoTokenizer fallback.")
            return None, tokenizer  # processor는 None, tokenizer만 반환
            
        except Exception as tokenizer_error:
            print(f"❌ AutoTokenizer fallback also failed: {tokenizer_error}")
            return None, None

class ModelLoader:
    def __init__(self, model_id):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.tokenizer = None

        if self.model_id:
            self.model = load_model(self.model_id)
            self.processor, self.tokenizer = load_processor_and_tokenizer(self.model_id)

            if not self.tokenizer:
                print(f"❌ Failed to load tokenizer for {self.model_id}")
            elif not self.model:
                print(f"❌ Failed to load model for {self.model_id}")
            else:
                # 기본 토크나이저 설정
                self.tokenizer.padding_side = "left"
                print(f"✅ Successfully loaded model and tokenizer for {self.model_id}")
                
                # processor가 없어도 tokenizer가 있으면 성공으로 간주
                if not self.processor:
                    print("ℹ️ Processor not available, but tokenizer loaded successfully (AutoTokenizer fallback)")
        else:
            print("❌ Model ID is not provided. Please set the MODEL_ID environment variable.")