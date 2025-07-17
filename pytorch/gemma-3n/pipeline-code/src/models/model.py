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
        if 'gemma' in model_id.lower():
            model = Gemma3nForConditionalGeneration.from_pretrained(model_id, device_map = "auto", torch_dtype=torch.bfloat16, token=os.getenv('HF_TOKEN'))
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map = "auto", torch_dtype=torch.bfloat16, token=os.getenv('HF_TOKEN'))
        return model
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        return None

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

        if self.model_id:
            self.model = load_model(self.model_id)
            self.processor = load_processor(self.model_id)

            if not self.model or not self.processor:
                print(f"Failed to load model or processor for {self.model_id}")
        else:
            print("Model ID is not provided. Please set the MODEL_ID environment variable.")