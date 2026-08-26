import os
import argparse
from pathlib import Path
import torch
import yaml
import importlib
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, AutoModelForImageTextToText

def gpu_supports_bf16() -> bool:
    """True only when the GPU has native bfloat16, i.e. Ampere (SM 8.0) or newer.

    torch.cuda.is_bf16_supported() cannot be used for this: it also returns True on
    Volta and Turing by counting emulation, which is roughly an order of magnitude
    slower than fp16. Because it returns True, Hugging Face's own bf16 guard passes
    and the user gets no warning at all -- just a much slower run.
    """
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 8


def parse_args():
    parser = argparse.ArgumentParser(description="VLM Model Loader")
    parser.add_argument('--model_id', type=str, default=os.getenv('MODEL_ID'),
                        help='ID of the model to load from Hugging Face Hub')
    parser.add_argument('--vlm_config', type=str, default='vlm_model_config.yaml',
                        help='Path to VLM model configuration file')
    
    return parser.parse_args()

def load_vlm_config(config_path: str) -> dict:
    """VLM 모델 설정 파일을 로드합니다."""
    from configs.settings import settings
    
    config_file = settings.config_path / config_path
    print(f"Loading VLM model config from: {config_file}")
    
    if not config_file.exists():
        print(f"⚠️ VLM config file not found: {config_file}")
        print("🔄 Using default AutoModel classes")
        return None
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def get_model_classes(model_id: str, vlm_config: dict):
    """모델 ID를 기반으로 적절한 모델 클래스들을 반환합니다."""
    if not vlm_config:
        # 기본 VLM Auto 클래스 사용
        return {
            'model_class': AutoModelForImageTextToText,
            'processor_class': AutoProcessor,
            'loading_params': {},
            'processor_params': {}
        }
    
    # 모델별 설정 확인
    model_classes = vlm_config.get('model_classes', {})
    default_fallback = vlm_config.get('default_fallback', {})
    loading_params = vlm_config.get('loading_params', {})
    processor_params = vlm_config.get('processor_params', {})
    
    if model_id in model_classes:
        model_config = model_classes[model_id]
        print(f"✅ Found specific configuration for model: {model_id}")
    else:
        model_config = default_fallback
        print(f"⚠️ No specific configuration found for {model_id}, using default fallback")
    
    # 클래스 동적 임포트
    try:
        import_path = model_config.get('import_path', 'transformers')
        module = importlib.import_module(import_path)
        
        # model_class는 필수, processor_class는 선택사항 (기본값: AutoProcessor)
        model_class = getattr(module, model_config['model_class'])
        
        # processor_class가 지정되어 있는지 확인
        if 'processor_class' in model_config:
            processor_class = getattr(module, model_config['processor_class'])
            print(f"📦 Using specific processor class: {model_config['processor_class']}")
        else:
            processor_class = AutoProcessor
            print(f"📦 Using default AutoProcessor (no specific processor_class configured)")
        
        return {
            'model_class': model_class,
            'processor_class': processor_class,
            'loading_params': loading_params,
            'processor_params': processor_params
        }
        
    except (ImportError, AttributeError) as e:
        print(f"⚠️ Failed to import classes: {e}")
        print("🔄 Falling back to VLM Auto classes")
        return {
            'model_class': AutoModelForImageTextToText,
            'processor_class': AutoProcessor,
            'loading_params': {},
            'processor_params': {}
        }

def load_model(model_source, model_class=None, loading_params=None):
    """Load a VLM model from either a hub model id or a local path.

    model_source: str | Path - hub id or filesystem directory containing config.json
    model_class: class used for from_pretrained
    loading_params: optional dict of extra kwargs (may include torch_dtype as string)
    """
    if model_class is None:
        model_class = AutoModelForImageTextToText
    if loading_params is None:
        loading_params = {}

    model_source = str(model_source)
    try:
        print(f"Loading VLM model from '{model_source}' using {model_class.__name__}")
        default_params = {
            'device_map': "auto",
            'torch_dtype': torch.bfloat16,
            'token': os.getenv('HF_TOKEN')
        }
        # Convert dtype strings to actual torch dtypes
        if 'torch_dtype' in loading_params and isinstance(loading_params['torch_dtype'], str):
            dtype_str = loading_params['torch_dtype']
            mapping = {
                'torch.bfloat16': torch.bfloat16,
                'torch.float16': torch.float16,
                'torch.float32': torch.float32,
            }
            loading_params['torch_dtype'] = mapping.get(dtype_str, loading_params['torch_dtype'])
        final_params = {**default_params, **loading_params}
        if final_params.get('torch_dtype') is torch.bfloat16 and not gpu_supports_bf16():
            cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
            print(f"⚠️ bfloat16 requested but this GPU (compute capability {cap}) has no native "
                  f"bfloat16 support. Loading in float16 instead.")
            final_params['torch_dtype'] = torch.float16
        model = model_class.from_pretrained(model_source, **final_params)
        print(f"✅ Successfully loaded VLM model from: {model_source}")
        return model
    except Exception as e:
        print(f"❌ Error loading VLM model from {model_source}: {e}")
        return None

def load_processor_and_tokenizer(model_source, processor_class=None, processor_params=None):
    """Load processor & tokenizer from a model id or local path, with fallback chain.

    model_source: hub id or local path; for local merged models we first try that path.
    Returns (processor, tokenizer) where either may be None.
    """
    if processor_class is None:
        processor_class = AutoProcessor
    if processor_params is None:
        processor_params = {}

    model_source = str(model_source)
    default_params = {'token': os.getenv('HF_TOKEN')}
    final_params = {**default_params, **processor_params}

    # First attempt with provided processor_class
    try:
        print(f"Loading processor from '{model_source}' using {processor_class.__name__}")
        processor = processor_class.from_pretrained(model_source, **final_params)
        try:
            tokenizer = processor.tokenizer
        except AttributeError:
            tokenizer = processor
        print("✅ Processor loaded successfully")
        return processor, tokenizer
    except Exception as e:
        print(f"⚠️ Primary processor load failed: {e}")

    # Fallback to AutoProcessor
    try:
        print("🔄 Falling back to AutoProcessor")
        processor = AutoProcessor.from_pretrained(model_source, **final_params)
        try:
            tokenizer = processor.tokenizer
        except AttributeError:
            tokenizer = processor
        print("✅ AutoProcessor fallback successful")
        return processor, tokenizer
    except Exception as e:
        print(f"⚠️ AutoProcessor fallback failed: {e}")

    # Final fallback AutoTokenizer
    try:
        print("🔄 Trying AutoTokenizer as last resort")
        tokenizer = AutoTokenizer.from_pretrained(model_source, token=os.getenv('HF_TOKEN'))
        print("✅ AutoTokenizer loaded (no processor)")
        return None, tokenizer
    except Exception as e:
        print(f"❌ All processor/tokenizer loading attempts failed: {e}")
        return None, None

class ModelLoader:
    def __init__(self, model_id, vlm_config_path='vlm_model_config.yaml', model_load_path: str | Path | None = None, processor_load_path: str | Path | None = None):
        """Unified loader for VLM models.

        model_id: hub identifier used for config mapping & default loading.
        model_load_path: optional local directory (merged fine-tuned model). If provided, model weights
                         are loaded from this path but configuration mapping still uses model_id.
        processor_load_path: optional separate local directory for processor/tokenizer.
        """
        self.model_id = model_id
        self.model_load_path = Path(model_load_path) if model_load_path else None
        self.processor_load_path = Path(processor_load_path) if processor_load_path else None
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.vlm_config = None

        if not self.model_id:
            print("❌ Model ID is not provided. Please set the MODEL_ID environment variable.")
            return

        # Load config & class mapping based on hub model_id
        self.vlm_config = load_vlm_config(vlm_config_path)
        class_config = get_model_classes(self.model_id, self.vlm_config)

        # Decide sources
        model_source = self.model_load_path if (self.model_load_path and (self.model_load_path / 'config.json').exists()) else self.model_id
        processor_source = self.processor_load_path if (self.processor_load_path and (self.processor_load_path / 'preprocessor_config.json').exists() or (self.processor_load_path and (self.processor_load_path / 'tokenizer_config.json').exists())) else model_source

        # Load model
        self.model = load_model(
            model_source,
            class_config['model_class'],
            class_config['loading_params']
        )

        # Load processor/tokenizer
        self.processor, self.tokenizer = load_processor_and_tokenizer(
            processor_source,
            class_config['processor_class'],
            class_config['processor_params']
        )

        if not self.tokenizer:
            print(f"❌ Failed to load tokenizer for VLM model {self.model_id}")
        elif not self.model:
            print(f"❌ Failed to load VLM model from {model_source}")
        else:
            if hasattr(self.tokenizer, 'padding_side'):
                self.tokenizer.padding_side = "left"
            print(f"✅ Loaded model (source='{model_source}') for id '{self.model_id}'")
            if not self.processor:
                print("⚠️ Processor missing; tokenizer-only mode")