import os
import argparse
from pathlib import Path
import torch
import yaml
import importlib
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer, AutoModelForImageTextToText

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

def load_model(model_id, model_class=None, loading_params=None):
    """
    VLM 모델을 로드합니다. 설정된 클래스 또는 AutoModelForImageTextToText을 사용합니다.
    """
    if model_class is None:
        model_class = AutoModelForImageTextToText
    
    if loading_params is None:
        loading_params = {}
    
    try:
        print(f"Loading VLM model: {model_id} using {model_class.__name__}")
        
        # 기본 로딩 파라미터 설정
        default_params = {
            'device_map': "auto",
            'torch_dtype': torch.bfloat16,
            'token': os.getenv('HF_TOKEN')
        }
        
        # YAML 설정에서 torch_dtype 문자열을 실제 타입으로 변환
        if 'torch_dtype' in loading_params:
            dtype_str = loading_params['torch_dtype']
            if dtype_str == 'torch.bfloat16':
                loading_params['torch_dtype'] = torch.bfloat16
            elif dtype_str == 'torch.float16':
                loading_params['torch_dtype'] = torch.float16
            elif dtype_str == 'torch.float32':
                loading_params['torch_dtype'] = torch.float32
        
        # 파라미터 병합 (YAML 설정이 기본값을 덮어씀)
        final_params = {**default_params, **loading_params}
        
        model = model_class.from_pretrained(model_id, **final_params)
        print(f"✅ Successfully loaded VLM model: {model_id}")
        return model
        
    except Exception as e:
        print(f"❌ Error loading VLM model {model_id}: {e}")
        return None

def load_processor_and_tokenizer(model_id, processor_class=None, processor_params=None):
    """
    VLM 프로세서와 토크나이저를 로드합니다.
    설정된 클래스 또는 AutoProcessor를 사용하며, VLM 모델에 최적화되어 있습니다.
    
    Returns:
        tuple: (processor, tokenizer) where tokenizer is always accessible
    """
    if processor_class is None:
        processor_class = AutoProcessor
        
    if processor_params is None:
        processor_params = {}
    
    # 기본 프로세서 파라미터
    default_params = {
        'token': os.getenv('HF_TOKEN')
    }
    
    # 파라미터 병합 (processor_params가 기본값을 덮어씀)
    final_params = {**default_params, **processor_params}
    
    # 1차 시도: 설정된 Processor 클래스
    try:
        print(f"Loading VLM processor for model: {model_id} using {processor_class.__name__}")
        processor = processor_class.from_pretrained(model_id, **final_params)
        
        # VLM 모델의 경우 processor.tokenizer 속성이 존재
        try:
            tokenizer = processor.tokenizer
            print("✅ VLM model detected. Extracted tokenizer from processor.")
            
            # 비디오 프로세서 관련 정보 제공 (경고 억제 없이)
            if hasattr(processor, 'video_processor') or 'video' in str(type(processor)).lower():
                print("📹 Video processor detected.")
                print("ℹ️  Note: Video processor deprecation warnings are normal and handled automatically.")
                print("   Files are auto-renamed from preprocessor.json to video_preprocessor.json when saved.")
            
            return processor, tokenizer
        except AttributeError:
            # 일부 VLM 모델에서는 processor 자체가 tokenizer 기능을 포함
            print("✅ VLM processor with integrated tokenizer detected.")
            return processor, processor
            
    except Exception as e:
        print(f"⚠️ {processor_class.__name__} loading failed: {e}")
        print("🔄 Falling back to AutoProcessor...")
        
        # 2차 시도: AutoProcessor (fallback)
        try:
            processor = AutoProcessor.from_pretrained(model_id, **final_params)
            
            try:
                tokenizer = processor.tokenizer
                print("✅ AutoProcessor fallback successful. Extracted tokenizer.")
                return processor, tokenizer
            except AttributeError:
                print("✅ AutoProcessor fallback successful. Using processor as tokenizer.")
                return processor, processor
                
        except Exception as processor_error:
            print(f"⚠️ AutoProcessor fallback failed: {processor_error}")
            print("🔄 Trying AutoTokenizer as last resort...")
            
            # 3차 시도: AutoTokenizer (최후의 수단)
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.getenv('HF_TOKEN'))
                print("✅ AutoTokenizer loaded successfully (no processor available).")
                return None, tokenizer
                
            except Exception as tokenizer_error:
                print(f"❌ All loading attempts failed. AutoTokenizer error: {tokenizer_error}")
                return None, None

class ModelLoader:
    def __init__(self, model_id, vlm_config_path='vlm_model_config.yaml'):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.vlm_config = None

        if self.model_id:
            # VLM 설정 로드
            self.vlm_config = load_vlm_config(vlm_config_path)
            
            # 모델별 클래스 설정 가져오기
            class_config = get_model_classes(self.model_id, self.vlm_config)
            
            # 모델 로드
            self.model = load_model(
                self.model_id, 
                class_config['model_class'],
                class_config['loading_params']
            )
            
            # 프로세서와 토크나이저 로드
            self.processor, self.tokenizer = load_processor_and_tokenizer(
                self.model_id,
                class_config['processor_class'],
                class_config['processor_params']
            )

            if not self.tokenizer:
                print(f"❌ Failed to load tokenizer for VLM model {self.model_id}")
            elif not self.model:
                print(f"❌ Failed to load VLM model {self.model_id}")
            else:
                # VLM 토크나이저 기본 설정
                if hasattr(self.tokenizer, 'padding_side'):
                    self.tokenizer.padding_side = "left"
                print(f"✅ Successfully loaded VLM model and processor for {self.model_id}")
                
                # processor가 없어도 tokenizer가 있으면 경고만 출력
                if not self.processor:
                    print("⚠️ Processor not available for this VLM model, using tokenizer only")
        else:
            print("❌ Model ID is not provided. Please set the MODEL_ID environment variable.")