# settings.py

import os
from pathlib import Path
from typing import Optional # Optional 임포트
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # 기본 경로 설정
    base_path: Path = Path(__file__).parent.parent
    
    # 모델 및 데이터셋 설정
    model_id: str = os.getenv('MODEL_ID', 'google/gemma-3n-e2b-it')
    dataset: str = os.getenv('DATASET', 'TheFinAI/Fino1_Reasoning_Path_FinQA')

    save_dataset_path: Optional[Path] = None
    save_model_path: Optional[Path] = None
    merged_model_path: Optional[Path] = None
    evaluation_output_path: Optional[Path] = None
    logging_dir: Optional[Path] = None
    config_path: Optional[Path] = None
    
    # .env 파일이나 환경 변수에 있는 키들을 필드로 명시적으로 추가합니다.
    hf_token: Optional[str] = os.getenv('HF_TOKEN', None)
    wandb_api_key: Optional[str] = os.getenv('WANDB_API_KEY', None)

    def model_post_init(self, __context):
    # 경로 설정
        self.save_dataset_path: Path = self.base_path / os.getenv('SAVE_DATASET_PATH', 'dataset')
        self.save_model_path: Path = self.base_path / os.getenv('SAVE_MODEL_PATH', 'results/models')
        self.merged_model_path: Path = self.base_path / os.getenv('MERGED_MODEL_PATH', 'results/merged_model')
        self.evaluation_output_path: Path = self.base_path / os.getenv('EVALUATION_OUTPUT_PATH', 'results/evaluation')
        self.logging_dir: Path = self.base_path / os.getenv('LOGGING_DIR', 'logs')
        self.config_path: Path = self.base_path / 'configs'

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

settings = Settings()