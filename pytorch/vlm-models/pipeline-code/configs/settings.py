# settings.py

import os
import shutil
from pathlib import Path
from typing import Optional # Optional 임포트
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # Backend.AI 파이프라인 환경 감지
    is_pipeline_env: bool = os.getenv('BACKENDAI_PIPELINE_JOB_ID', None) is not None
    
    # 기본 경로 설정
    base_path: Path = Path(__file__).parent.parent
    
    # 모델 및 데이터셋 설정
    model_id: str = os.getenv('MODEL_ID', '')
    dataset: str = os.getenv('DATASET', '')

    save_dataset_path: Optional[Path] = None
    save_dataset_path_raw: Optional[Path] = None
    save_dataset_path_preprocessed: Optional[Path] = None
    save_dataset_path_formatted: Optional[Path] = None
    save_model_path: Optional[Path] = None
    deployment_model_path: Optional[Path] = None  # LoRA 가중치가 병합된 배포용 모델 경로
    evaluation_output_path: Optional[Path] = None
    logging_dir: Optional[Path] = None
    config_path: Optional[Path] = None
    
    # Pipeline-specific paths
    pipeline_input_path: Optional[Path] = None
    pipeline_output_path: Optional[Path] = None
    pipeline_vfroot_path: Optional[Path] = None
    
    # VLM 특화 설정 (CLI 인자를 주지 않았을 때의 기본값)
    vlm_model_config: str = os.getenv('VLM_MODEL_CONFIG', 'vlm_model_config.yaml')
    vlm_collator_config: str = os.getenv('VLM_COLLATOR_CONFIG', 'vlm_collator_config.yaml')
    
    # .env 파일이나 환경 변수에 있는 키들을 필드로 명시적으로 추가합니다.
    hf_token: Optional[str] = os.getenv('HF_TOKEN', None)
    wandb_api_key: Optional[str] = os.getenv('WANDB_API_KEY', None)

    def copy_readonly_to_writable(self, readonly_path: Path, task_name: str) -> Path:
        """
        읽기 전용 폴더를 쓰기 가능한 임시 폴더로 복사하는 함수
        
        Args:
            readonly_path: 읽기 전용 소스 경로
            task_name: 현재 작업 이름 (폴더명으로 사용)
            
        Returns:
            Path: 복사된 쓰기 가능한 폴더 경로
        """
        if not self.is_pipeline_env:
            return readonly_path
            
        # 임시 폴더 경로 생성
        temp_base = self.pipeline_vfroot_path / 'tmp'
        temp_path = temp_base / task_name
        
        if not readonly_path.exists():
            print(f"⚠️ Source path does not exist: {readonly_path}")
            # 빈 디렉토리 생성하여 반환 (후속 처리에서 적절히 에러 처리됨)
            temp_path.mkdir(parents=True, exist_ok=True)
            return temp_path
        
        print(f"📁 Copying {readonly_path} to {temp_path}")
        try:
            # 기존 폴더가 있으면 제거하고 새로 복사
            if temp_path.exists():
                print(f"🗑️ Removing existing temp folder: {temp_path}")
                shutil.rmtree(temp_path)
            
            # shutil.copytree로 한 번에 전체 디렉토리 트리 복사
            shutil.copytree(
                src=readonly_path,
                dst=temp_path,
                symlinks=False,  # 심볼릭 링크를 실제 파일로 복사
                ignore_dangling_symlinks=True,  # 깨진 심볼릭 링크 무시
                dirs_exist_ok=False  # 대상 디렉토리가 이미 존재하면 에러 (이미 제거했으므로 문제없음)
            )
            
            print(f"✅ Successfully copied dataset to writable location: {temp_path}")
            return temp_path
            
        except Exception as e:
            print(f"❌ Error copying dataset: {e}")
            print(f"🔄 Falling back to original readonly path: {readonly_path}")
            # 복사에 실패하면 원본 경로 반환 (에러는 후속 처리에서 발생할 것)
            return readonly_path

    def model_post_init(self, __context):
        # Backend.AI 파이프라인 환경인지 확인
        if self.is_pipeline_env:
            # 파이프라인 환경에서의 경로 설정
            self.pipeline_input_path = Path('/pipeline/input1')
            self.pipeline_output_path = Path('/pipeline/outputs')
            self.pipeline_vfroot_path = Path('/pipeline/vfroot')
            
            # 파이프라인 환경에서의 데이터셋 경로 설정
            # download_dataset.py -> output: /pipeline/outputs/
            self.save_dataset_path_raw = self.pipeline_output_path
            
            # preprocess_dataset.py -> input: /pipeline/input1, output: /pipeline/outputs
            self.save_dataset_path_preprocessed = self.pipeline_output_path
            
            # format_dataset.py -> input: /pipeline/input1, output: /pipeline/outputs  
            self.save_dataset_path_formatted = self.pipeline_output_path
            
            # 설정 파일들은 /pipeline/vfroot/configs에서 읽기
            self.config_path = self.pipeline_vfroot_path / 'configs'
            
            # 모델 및 결과 저장 경로는 /pipeline/vfroot 하위에 설정
            self.save_model_path = self.pipeline_vfroot_path / 'results' / 'models'
            self.deployment_model_path = self.pipeline_vfroot_path / 'results' / 'deployment_model'
            self.evaluation_output_path = self.pipeline_vfroot_path / 'results' / 'evaluation'
            self.logging_dir = self.pipeline_vfroot_path / 'logs'
            
            # 기본 데이터셋 경로도 설정 (호환성 유지)
            self.save_dataset_path = self.pipeline_vfroot_path / 'dataset'
            
        else:
            # 로컬 환경에서의 기존 경로 설정 유지
            self.save_dataset_path: Path = self.base_path / os.getenv('SAVE_DATASET_PATH', 'dataset')
            self.save_dataset_path_raw: Path = self.save_dataset_path / "raw"
            self.save_dataset_path_preprocessed: Path = self.save_dataset_path / "preprocessed"
            self.save_dataset_path_formatted: Path = self.save_dataset_path / "formatted"
            self.save_model_path: Path = self.base_path / os.getenv('SAVE_MODEL_PATH', 'results/models')
            self.deployment_model_path: Path = self.base_path / os.getenv('DEPLOYMENT_MODEL_PATH', 'results/deployment_model')
            self.evaluation_output_path: Path = self.base_path / os.getenv('EVALUATION_OUTPUT_PATH', 'results/evaluation')
            self.logging_dir: Path = self.base_path / os.getenv('LOGGING_DIR', 'logs')
            self.config_path: Path = self.base_path / 'configs'

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

settings = Settings()