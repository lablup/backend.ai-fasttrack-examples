#!/usr/bin/env python3
"""
VLM Data Collator
VLM 모델을 위한 사용자 정의 데이터 콜레이터
이미지와 텍스트를 함께 처리하며, 설정 파일을 통해 커스터마이징 가능
"""

import os
import yaml
import torch
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional, Union

# 비디오 처리를 위한 선택적 import
try:
    import decord
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    decord = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

def load_collator_config(config_path: str) -> dict:
    """콜레이터 설정 파일을 로드합니다."""
    from configs.settings import settings
    
    config_file = settings.config_path / config_path
    print(f"Loading VLM collator config from: {config_file}")
    
    if not config_file.exists():
        raise FileNotFoundError(f"Collator config file not found: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

class VLMDataCollator:
    """
    VLM 모델을 위한 데이터 콜레이터
    이미지와 텍스트를 함께 처리하며, 설정 파일을 통해 커스터마이징 가능
    """
    
    def __init__(self, processor: Any, config: dict):
        """
        VLMDataCollator 초기화
        
        Args:
            processor: VLM processor (tokenizer + image processor)
            config: 콜레이터 설정 딕셔너리
        """
        self.processor = processor
        self.config = config
        
        # 특수 토큰 ID 미리 계산
        self._setup_special_tokens()
        
        # 설정값들을 쉽게 접근할 수 있도록 저장
        self.dataset_columns = self.config.get('dataset_columns', {})
        self.message_format = self.config.get('message_format', {})
        self.data_processing = self.config.get('data_processing', {})  # 새로 추가
        self.image_processing = self.config.get('image_processing', {})
        self.text_processing = self.config.get('text_processing', {})
        self.label_masking = self.config.get('label_masking', {})
        self.batch_processing = self.config.get('batch_processing', {})
        self.special_tokens_config = self.config.get('special_tokens', {})
        self.video_processing = self.config.get('video_processing', {})  # 비디오 처리 설정 추가
    
    def _setup_special_tokens(self):
        """특수 토큰 ID들을 미리 설정합니다."""
        try:
            # 이미지 토큰 ID 찾기
            image_token = self.special_tokens_config.get('image_token', '<image>')
            
            if hasattr(self.processor, 'tokenizer'):
                tokenizer = self.processor.tokenizer
            else:
                tokenizer = self.processor
            
            # 이미지 토큰 ID 설정
            if hasattr(tokenizer, 'additional_special_tokens_ids') and hasattr(tokenizer, 'additional_special_tokens'):
                try:
                    token_index = tokenizer.additional_special_tokens.index(image_token)
                    self.image_token_id = tokenizer.additional_special_tokens_ids[token_index]
                except (ValueError, IndexError):
                    print(f"⚠️ Image token '{image_token}' not found in additional_special_tokens")
                    self.image_token_id = None
            else:
                # 일반 토큰으로 인코딩 시도
                try:
                    encoded = tokenizer.encode(image_token, add_special_tokens=False)
                    self.image_token_id = encoded[0] if encoded else None
                except:
                    print(f"⚠️ Could not encode image token '{image_token}'")
                    self.image_token_id = None
            
            # 패드 토큰 ID
            self.pad_token_id = tokenizer.pad_token_id
            
            print(f"✅ Special tokens setup - Image: {self.image_token_id}, Pad: {self.pad_token_id}")
            
        except Exception as e:
            print(f"⚠️ Error setting up special tokens: {e}")
            self.image_token_id = None
            self.pad_token_id = None
    
    def _process_image(self, image) -> Optional[Image.Image]:
        """이미지 전처리를 수행합니다. 파일 경로와 PIL Image 모두 지원합니다."""
        if image is None:
            return None
            
        # 1. 문자열(파일 경로)인 경우 이미지 로드
        if isinstance(image, str):
            try:
                # 절대 경로 또는 상대 경로 처리
                image_path = Path(image)
                if not image_path.is_absolute():
                    # 상대 경로인 경우 현재 작업 디렉토리를 기준으로 처리
                    image_path = Path.cwd() / image_path
                
                if not image_path.exists():
                    print(f"⚠️ Image file not found: {image_path}")
                    return None
                
                # PIL Image로 로드
                image = Image.open(image_path)
                print(f"✅ Successfully loaded image from path: {image_path}")
                
            except Exception as e:
                print(f"❌ Error loading image from path '{image}': {e}")
                return None
        
        # 2. PIL Image가 아닌 경우 변환
        elif not isinstance(image, Image.Image):
            if hasattr(image, 'convert'):  # PIL-like object
                try:
                    image = image.convert('RGB')
                except Exception as e:
                    print(f"⚠️ Error converting PIL-like object: {e}")
                    return None
            else:
                # numpy array나 다른 형식인 경우
                try:
                    import numpy as np
                    if isinstance(image, np.ndarray):
                        image = Image.fromarray(image)
                    else:
                        print(f"⚠️ Unsupported image type: {type(image)}")
                        return None
                except Exception as e:
                    print(f"⚠️ Could not convert image to PIL Image: {e}")
                    return None
        
        # 3. RGB 변환 (설정에 따라)
        if self.image_processing.get('convert_to_rgb', True):
            try:
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                print(f"⚠️ Error converting to RGB: {e}")
                return None
        
        return image
    
    def _process_video(self, video) -> Optional[List[Image.Image]]:
        """비디오를 처리하여 프레임 리스트를 반환합니다."""
        if video is None or not self.video_processing.get('enabled', False):
            return None
            
        # 1. 문자열(파일 경로)인 경우 비디오 로드
        if isinstance(video, str):
            try:
                # 절대 경로 또는 상대 경로 처리
                video_path = Path(video)
                if not video_path.is_absolute():
                    video_path = Path.cwd() / video_path
                
                if not video_path.exists():
                    print(f"⚠️ Video file not found: {video_path}")
                    return None
                
                return self._extract_video_frames(str(video_path))
                
            except Exception as e:
                print(f"❌ Error processing video from path '{video}': {e}")
                return None
        else:
            print(f"⚠️ Unsupported video type: {type(video)}")
            return None
    
    def _extract_video_frames(self, video_path: str) -> Optional[List[Image.Image]]:
        """비디오 파일에서 프레임을 추출합니다."""
        frame_config = self.video_processing.get('frame_extraction', {})
        library = frame_config.get('library', 'decord')
        num_frames = frame_config.get('num_frames', 8)
        sampling_strategy = frame_config.get('sampling_strategy', 'uniform')
        
        if library == 'decord' and DECORD_AVAILABLE:
            return self._extract_frames_with_decord(video_path, num_frames, sampling_strategy)
        elif library == 'cv2' and CV2_AVAILABLE:
            return self._extract_frames_with_cv2(video_path, num_frames, sampling_strategy)
        else:
            # Fallback to cv2 if available
            if CV2_AVAILABLE:
                print(f"⚠️ {library} not available, falling back to cv2")
                return self._extract_frames_with_cv2(video_path, num_frames, sampling_strategy)
            else:
                print(f"❌ No video processing library available (decord: {DECORD_AVAILABLE}, cv2: {CV2_AVAILABLE})")
                return None
    
    def _extract_frames_with_decord(self, video_path: str, num_frames: int, sampling_strategy: str) -> Optional[List[Image.Image]]:
        """Decord를 사용하여 비디오 프레임을 추출합니다."""
        try:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            total_frames = len(vr)
            
            if total_frames == 0:
                print(f"⚠️ Video has no frames: {video_path}")
                return None
            
            # 프레임 인덱스 계산
            if sampling_strategy == 'uniform':
                if total_frames < num_frames:
                    # 비디오가 요청된 프레임 수보다 적은 경우 모든 프레임 사용
                    indices = list(range(total_frames))
                else:
                    # 균등하게 샘플링
                    indices = torch.linspace(0, total_frames - 1, num_frames).long().tolist()
            else:
                # 기본적으로 uniform 사용
                indices = torch.linspace(0, total_frames - 1, min(num_frames, total_frames)).long().tolist()
            
            # 프레임 추출
            frames = vr.get_batch(indices).asnumpy()  # Shape: (num_frames, H, W, C)
            
            # PIL Image로 변환
            pil_frames = []
            for frame in frames:
                pil_frame = Image.fromarray(frame)
                # RGB 변환 옵션 적용
                if self.video_processing.get('file_processing', {}).get('convert_to_rgb', True):
                    if pil_frame.mode != 'RGB':
                        pil_frame = pil_frame.convert('RGB')
                pil_frames.append(pil_frame)
            
            print(f"✅ Successfully extracted {len(pil_frames)} frames from video: {video_path}")
            return pil_frames
            
        except Exception as e:
            print(f"❌ Error extracting frames with decord: {e}")
            return None
    
    def _extract_frames_with_cv2(self, video_path: str, num_frames: int, sampling_strategy: str) -> Optional[List[Image.Image]]:
        """OpenCV를 사용하여 비디오 프레임을 추출합니다."""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                print(f"⚠️ Video has no frames: {video_path}")
                cap.release()
                return None
            
            # 프레임 인덱스 계산
            if sampling_strategy == 'uniform':
                if total_frames < num_frames:
                    indices = list(range(total_frames))
                else:
                    indices = torch.linspace(0, total_frames - 1, num_frames).long().tolist()
            else:
                indices = torch.linspace(0, total_frames - 1, min(num_frames, total_frames)).long().tolist()
            
            # 프레임 추출
            pil_frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # BGR to RGB 변환
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb)
                    pil_frames.append(pil_frame)
                else:
                    print(f"⚠️ Failed to read frame at index {idx}")
            
            cap.release()
            
            if pil_frames:
                print(f"✅ Successfully extracted {len(pil_frames)} frames from video: {video_path}")
                return pil_frames
            else:
                print(f"❌ No frames could be extracted from video: {video_path}")
                return None
                
        except Exception as e:
            print(f"❌ Error extracting frames with cv2: {e}")
            return None
    
    def _format_messages(self, example: dict, is_training: bool = True) -> list:
        """예제를 메시지 형식으로 변환합니다."""
        # 컬럼명 매핑
        image_col = self.dataset_columns.get('image_column', 'image')
        question_col = self.dataset_columns.get('question_column', 'question')
        answer_col = self.dataset_columns.get('answer_column', 'answer')
        
        # 데이터 추출
        question = example.get(question_col, '')
        answer = example.get(answer_col, '') if is_training else ''
        
        # 시스템 프롬프트
        system_prompt = self.message_format.get('system_prompt', 'Answer briefly.')
        
        # 메시지 템플릿 선택
        if is_training:
            messages_template = self.message_format.get('training_messages', [])
        else:
            messages_template = self.message_format.get('evaluation_messages', [])
        
        # 템플릿에 데이터 채우기
        messages = []
        for msg_template in messages_template:
            message = {
                'role': msg_template['role'],
                'content': []
            }
            
            for content_item in msg_template['content']:
                if content_item['type'] == 'text':
                    text = content_item['text'].format(
                        system_prompt=system_prompt,
                        question=question,
                        answer=answer
                    )
                    message['content'].append({
                        'type': 'text',
                        'text': text
                    })
                elif content_item['type'] == 'image':
                    message['content'].append({
                        'type': 'image'
                    })
                else:
                    # 다른 타입들 (비디오 등) 지원
                    message['content'].append(content_item.copy())
            
            messages.append(message)
        
        return messages
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        배치 데이터를 처리하는 메인 함수 - 플래그 기반으로 이미지와 비디오 처리를 제어
        
        Args:
            examples: 배치 데이터 리스트
            
        Returns:
            Dict[str, torch.Tensor]: 모델 입력용 텐서 딕셔너리
        """
        texts = []
        visual_data = []  # 이미지 또는 비디오 프레임을 담을 리스트
        
        # 데이터 처리 플래그 확인
        process_image = self.data_processing.get('image_data', True)  # 기본값: True
        process_video = self.data_processing.get('video_data', False)  # 기본값: False
        
        # 두 플래그가 모두 활성화된 경우 확인 (processor 호환성 검사)
        if process_image and process_video:
            # 대부분의 VLM processor는 images 파라미터에 하나의 타입만 받을 수 있음
            # 여기서는 video를 우선 사용하고 경고 메시지 출력
            print("⚠️ Both image_data and video_data are enabled. Using video data as priority.")
            print("💡 Note: Most VLM processors can only handle one visual data type at a time.")
            process_image = False  # 이미지 처리 비활성화
        
        # 배치의 각 예제 처리
        for example in examples:
            processed_visuals = []
            
            # 1. 비디오 처리 (video_data 플래그가 활성화된 경우)
            if process_video:
                video_col = self.dataset_columns.get('video_column', 'video')
                if video_col in example and example[video_col] is not None:
                    video_frames = self._process_video(example[video_col])
                    if video_frames:
                        processed_visuals.extend(video_frames)
                        print(f"📹 Processed video with {len(video_frames)} frames")
            
            # 2. 이미지 처리 (image_data 플래그가 활성화된 경우)
            if process_image and not processed_visuals:  # 비디오가 처리되지 않은 경우에만
                image_col = self.dataset_columns.get('image_column', 'image')
                if image_col in example and example[image_col] is not None:
                    processed_image = self._process_image(example[image_col])
                    if processed_image is not None:
                        processed_visuals.append(processed_image)
                        print(f"🖼️ Processed single image")
            
            # 프로세서에 맞는 형태로 래핑
            if processed_visuals:
                visual_data.append(processed_visuals)
            else:
                visual_data.append([])  # 빈 리스트로 처리
            
            # 3. 메시지 형식으로 변환 (학습용)
            messages = self._format_messages(example, is_training=True)
            
            # 4. 채팅 템플릿 적용
            try:
                text = self.processor.apply_chat_template(
                    messages, 
                    add_generation_prompt=self.text_processing.get('add_generation_prompt', False),
                    tokenize=False
                )
                texts.append(text.strip())
            except Exception as e:
                print(f"⚠️ Error applying chat template: {e}")
                # fallback: 간단한 텍스트 결합
                question = example.get(self.dataset_columns.get('question_column', 'question'), '')
                answer = example.get(self.dataset_columns.get('answer_column', 'answer'), '')
                
                # 비디오/이미지 태그 추가
                if processed_visuals:
                    if len(processed_visuals) > 1:  # 비디오 (다중 프레임)
                        visual_tag = "<video>"
                    else:  # 단일 이미지
                        visual_tag = "<image>"
                    texts.append(f"{visual_tag}\nQuestion: {question}\nAnswer: {answer}")
                else:
                    texts.append(f"Question: {question}\nAnswer: {answer}")
        
        # 5. 프로세서로 배치 처리
        try:
            # VLM 모델에 따라 다른 처리 방식 적용
            batch = self._process_with_processor(texts, visual_data)
        except Exception as e:
            print(f"❌ Error processing batch with processor: {e}")
            # fallback: 텍스트만 처리
            batch = self.processor(
                text=texts,
                return_tensors=self.batch_processing.get('return_tensors', 'pt'),
                padding=self.text_processing.get('padding', True),
                truncation=self.text_processing.get('truncation', True),
                max_length=self.text_processing.get('max_length', 2048)
            )
        
        # 6. 레이블 생성 및 마스킹
        labels = batch["input_ids"].clone()
        ignore_index = self.label_masking.get('ignore_index', -100)
        
        # 패딩 토큰 마스킹
        if self.label_masking.get('mask_pad_token', True) and self.pad_token_id is not None:
            labels[labels == self.pad_token_id] = ignore_index
        
        # 이미지/비디오 토큰 마스킹
        if self.label_masking.get('mask_image_token', True) and self.image_token_id is not None:
            labels[labels == self.image_token_id] = ignore_index
        
        # 비디오 토큰 마스킹 (있는 경우)
        video_token = self.special_tokens_config.get('video_token', '<video>')
        if hasattr(self, 'video_token_id') and self.video_token_id is not None:
            labels[labels == self.video_token_id] = ignore_index
        
        batch["labels"] = labels
        
        return batch
    
    def _process_with_processor(self, texts: List[str], visual_data: List[List[Image.Image]]) -> Dict[str, torch.Tensor]:
        """프로세서를 사용하여 텍스트와 시각 데이터를 처리합니다."""
        
        # 빈 시각 데이터 필터링
        non_empty_visuals = []
        for visuals in visual_data:
            if visuals:
                # 비디오인 경우 (다중 프레임) 첫 번째 프레임만 사용 (대부분의 VLM processor 호환성)
                # 실제 비디오를 지원하는 모델의 경우 여기서 다르게 처리할 수 있음
                non_empty_visuals.append(visuals[0] if len(visuals) == 1 else visuals)
            else:
                non_empty_visuals.append(None)
        
        # None이 아닌 이미지들만 추출
        actual_images = [img for img in non_empty_visuals if img is not None]
        
        if actual_images:
            # 이미지가 있는 경우
            try:
                batch = self.processor(
                    text=texts,
                    images=actual_images,
                    return_tensors=self.batch_processing.get('return_tensors', 'pt'),
                    padding=self.text_processing.get('padding', True),
                    truncation=self.text_processing.get('truncation', True),
                    max_length=self.text_processing.get('max_length', 2048)
                )
                return batch
            except Exception as e:
                print(f"⚠️ Error processing with images, trying text-only: {e}")
        
        # 이미지가 없거나 처리 실패 시 텍스트만 처리
        batch = self.processor(
            text=texts,
            return_tensors=self.batch_processing.get('return_tensors', 'pt'),
            padding=self.text_processing.get('padding', True),
            truncation=self.text_processing.get('truncation', True),
            max_length=self.text_processing.get('max_length', 2048)
        )
        
        return batch

def create_vlm_collator(processor, config_path: str = 'vlm_collator_config.yaml'):
    """
    VLM 데이터 콜레이터를 생성합니다.
    
    Args:
        processor: VLM 프로세서
        config_path: 콜레이터 설정 파일 경로
        
    Returns:
        VLMDataCollator: 설정된 데이터 콜레이터
    """
    config = load_collator_config(config_path)
    return VLMDataCollator(processor=processor, config=config)
