#!/usr/bin/env python3
"""
VLM Data Collator
VLM 모델을 위한 사용자 정의 데이터 콜레이터
이미지와 텍스트를 함께 처리하며, 설정 파일을 통해 커스터마이징 가능
"""

import os
import re
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
        
        # 설정값들을 쉽게 접근할 수 있도록 저장 (특수 토큰 설정 전에 먼저 초기화)
        self.dataset_columns = self.config.get('dataset_columns', {})
        self.message_format = self.config.get('message_format', {})
        self.data_processing = self.config.get('data_processing', {})  # 새로 추가
        self.image_processing = self.config.get('image_processing', {})
        self.text_processing = self.config.get('text_processing', {})
        self.label_masking = self.config.get('label_masking', {})
        self.batch_processing = self.config.get('batch_processing', {})
        self.special_tokens_config = self.config.get('special_tokens', {})
        self.video_processing = self.config.get('video_processing', {})  # 비디오 처리 설정 추가
        
        # 특수 토큰 ID 미리 계산 (설정 초기화 후)
        self._setup_special_tokens()
    
    def _setup_special_tokens(self):
        """토크나이저의 모든 특수 토큰을 자동으로 감지하고 설정합니다."""
        print("🔍 Auto-detecting special tokens from tokenizer...")
        self.special_token_ids = {}
        self.ignore_in_loss_ids = set()  # 손실 계산 시 무시할 토큰 ID 집합

        # tokenizer 객체 가져오기 (getattr로 통일)
        tokenizer = getattr(self.processor, 'tokenizer', self.processor)
        print(f"📊 Tokenizer type: {type(tokenizer).__name__}")
        
        # 1. special_tokens_map의 모든 토큰 처리 (additional_special_tokens 포함)
        self._process_all_special_tokens(tokenizer)
        
        # 2. apply_chat_template 호환성 검증
        self._verify_chat_template_compatibility(tokenizer)
        
        # 3. Manual config 처리 (고급 사용자용 override)
        self._process_manual_config_if_enabled(tokenizer)
        
        print(f"✅ Auto-detection complete.")
        print(f"   📋 Total special tokens found: {len(self.special_token_ids)}")
        print(f"   🚫 Tokens to ignore in loss: {len(self.ignore_in_loss_ids)}")

    def _process_all_special_tokens(self, tokenizer):
        """special_tokens_map의 모든 특수 토큰과 additional_special_tokens를 모두 처리합니다."""
        print("  🔧 Processing all special tokens from special_tokens_map...")
        
        special_tokens_map = getattr(tokenizer, 'special_tokens_map', {})
        print(f"special_tokens_map length : {len(special_tokens_map)}")

        # 1. special_tokens_map 처리
        for token_attr, token_str in special_tokens_map.items():
            try:
                # additional_special_tokens는 별도로 처리하므로 건너뛰기
                if token_attr == 'additional_special_tokens':
                    continue
                    
                # 토큰 ID 가져오기 (getattr로 통일)
                token_id = getattr(tokenizer, f'{token_attr}_id', None)
                
                # token_id가 리스트인 경우 처리 (실제 문제 원인)
                if isinstance(token_id, list):
                    if len(token_id) > 0:
                        # 리스트인 경우 모든 ID를 처리
                        print(f"    📝 Token ID '{token_attr}_id' is a list: {token_id}, adding all IDs")
                        for idx, single_id in enumerate(token_id):
                            # token_attr 그대로 사용 (clean_name 사용 안함)
                            if len(token_id) > 1:
                                # 여러 ID가 있는 경우 인덱스 추가
                                token_name = f"{token_attr}_{idx}"
                            else:
                                token_name = token_attr
                            
                            self.special_token_ids[token_name] = single_id
                            self.ignore_in_loss_ids.add(single_id)
                            print(f"    ✅ {token_name}: '{token_str}' -> ID: {single_id}")
                    else:
                        print(f"    ⚠️ Token ID '{token_attr}_id' is an empty list, skipping")
                        continue
                else:
                    # 단일 ID인 경우
                    if token_id is not None:
                        # token_attr 그대로 사용 (clean_name 사용 안함)
                        self.special_token_ids[token_attr] = token_id
                        
                        # 모든 특수 토큰은 기본적으로 손실 계산에서 제외
                        self.ignore_in_loss_ids.add(token_id)
                        
                        print(f"    ✅ {token_attr}: '{token_str}' -> ID: {token_id}")
                    else:
                        print(f"    ⚠️ No ID found for token '{token_attr}': '{token_str}'")
                    
            except Exception as e:
                print(f"    ❌ Error processing token '{token_attr}': {e}")
                print(f"    🔍 Token value type: {type(token_str)}, Token ID type: {type(token_id)}")
                print(f"    🔍 Token attr: '{token_attr}', Token str: {token_str}, Token ID: {token_id}")
                continue

        # 2. additional_special_tokens 처리 (통합)
        print("  🎯 Processing additional special tokens...")
        additional_tokens = getattr(tokenizer, 'additional_special_tokens', [])
        additional_token_ids = getattr(tokenizer, 'additional_special_tokens_ids', [])
        
        if additional_tokens:
            print(f"    📝 Found {len(additional_tokens)} additional special tokens")
            
            for i, token_str in enumerate(additional_tokens):
                if i < len(additional_token_ids):
                    token_id = additional_token_ids[i]
                    
                    # 간단한 토큰 이름 생성
                    clean_token = token_str.replace('<', '').replace('>', '').replace('|', '_')
                    token_name = f"special_{clean_token}"
                    
                    self.special_token_ids[token_name] = token_id
                    
                    # 모든 추가 특수 토큰도 손실 계산에서 제외
                    self.ignore_in_loss_ids.add(token_id)
                    
                    print(f"    ✅ {token_name}: '{token_str}' -> ID: {token_id}")

    def _verify_chat_template_compatibility(self, tokenizer):
        """apply_chat_template과의 호환성을 검증합니다."""
        print("  🔍 Verifying chat template compatibility...")
        
        # 테스트 메시지 생성
        test_messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Test message"},
                {"type": "image"},
                {"type": "text", "text": "What do you see?"}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "I see an image."}
            ]}
        ]
        
        try:
            test_text = self.processor.apply_chat_template(
                test_messages, 
                tokenize=False,
                add_generation_prompt=False
            )
            
            # 생성된 텍스트에서 특수 토큰 확인
            self._check_template_tokens(test_text, tokenizer)
            print("    ✅ Chat template compatibility verified")
                
        except Exception as e:
            print(f"    ⚠️ Chat template test failed: {e}")

    def _check_template_tokens(self, template_text, tokenizer):
        """템플릿 텍스트에 포함된 특수 토큰들을 확인합니다."""
        print(f"    📄 Template text preview: {template_text[:100]}...")
        
        # 템플릿에서 특수 토큰 패턴 찾기
        special_token_pattern = r'<[^>]+>'
        found_tokens = re.findall(special_token_pattern, template_text)
        
        if found_tokens:
            print(f"    🎯 Found template tokens: {found_tokens}")
            
            # 발견된 토큰들이 우리가 감지한 토큰 목록에 있는지 확인
            for token in found_tokens:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != tokenizer.unk_token_id:
                    # 새로운 토큰 발견시 추가
                    token_name = f"template_token_{token.replace('<', '').replace('>', '')}"
                    if token_id not in self.special_token_ids.values():
                        self.special_token_ids[token_name] = token_id
                        self.ignore_in_loss_ids.add(token_id)
                        print(f"    🆕 Added template token: '{token}' -> ID: {token_id}")

    def _process_manual_config_if_enabled(self, tokenizer):
        """YAML 설정에서 manual_tokens가 활성화된 경우 처리합니다."""
        manual_tokens = self.special_tokens_config.get('manual_tokens', {})
        
        if not manual_tokens.get('enabled', False):
            print("  ⏭️ Manual token configuration disabled (using auto-detection only)")
            return
        
        print("  🔧 Processing manual token configuration...")
        manual_token_list = manual_tokens.get('tokens', [])
        
        if not manual_token_list:
            print("    ⚠️ Manual tokens enabled but no tokens specified")
            return
        
        override_count = 0
        new_count = 0
        
        for token_config in manual_token_list:
            if not isinstance(token_config, dict):
                print(f"    ❌ Invalid token config (must be dict): {token_config}")
                continue
                
            token_name = token_config.get('name')
            token_text = token_config.get('token')
            ignore_in_loss = token_config.get('ignore_in_loss', True)
            
            if not token_name or not token_text:
                print(f"    ❌ Invalid token config (missing name/token): {token_config}")
                continue
            
            # 토큰 ID 계산
            token_id = tokenizer.convert_tokens_to_ids(token_text)
            if token_id == tokenizer.unk_token_id:
                print(f"    ⚠️ Unknown token '{token_text}' for '{token_name}' - skipping")
                continue
            
            # 기존 토큰 override 또는 새 토큰 추가
            if token_name in self.special_token_ids:
                old_id = self.special_token_ids[token_name]
                print(f"    🔄 Override '{token_name}': {old_id} -> {token_id}")
                override_count += 1
                
                # 기존 ID 제거
                if old_id in self.ignore_in_loss_ids:
                    self.ignore_in_loss_ids.remove(old_id)
            else:
                print(f"    ➕ Add manual token '{token_name}': {token_id}")
                new_count += 1
            
            # 새 설정 적용
            self.special_token_ids[token_name] = token_id
            if ignore_in_loss:
                self.ignore_in_loss_ids.add(token_id)
        
        if override_count > 0 or new_count > 0:
            print(f"    ✅ Manual config processed: {override_count} overrides, {new_count} new tokens")
        else:
            print("    ℹ️ No valid manual tokens processed")

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
            convert_method = getattr(image, 'convert', None)
            if convert_method:  # PIL-like object
                try:
                    image = convert_method('RGB')
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

            
            # 2. 이미지 처리 (image_data 플래그가 활성화된 경우)
            if process_image and not processed_visuals:  # 비디오가 처리되지 않은 경우에만
                image_col = self.dataset_columns.get('image_column', 'image')
                if image_col in example and example[image_col] is not None:
                    processed_image = self._process_image(example[image_col])
                    if processed_image is not None:
                        processed_visuals.append(processed_image)
            
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
        
        # 6. 레이블 생성 및 마스킹 (일반화된 버전)
        labels = batch["input_ids"].clone()
        ignore_index = self.label_masking.get('ignore_index', -100)
        
        # 미리 계산된 ignore_in_loss_ids 집합을 사용하여 한 번에 마스킹
        if self.ignore_in_loss_ids:
            # boolean 마스크 생성: labels 텐서의 각 요소가 무시할 ID 집합에 속하는지 확인
            mask = torch.isin(labels, torch.tensor(list(self.ignore_in_loss_ids), device=labels.device))
            # 마스크가 True인 위치의 값을 ignore_index로 변경
            labels[mask] = ignore_index
            print(f"🔧 Masked {torch.sum(mask).item()} tokens in loss calculation")
        
        # (선택적) 추가 마스킹 로직
        # 프롬프트 부분 마스킹이 필요한 경우 여기에 추가할 수 있습니다.
        # 예: assistant 응답 시작 전까지의 모든 토큰을 마스킹
        if self.label_masking.get('mask_input_tokens', False):
            # 이 부분은 모델별 chat template에 따라 다르게 구현될 수 있습니다.
            print("💡 Input token masking is enabled but not implemented yet.")
        
        batch["labels"] = labels
        
        return batch
    
    def _process_with_processor(self, texts: List[str], visual_data: List[List[Image.Image]]) -> Dict[str, torch.Tensor]:
        """프로세서를 사용하여 텍스트와 시각 데이터를 처리합니다."""
        
        # 빈 시각 데이터 필터링 및 평탄화
        actual_images = []
        for visuals in visual_data:
            if visuals and len(visuals) > 0:
                # 시각 데이터가 PIL Image 리스트인지 확인
                if isinstance(visuals[0], Image.Image):
                    if len(visuals) == 1:
                        # 단일 이미지: 그대로 사용
                        actual_images.append(visuals[0])
                    else:
                        # 다중 프레임 (비디오): 첫 번째 프레임만 사용 (VLM processor 호환성)
                        actual_images.append(visuals[0])
                        print(f"📹 Using first frame from {len(visuals)} video frames")
        
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
