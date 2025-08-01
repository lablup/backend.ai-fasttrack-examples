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
from typing import List, Dict, Any, Optional

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
        self.image_processing = self.config.get('image_processing', {})
        self.text_processing = self.config.get('text_processing', {})
        self.label_masking = self.config.get('label_masking', {})
        self.batch_processing = self.config.get('batch_processing', {})
        self.special_tokens_config = self.config.get('special_tokens', {})
    
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
    
    def _process_image(self, image) -> Image.Image:
        """이미지 전처리를 수행합니다."""
        if image is None:
            return None
            
        # PIL Image가 아닌 경우 변환
        if not isinstance(image, Image.Image):
            if hasattr(image, 'convert'):  # PIL-like object
                image = image.convert('RGB')
            else:
                # numpy array나 다른 형식인 경우
                try:
                    image = Image.fromarray(image)
                except:
                    print("⚠️ Could not convert image to PIL Image")
                    return None
        
        # RGB 변환
        if self.image_processing.get('convert_to_rgb', True):
            if image.mode != 'RGB':
                image = image.convert('RGB')
        
        return image
    
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
        배치 데이터를 처리하는 메인 함수
        
        Args:
            examples: 배치 데이터 리스트
            
        Returns:
            Dict[str, torch.Tensor]: 모델 입력용 텐서 딕셔너리
        """
        texts = []
        images = []
        
        # 배치의 각 예제 처리
        for example in examples:
            # 이미지 처리
            image_col = self.dataset_columns.get('image_column', 'image')
            image = example.get(image_col)
            processed_image = self._process_image(image)
            
            if processed_image is not None:
                images.append([processed_image])  # 리스트로 감싸기 (processor 요구사항)
            else:
                # 이미지가 없는 경우 빈 리스트
                images.append([])
            
            # 메시지 형식으로 변환 (학습용)
            messages = self._format_messages(example, is_training=True)
            
            # 채팅 템플릿 적용
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
                texts.append(f"Question: {question}\nAnswer: {answer}")
        
        # 프로세서로 배치 처리
        try:
            batch = self.processor(
                text=texts,
                images=images,
                return_tensors=self.batch_processing.get('return_tensors', 'pt'),
                padding=self.text_processing.get('padding', True),
                truncation=self.text_processing.get('truncation', True),
                max_length=self.text_processing.get('max_length', 2048)
            )
        except Exception as e:
            print(f"⚠️ Error processing batch with images: {e}")
            # 이미지 없이 텍스트만 처리 시도
            try:
                batch = self.processor(
                    text=texts,
                    return_tensors=self.batch_processing.get('return_tensors', 'pt'),
                    padding=self.text_processing.get('padding', True),
                    truncation=self.text_processing.get('truncation', True),
                    max_length=self.text_processing.get('max_length', 2048)
                )
            except Exception as e2:
                print(f"❌ Error processing text-only batch: {e2}")
                raise e2
        
        # 레이블 생성 및 마스킹
        labels = batch["input_ids"].clone()
        ignore_index = self.label_masking.get('ignore_index', -100)
        
        # 패딩 토큰 마스킹
        if self.label_masking.get('mask_pad_token', True) and self.pad_token_id is not None:
            labels[labels == self.pad_token_id] = ignore_index
        
        # 이미지 토큰 마스킹
        if self.label_masking.get('mask_image_token', True) and self.image_token_id is not None:
            labels[labels == self.image_token_id] = ignore_index
        
        batch["labels"] = labels
        
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
