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
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import defaultdict

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
        self.video_processing = self.config.get('video_processing', {})  # 비디오 처리 설정 추가
        
        # 특수 토큰 ID 미리 계산 (설정 초기화 후)
        self._setup_special_tokens()
    
    def _setup_special_tokens(self):
        """토크나이저의 특수 토큰을 간단하게 감지하여 손실에서 무시합니다."""
        print("🔍 Detecting special tokens via tokenizer.all_special_ids...")
        self.special_token_ids = {}
        self.ignore_in_loss_ids = set()

        tokenizer = getattr(self.processor, 'tokenizer', self.processor)
        all_ids = set(getattr(tokenizer, 'all_special_ids', []) or [])

        # 간단화: 모든 special id를 무시 대상에 추가
        self.ignore_in_loss_ids.update(all_ids)
        print(f"special token ids : {self.ignore_in_loss_ids}  will be ignored in loss.")
        # 선택적으로 이름 매핑(로깅용) 채우기
        try:
            special_map = getattr(tokenizer, 'special_tokens_map', {}) or {}
            print(f"special_tokens_map: {special_map}")
            for k, tok in special_map.items():
                if k == 'additional_special_tokens':
                    # list 처리
                    add_ids = getattr(tokenizer, 'additional_special_tokens_ids', []) or []
                    for i, sid in enumerate(add_ids):
                        self.special_token_ids[f"additional_special_tokens_{i}"] = sid
                else:
                    sid = getattr(tokenizer, f"{k}_id", None)
                    if isinstance(sid, list):
                        for i, s in enumerate(sid):
                            self.special_token_ids[f"{k}_{i}"] = s
                    elif sid is not None:
                        self.special_token_ids[k] = sid
            print(f"special_token_ids detected: {self.special_token_ids}")
        except Exception:
            pass

        # EOS must stay in the loss. It is a special token, but it is also the only
        # signal that teaches the model to end its turn -- mask it and generation runs
        # to max_new_tokens forever. Padding and the visual placeholders stay masked.
        eos_id = getattr(tokenizer, 'eos_token_id', None)
        if eos_id is not None and eos_id in self.ignore_in_loss_ids:
            self.ignore_in_loss_ids.discard(eos_id)
            print(f"↩️ Keeping eos_token_id={eos_id} in the loss so the model learns to stop.")

        print(f"✅ Special tokens collected: ignore_in_loss_ids={len(self.ignore_in_loss_ids)}")

    

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

        # 4. 리사이즈 (max_size가 설정된 경우에만 적용)
        image = self._resize_image(image)

        return image

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """image_processing.max_size / resize_mode 설정에 따라 이미지를 리사이즈합니다.

        max_size가 null(기본값)이면 원본 크기를 그대로 유지합니다.
        - keep_aspect: 가로세로 비율을 유지한 채 긴 변을 max_size로 축소
        - stretch: 비율을 무시하고 max_size x max_size로 변형
        - crop: 짧은 변 기준으로 중앙을 정사각형으로 잘라낸 뒤 max_size x max_size로 축소
        """
        max_size = self.image_processing.get('max_size')
        if not max_size:
            return image

        try:
            max_size = int(max_size)
        except (TypeError, ValueError):
            print(f"⚠️ Invalid image_processing.max_size: {max_size!r}. Skipping resize.")
            return image

        mode = self.image_processing.get('resize_mode', 'keep_aspect')
        try:
            if mode == 'keep_aspect':
                if max(image.size) > max_size:
                    image = image.copy()
                    image.thumbnail((max_size, max_size), Image.LANCZOS)
            elif mode == 'stretch':
                image = image.resize((max_size, max_size), Image.LANCZOS)
            elif mode == 'crop':
                width, height = image.size
                side = min(width, height)
                left = (width - side) // 2
                top = (height - side) // 2
                image = image.crop((left, top, left + side, top + side))
                image = image.resize((max_size, max_size), Image.LANCZOS)
            else:
                print(f"⚠️ Unknown image_processing.resize_mode: {mode!r}. Skipping resize.")
        except Exception as e:
            print(f"⚠️ Error resizing image ({mode}, max_size={max_size}): {e}")

        return image
    
    def _process_video(self, video) -> Optional[List[Tuple[Image.Image, Optional[float]]]]:
        """비디오를 처리하여 (프레임, 타임스탬프) 리스트를 반환합니다.
        Returns: List of tuples (PIL.Image, timestamp_seconds or None)
        """
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
    
    def _extract_video_frames(self, video_path: str) -> Optional[List[Tuple[Image.Image, Optional[float]]]]:
        """비디오 파일에서 프레임과 타임스탬프를 추출합니다.
        Returns: List of tuples (PIL.Image, timestamp_seconds or None)
        """
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
    
    def _extract_frames_with_decord(self, video_path: str, num_frames: int, sampling_strategy: str) -> Optional[List[Tuple[Image.Image, Optional[float]]]]:
        """Decord를 사용하여 비디오 프레임과 타임스탬프를 추출합니다."""
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

            # 타임스탬프 계산 시도
            timestamps: List[Optional[float]] = []
            for idx in indices:
                ts_val: Optional[float] = None
                try:
                    ts_val = float(vr.get_frame_timestamp(int(idx)))
                except Exception:
                    try:
                        fps = float(vr.get_avg_fps())
                        ts_val = (float(idx) / fps) if fps and fps > 0 else None
                    except Exception:
                        ts_val = None
                timestamps.append(ts_val)

            # PIL Image로 변환
            pil_frames_with_ts: List[Tuple[Image.Image, Optional[float]]] = []
            for frame, ts in zip(frames, timestamps):
                pil_frame = Image.fromarray(frame)
                # RGB 변환 옵션 적용
                if self.video_processing.get('file_processing', {}).get('convert_to_rgb', True):
                    if pil_frame.mode != 'RGB':
                        pil_frame = pil_frame.convert('RGB')
                pil_frames_with_ts.append((pil_frame, ts))
            
            print(f"✅ Successfully extracted {len(pil_frames_with_ts)} frames from video: {video_path}")
            return pil_frames_with_ts
            
        except Exception as e:
            print(f"❌ Error extracting frames with decord: {e}")
            return None
    
    def _extract_frames_with_cv2(self, video_path: str, num_frames: int, sampling_strategy: str) -> Optional[List[Tuple[Image.Image, Optional[float]]]]:
        """OpenCV를 사용하여 비디오 프레임과 타임스탬프를 추출합니다."""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) else 0.0
            
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
            pil_frames_with_ts: List[Tuple[Image.Image, Optional[float]]] = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # BGR to RGB 변환
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb)
                    ts_val: Optional[float] = (float(idx) / fps) if fps and fps > 0 else None
                    pil_frames_with_ts.append((pil_frame, ts_val))
                else:
                    print(f"⚠️ Failed to read frame at index {idx}")
            
            cap.release()
            
            if pil_frames_with_ts:
                print(f"✅ Successfully extracted {len(pil_frames_with_ts)} frames from video: {video_path}")
                return pil_frames_with_ts
            else:
                print(f"❌ No frames could be extracted from video: {video_path}")
                return None
                
        except Exception as e:
            print(f"❌ Error extracting frames with cv2: {e}")
            return None
    
    def _format_messages(self, example: dict, is_training: bool = True) -> list:
        """메시지 텍스트에 사용할 포맷 변수를 dataset_columns 전체로 일반화합니다.

        - vlm_collator_config.yaml 의 dataset_columns 딕셔너리의 key들을 변수명으로 사용합니다.
        - 각 key에 매핑된 실제 컬럼명(값)으로부터 example에서 데이터를 일괄 추출하여 포맷 컨텍스트를 구성합니다.
        - key가 *_column 으로 끝나면, 접미사를 제거한 별칭(예: question_column -> question)도 함께 제공합니다.
        - system_prompt도 포맷 컨텍스트에 포함합니다.
        """
        # 1) 포맷 컨텍스트 구성
        format_ctx: Dict[str, Any] = {}
        # 시스템 프롬프트 포함
        format_ctx['system_prompt'] = self.message_format.get('system_prompt', 'Answer briefly.')

        # dataset_columns의 모든 키를 변수로 노출하고 example에서 값 추출
        for key, col_name in (self.dataset_columns or {}).items():
            if col_name is None:
                value = ''
            else:
                value = example.get(col_name, '')
            format_ctx[key] = value
            # *_column 별칭 제공 (message 템플릿에서 간결한 이름을 사용할 수 있도록)
            if key.endswith('_column'):
                base_key = key[:-7]  # remove '_column'
                if base_key and base_key not in format_ctx:
                    format_ctx[base_key] = value

        # 2) 메시지 템플릿 선택
        messages_template = (
            self.message_format.get('training_messages', []) if is_training
            else self.message_format.get('evaluation_messages', [])
        )

        # 3) 템플릿 렌더링 (이미지 placeholder는 무시; 이후 비주얼 주입 단계에서 처리)
        messages: List[Dict[str, Any]] = []
        safe_ctx = defaultdict(lambda: '', format_ctx)
        for msg_template in messages_template:
            message = {'role': msg_template['role'], 'content': []}
            for content_item in msg_template.get('content', []) or []:
                if content_item.get('type') == 'text':
                    try:
                        text = content_item['text'].format_map(safe_ctx)
                    except Exception:
                        # 형식 오류 시 원문 유지 + 최소한의 안전 장치
                        text = str(content_item.get('text', ''))
                    message['content'].append({'type': 'text', 'text': text})
                elif content_item.get('type') == 'image':
                    # 고정 이미지 placeholder는 여기서 무시 (실제 이미지는 _build_messages_with_visuals에서 주입)
                    continue
                else:
                    message['content'].append(content_item.copy())
            messages.append(message)

        return messages
    
    def _build_messages_with_visuals(self, base_messages: list, visuals_count: int, timestamps: Optional[List[Optional[float]]]) -> list:
        """Inject visuals for user messages: visuals first, then all user text.
        """
        updated: List[Dict[str, Any]] = []
        for msg in base_messages:
            role = msg.get('role')
            contents = msg.get('content', []) or []

            # Non-user: copy as-is
            if role != 'user' or visuals_count <= 0:
                updated.append({'role': role, 'content': [c.copy() if isinstance(c, dict) else c for c in contents]})
                continue

            # Collect text segments (ignore any pre-existing image placeholders)
            text_segments: List[Dict[str, Any]] = [
                {'type': 'text', 'text': c.get('text', '')}
                for c in contents
                if isinstance(c, dict) and c.get('type') == 'text'
            ]

            # Build new content: visuals first
            new_content: List[Dict[str, Any]] = []
            for i in range(visuals_count):
                ts = None
                if timestamps is not None and i < len(timestamps):
                    ts = timestamps[i]

                label = f"Frame {i+1} (t={ts:.2f}s):" if ts is not None else f"Image {i+1}:"
                new_content.append({"type" : "text", "text" : label})
                new_content.append({"type": "image"})

            # Then append all original user text segments (order preserved)
            new_content.extend(text_segments)

            updated.append({'role': 'user', 'content': new_content})

        return updated

    def __call__(self, examples: List[Dict[str, Any]], is_training: bool = True) -> Dict[str, torch.Tensor]:
        """
        배치 데이터를 처리하는 메인 함수 - 플래그 기반으로 이미지와 비디오 처리를 제어
        
        Args:
            examples: 배치 데이터 리스트
            is_training: 학습 모드 여부 (True: 학습, False: 평가)
            
        Returns:
            Dict[str, torch.Tensor]: 모델 입력용 텐서 딕셔너리
        """

        texts: List[str] = []
        visual_data: List[List[Image.Image]] = []  # 각 샘플별 이미지 리스트

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
            visuals_images: List[Image.Image] = []
            frame_timestamps: Optional[List[Optional[float]]] = None

            # 1) 비디오 우선 처리
            if process_video:
                video_col = self.dataset_columns.get('video_column', 'video')
                if video_col in example and example[video_col] is not None:
                    video_frames = self._process_video(example[video_col])
                    if video_frames:
                        visuals_images = [img for (img, _ts) in video_frames]
                        frame_timestamps = [ts for (_img, ts) in video_frames]

            # 2) 이미지 처리 (비디오 미사용이거나 실패 시)
            if (not process_video) or (not visuals_images):
                if self.data_processing.get('image_data', True):
                    image_col = self.dataset_columns.get('image_column', 'image')
                    if image_col in example and example[image_col] is not None:
                        img_val = example[image_col]
                        if isinstance(img_val, (list, tuple)):
                            for one in img_val:
                                processed = self._process_image(one)
                                if processed is not None:
                                    visuals_images.append(processed)
                        else:
                            processed = self._process_image(img_val)
                            if processed is not None:
                                visuals_images.append(processed)
                        # 이미지 데이터에는 타임스탬프 없음
                        if visuals_images and frame_timestamps is None:
                            frame_timestamps = [None] * len(visuals_images)

            # 3) 메시지 생성 및 chat template 적용 (다중 프레임/이미지 지원)
            base_messages = self._format_messages(example, is_training=is_training)
            messages = self._build_messages_with_visuals(base_messages, len(visuals_images), frame_timestamps)

            try:
                # Generation prompt 정책:
                # - 학습(is_training=True): 마지막 assistant 응답이 이미 messages에 포함되어 있으므로 False
                # - 평가(is_training=False): 모델이 응답을 생성하도록 assistant 시작 프롬프트가 필요하므로 True
                text = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=(not is_training),
                    tokenize=False
                )
                texts.append(text.strip())
            except Exception as e:
                print(f"⚠️ Error applying chat template: {e}")
                # fallback: 간단한 텍스트 결합
                question = example.get(self.dataset_columns.get('question_column', 'question'), '')
                answer = example.get(self.dataset_columns.get('answer_column', 'answer'), '')
                # 평가 시에는 정답을 프롬프트에 넣지 않는다 (모델이 정답을 복사해 지표가 부풀려짐)
                tail = f"Answer: {answer}" if is_training else "Answer:"
                if visuals_images:
                    tag = "<video>" if len(visuals_images) > 1 else "<image>"
                    texts.append(f"{tag}\nQuestion: {question}\n{tail}")
                else:
                    texts.append(f"Question: {question}\n{tail}")

            # 4) 시각 데이터 저장 (샘플 단위의 리스트)
            visual_data.append(visuals_images)

        # 5. 프로세서로 배치 처리
        try:
            # VLM 모델에 따라 다른 처리 방식 적용 (샘플별 다중 이미지 지원)
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
        if is_training:
            # 6. 레이블 생성 및 마스킹
            # 목표: assistant 응답 토큰만 supervised signal을 주고,
            # system/user(프롬프트) 부분은 -100으로 마스킹
            labels = batch["input_ids"].clone()
            ignore_index = self.label_masking.get('ignore_index', -100)

            # 우선 특수 토큰은 항상 무시
            if self.ignore_in_loss_ids:
                # print(f"{len(self.ignore_in_loss_ids)} special token ids will be ignored in loss.")
                ignore_ids = sorted({int(i) for i in self.ignore_in_loss_ids if isinstance(i, (int,))})
                
                ids_tensor = torch.tensor(ignore_ids, dtype=labels.dtype, device=labels.device)
                mask_special = torch.isin(labels, ids_tensor)
                masked_count = int(mask_special.sum().item())
                # print(f"🔍 Masking {masked_count} special tokens in labels (from {len(ignore_ids)} ids)")
                if masked_count:
                    total = int(labels.numel())
                    # print(f"   -> {masked_count}/{total} tokens ({100.0 * masked_count / total:.2f}%) will be ignored in loss")
                    # If nearly all tokens are considered special, it's likely wrong; skip to be safe
                    labels[mask_special] = ignore_index

            # 응답 시작 위치 기반 마스킹: 문자열 기준으로 answer 텍스트 시작 지점을 찾아 토큰 오프셋과 정렬
            if self.label_masking.get('mask_input_tokens', True):
                try:
                    tokenizer = getattr(self.processor, 'tokenizer', self.processor)
                    answer_col = self.dataset_columns.get('answer_column', 'answer')
                    answers: List[str] = []
                    answer_starts: List[int] = []
                    for ex, txt in zip(examples, texts):
                        ans = ex.get(answer_col, '')
                        ans = '' if ans is None else str(ans)
                        answers.append(ans)
                        # 답변 문자열이 템플릿 내 어디서 시작하는지 찾음 (없으면 -1)
                        try:
                            start_idx = txt.rfind(ans) if ans else -1
                        except Exception:
                            start_idx = -1
                        answer_starts.append(start_idx)

                    # offsets를 얻기 위해 동일한 텍스트에 대해 토크나이저를 한 번 더 호출
                    tok_out = tokenizer(
                        texts,
                        return_offsets_mapping=True,
                        padding=self.text_processing.get('padding', True),
                        truncation=self.text_processing.get('truncation', True),
                        max_length=self.text_processing.get('max_length', 2048),
                        add_special_tokens=True
                    )
                    offsets = tok_out.get('offset_mapping')

                    if offsets is not None:
                        # 배치 차원 정렬 확인 (패딩으로 길이 통일되었음)
                        for i in range(labels.size(0)):
                            ans_start = answer_starts[i]
                            if ans_start is None or ans_start < 0:
                                # 답변 위치를 찾지 못한 경우: 프롬프트 마스킹을 적용하지 않고 특수 토큰만 무시
                                continue
                            seq_offsets = offsets[i]
                            # offsets 길이가 labels 길이와 동일해야 함
                            L = min(len(seq_offsets), labels.size(1))
                            for j in range(L):
                                # (start, end) = (0, 0)인 토큰은 보통 special
                                st, ed = seq_offsets[j]
                                # 답변 시작 이전(end <= ans_start)인 토큰은 프롬프트로 간주하고 마스킹
                                if ed <= ans_start:
                                    labels[i, j] = ignore_index
                    else:
                        print("⚠️ Token offsets not available; skipping prompt masking (only special tokens masked)")
                except Exception as e:
                    print(f"⚠️ Prompt masking failed: {e}; falling back to special-token-only masking")

            # # 선택적 추가 마스킹 옵션(유지)
            # if self.label_masking.get('mask_input_tokens', False):
            #     # 이미 위에서 프롬프트 마스킹을 수행하므로 별도 동작 불필요
            #     pass

            batch["labels"] = labels

        return batch
    
    def _process_with_processor(self, texts: List[str], visual_data: List[List[Image.Image]]) -> Dict[str, torch.Tensor]:
        """프로세서를 사용하여 텍스트와 시각 데이터를 처리합니다.
        - 각 샘플별로 다중 이미지/프레임을 지원합니다.
        - visual_data는 len(texts)와 동일한 길이의 리스트이며, 각 원소는 해당 샘플의 이미지 리스트입니다.
        """

        # 입력 정규화: None -> [] , 단일 이미지 -> [image]
        images_per_sample: List[List[Image.Image]] = []
        has_any_image = False
        for visuals in visual_data:
            if visuals is None:
                images_per_sample.append([])
                continue
            # 일부 구현에서 visuals가 단일 이미지일 수 있으므로 보정
            if isinstance(visuals, Image.Image):
                images_per_sample.append([visuals])
                has_any_image = True
            elif isinstance(visuals, (list, tuple)):
                # 리스트 내부 타입 확인 후 PIL이 아닌 경우 필터링
                valid_imgs = [v for v in visuals if isinstance(v, Image.Image)]
                images_per_sample.append(valid_imgs)
                if len(valid_imgs) > 0:
                    has_any_image = True
            else:
                # 알 수 없는 타입 -> 빈 처리
                images_per_sample.append([])

        if has_any_image:
            try:
                # 샘플별 다중 이미지 지원: images=[ [img1,img2], [], [img3] , ... ]
                batch = self.processor(
                    text=texts,
                    images=images_per_sample,
                    return_tensors=self.batch_processing.get('return_tensors', 'pt'),
                    padding=self.text_processing.get('padding', True),
                    truncation=self.text_processing.get('truncation', True),
                    max_length=self.text_processing.get('max_length', 2048)
                )
                return batch
            except Exception as e:
                print(f"⚠️ Error processing with multi-image inputs, trying text-only: {e}")

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
