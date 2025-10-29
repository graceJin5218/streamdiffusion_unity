import sys, os, io, socket, torch
from pydantic import BaseModel, Field
from PIL import Image
from typing import Literal, Optional
import numpy as np
import random
import torchvision.transforms as T
import cv2
import base64
import torch
torch.backends.cudnn.benchmark = True   # 고정 해상도에서 Convolution autotune
DEBUG = False                           # 로그 줄이기
from scipy import ndimage

def make_solid_rgb(h, w, rgb):
    """Create solid (H,W,3) RGB uint8 contiguous image."""
    r,g,b = map(int, rgb)
    img = np.empty((h,w,3), dtype=np.uint8)
    img[...,0] = r; img[...,1] = g; img[...,2] = b
    return np.require(img, dtype=np.uint8, requirements=["C"])


sys.path.append(os.path.join(os.path.dirname(__file__), "streamdiffusion"))
from utils.wrapper import StreamDiffusionWrapper

# 현재 스크립트가 위치한 디렉토리의 절대 경로를 가져옴
base_dir = os.path.dirname(os.path.abspath(__file__))

# 여러 위치에서 모델 파일을 찾도록 시도
model_filename = ""  # 移除硬编码的默认模型名称
possible_paths = [
    os.path.join(base_dir, "models", "Model", model_filename),  # 标准路径
    os.path.join("D:", "streamdiffusion_unity", "Assets", "StreamingAssets", "models", "Model", model_filename),  # 绝对路径
]


found_model_path = None
for path in possible_paths:
    if model_filename and os.path.exists(path):
        found_model_path = path
        print(f"Found model file at: {found_model_path}")
        break

if found_model_path is None:
    found_model_path = ""


vae_possible_paths = [
    os.path.join(base_dir, "models", "VAE"),
    os.path.join("D:", "streamdiffusion_unity", "Assets", "StreamingAssets", "models", "VAE"),
]

found_vae_path = None
for path in vae_possible_paths:
    config_path = os.path.join(path, "config.json")
    if os.path.exists(config_path):
        found_vae_path = path
        break

if found_vae_path is None:
    found_vae_path = "madebyollin/taesd"
    

found_lora_path = None
found_lora_path2 = None

base_model = found_model_path
taesd_model = found_vae_path
lora_model = found_lora_path
lora_model2 = found_lora_path2
pipeline_object = None
default_strength = 9.0
default_lora_scale = 0.85
default_lora_scale2 = 0.5
bypass_mode = False
is_linear_space = False
is_in_prediction = False
pattern_recognition_enabled = True  # 패턴 인식 기능 활성화 플래그
pattern_threshold = 0.3  # 패턴 인식 임계값

def analyze_image_pattern(image_array):
    """이미지의 의미 있는 패턴을 분석"""
    
    # 입력 배열 검증
    if image_array is None or image_array.size == 0:
        print("Warning: Empty image array")
        return 0.0
    
    # 그레이스케일 변환
    if len(image_array.shape) == 3:
        if image_array.shape[2] == 4:  # RGBA
            image_array = image_array[:, :, :3]  # RGB만 사용
        elif image_array.shape[2] != 3:
            print(f"Warning: unexpected channel count {image_array.shape[2]}")
            return 0.0
        gray = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    elif len(image_array.shape) == 2:
        gray = image_array.astype(np.uint8)
    else:
        print(f"Warning: unexpected array shape {image_array.shape}")
        return 0.0
    
    # 1. 엣지 검출로 형태 복잡도 측정
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # 2. 컨투어 검출로 닫힌 형태 찾기
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    closed_shapes = 0
    significant_contours = 0
    total_contour_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # 의미 있는 크기의 컨투어
            significant_contours += 1
            total_contour_area += area
            
            # 닫힌 형태 확인
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.3:  # 어느 정도 원형에 가까운 형태
                    closed_shapes += 1
    
    # 3. 연결된 컴포넌트 분석
    _, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    num_components = len(stats) - 1  # 배경 제외
    
    # 큰 컴포넌트의 비율 계산
    if num_components > 0:
        component_sizes = stats[1:, cv2.CC_STAT_AREA]
        large_components = np.sum(component_sizes > 500)
        max_component_size = np.max(component_sizes) if len(component_sizes) > 0 else 0
    else:
        large_components = 0
        max_component_size = 0
    
    # 4. 그려진 픽셀의 밀도
    drawn_pixels = np.sum(gray < 250)  # 흰색이 아닌 픽셀
    drawing_density = drawn_pixels / gray.size
    
    # 5. 선의 연속성 측정 (스켈레톤 분석)
    if drawn_pixels > 50:
        # 이진화
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        # 스켈레톤 추출 (선의 중심선)
        skeleton = cv2.ximgproc.thinning(binary) if hasattr(cv2, 'ximgproc') else binary
        skeleton_pixels = np.sum(skeleton > 0)
        line_complexity = skeleton_pixels / max(drawn_pixels, 1)
    else:
        line_complexity = 0
    
    # 종합 점수 계산
    pattern_score = 0.0
    
    # 엣지 밀도 (너무 적거나 너무 많으면 낮은 점수)
    if 0.01 < edge_density < 0.3:
        pattern_score += min(edge_density * 3.0, 1.0) * 0.2
    
    # 의미 있는 컨투어
    if significant_contours > 0:
        pattern_score += min(significant_contours / 5.0, 1.0) * 0.2
    
    # 닫힌 형태
    if closed_shapes > 0:
        pattern_score += min(closed_shapes / 3.0, 1.0) * 0.15
    
    # 큰 컴포넌트
    if max_component_size > 1000:
        pattern_score += min(max_component_size / 10000.0, 1.0) * 0.2
    
    # 그림 밀도 (적당한 밀도가 좋음)
    if 0.02 < drawing_density < 0.5:
        pattern_score += drawing_density * 0.15
    
    # 선의 복잡도
    if line_complexity > 0:
        pattern_score += min(line_complexity, 1.0) * 0.1
    
    if DEBUG: 
        print(f"패턴 분석: 엣지밀도={edge_density:.3f}, 컨투어={significant_contours}, "
            f"닫힌형태={closed_shapes}, 컴포넌트={num_components}, "
            f"그림밀도={drawing_density:.3f}, 최종점수={pattern_score:.3f}")
    
    return min(pattern_score, 1.0)

class Pipeline:
    def __init__(self, w: int, h: int, seed: int, device: torch.device, torch_dtype: torch.dtype,
                 use_vae: bool, use_lora: bool, gc_mode: Literal["img2img", "txt2img"], acc_mode: str,
                 positive_prompt: str, negative_prompt: str = "", preloaded_pipe=None, model_path=None, lora_dict=None,
                 cfg_type: str = "none", delta: float = 0.1, do_add_noise: bool = True, enable_similar_image_filter: bool = True,
                 similar_image_filter_threshold: float = 0.2, similar_image_filter_max_skip_frame: int = 10,
                 initial_guidance_scale: Optional[float] = None, user_strength: Optional[float] = None,
                 user_guidance_scale: Optional[float] = None, user_delta: Optional[float] = None):
        
        actual_model_path = model_path if model_path else base_model
        print(f"Initializing pipeline with model path: {actual_model_path}")
        
        self.delta = float(delta)
        self.cfg_type = cfg_type
        self.guidance_scale = float(initial_guidance_scale if initial_guidance_scale is not None else 7.0)

        #self.strength = float(default_strength)

        base_strength = default_strength if default_strength is not None else 9.0
        self.strength = float(base_strength)

        self.user_strength = float(user_strength) if user_strength is not None else None
        if self.user_strength is not None:
            self.strength = float(self.user_strength)

        self.user_guidance_scale = float(user_guidance_scale) if user_guidance_scale is not None else None
        if self.user_guidance_scale is not None:
            self.guidance_scale = float(self.user_guidance_scale)

        self.user_delta = float(user_delta) if user_delta is not None else None
        if self.user_delta is not None:
            self.delta = float(self.user_delta)
        
        self.do_add_noise = do_add_noise
        self.enable_similar_image_filter = enable_similar_image_filter
        self.similar_image_filter_threshold = similar_image_filter_threshold
        self.similar_image_filter_max_skip_frame = similar_image_filter_max_skip_frame

        # ====== 새로 추가: 전환/스타일 파라미터 (원하는 느낌으로 튜닝) ======
        self._mask_blur_px: float = 1.5   # 선 마스크 퍼짐 정도(픽셀)

        # --- alpha by drawing progress (coverage) ---
        self._blank_threshold = 0.0003  # 0.1% 미만은 사실상 백지 → α=0
        self._cov_full       = 0.012  # 화면의 ~3.5%를 그리면 α≈1에 도달 (느리게 하려면 ↑, 빠르게 하려면 ↓)

        # __init__ 끝부분 근처에 추가  // PATCH: detail ramp params
        self._detail_steps_min = 10     # 초반 스텝
        self._detail_steps_max = 24     # 후반 스텝 (LCM이면 16~24 권장)

        self._detail_guid_min  = 0.6    # 초반 guidance (LCM은 1~3대가 자연스러움)
        self._detail_guid_max  = 1.2    # 후반 guidance


        # 불필요한 잦은 prepare 방지를 위한 상태
        self._last_detail_bucket = None
        self._last_prompt_cache = ""        # prepare에 넣을 캐시
        self._last_negative_cache = ""

        # __init__ 끝부분(기존 params 인근)에 추가  # PATCH: background grain
        self._bg_noise_enable = True      # 배경 노이즈 항상 유지
        self._bg_noise_level  = 0.03      # 0.08~0.15 권장 (값↑ = 노이즈↑)
        self._bg_noise_sigma  = 0.1       # 가우시안 블러 정도(0.6~1.2)
        self._bg_noise_seed   = 12345     # 프레임마다 랜덤으로 바꾸고 싶으면 None로 두고 np.random.seed() 제거
        self._bg_noise_mask_soft = 0.4    # 선 주변부에 노이즈가 너무 끼지 않도록 가장자리 소프트닝(0.4~0.8)

        # --- 구조(라인) 기반 디테일 램프 ---  # PATCH
        self._struct_dilate_px       = 2      # 얇은 선 보정(1~5)
        self._struct_cov_full        = 0.0008  # 이 정도 구조면 충분하다고 간주
        self._detail_curve_pow_line  = 0.9    # 라인 램프 곡률(초반 민감도)

        # 창의성 램프 관련 (추가)
        self._strength_min = 1.5     # 초반(보수적)
        self._strength_max = 7    # 후반(창의적)

        self._delta_low   = 0.2     # 초반(수렴↑)
        self._delta_high  = 0.1     # 후반(수렴↓ = 창의성↑)


        # 선만 있어도 프롬프트를 따르게 하는 하한치(가드레일)
        self._guidance_floor_with_lines = 2.2  # LCM 계열 2.6~3.2 권장
        self._steps_floor_with_lines    = 22     # 최소 스텝


        # === Ghost(잔상) 점진 노출 파라미터 & 상태 ===
        self._ghost_bg_rgb   = (10, 10, 10)  # 배경 합성 색(어두운 회색 권장)
        self._ghost_matte_max = 1.0          # 초반: 배경과 많이 섞음(고스트 거의 안 보임)
        self._ghost_matte_min = 0.2          # 후반: 아직도 약간만 가림(완전 노출 원하면 0.0)
        self._ghost_reveal_frames = 24       # 몇 프레임에 걸쳐 0→1로 노출할지(0.5~1초에 해당)
        self._ghost_reveal = 1.0             # 현재 노출 값(0=숨김, 1=완전 노출)
        self._prev_drawn_px = 0              # 이전 프레임의 선 픽셀 수(변화 감지용)
        self._rearm_eps = 96                 # 선 픽셀 증가가 이 값 이상이면 '새 스트로크'로 간주


        # ================================================================
        
        if not os.path.exists(actual_model_path):
            print(f"Error: Model path does not exist: {actual_model_path}")
            return
        
        vae_path = None
        if use_vae:
            vae_path = "madebyollin/taesd"
            if taesd_model and os.path.exists(taesd_model) and os.path.isdir(taesd_model):
                print(f"Using local VAE folder: {taesd_model}")
                vae_path = taesd_model
        
        if use_lora and lora_dict is not None:
            print(f"LoRA will be used: {lora_dict}")
            
        if not preloaded_pipe:
            self.stream = StreamDiffusionWrapper(
                model_id_or_path=actual_model_path,
                t_index_list=[22],
                lora_dict=lora_dict,
                vae_id=vae_path,
                mode=gc_mode,
                seed=seed,
                cfg_type=cfg_type,
                use_safety_checker=False,
                engine_dir="engines",
                frame_buffer_size=1,
                width=w, 
                height=h,
                warmup=2,
                acceleration=acc_mode,
                use_denoising_batch=True,
                device=device,
                dtype=torch_dtype,
                output_type="pil"
            )
        else:
            self.stream = StreamDiffusionWrapper(
                model_id_or_path=actual_model_path,
                t_index_list=[22],
                lora_dict=lora_dict,
                vae_id=vae_path,
                mode=gc_mode,
                seed=seed,
                cfg_type=cfg_type,
                use_safety_checker=False,
                engine_dir="engines",
                frame_buffer_size=1,
                width=w, 
                height=h,
                warmup=2,
                acceleration=acc_mode,
                use_denoising_batch=True,
                device=device,
                dtype=torch_dtype,
                output_type="pil"
            )
    
    def _apply_runtime_parameters(self, guidance_scale_value: float, delta_value: float, strength_value: Optional[float]):
        try:
            self.guidance_scale = float(guidance_scale_value)
            self.delta = float(delta_value)
            if strength_value is not None:
                self.strength = float(strength_value)

            if hasattr(self, "stream") and hasattr(self.stream, "stream"):
                inner_stream = self.stream.stream
                if hasattr(inner_stream, "guidance_scale"):
                    inner_stream.guidance_scale = float(guidance_scale_value)
                if hasattr(inner_stream, "delta"):
                    inner_stream.delta = float(delta_value)
                if strength_value is not None and hasattr(inner_stream, "strength"):
                    inner_stream.strength = float(strength_value)
        except Exception as e:
            if DEBUG:
                print(f"Failed to apply runtime parameters: {e}")

    def update_user_params(self, strength: Optional[float] = None, delta: Optional[float] = None,
                           guidance_scale: Optional[float] = None):
        updated = False

        if strength is not None:
            try:
                strength_val = float(strength)
                self.user_strength = strength_val
                self.strength = strength_val
                updated = True
            except (TypeError, ValueError):
                if DEBUG:
                    print(f"Invalid strength override: {strength}")

        if delta is not None:
            try:
                delta_val = float(delta)
                if delta_val > 0:
                    self.user_delta = delta_val
                    self.delta = delta_val
                    updated = True
            except (TypeError, ValueError):
                if DEBUG:
                    print(f"Invalid delta override: {delta}")

        if guidance_scale is not None:
            try:
                guidance_val = float(guidance_scale)
                if guidance_val > 0:
                    self.user_guidance_scale = guidance_val
                    self.guidance_scale = guidance_val
                    updated = True
            except (TypeError, ValueError):
                if DEBUG:
                    print(f"Invalid guidance override: {guidance_scale}")

        if updated:
            self._last_detail_bucket = None
            try:
                effective_strength = self.user_strength if self.user_strength is not None else self.strength
                effective_delta = self.user_delta if self.user_delta is not None else self.delta
                effective_guidance = self.user_guidance_scale if self.user_guidance_scale is not None else self.guidance_scale
                self._apply_runtime_parameters(effective_guidance, effective_delta, effective_strength)
            except Exception as e:
                if DEBUG:
                    print(f"Runtime parameter sync failed: {e}")


    def _structure_score_from_lines(self, line_mask: np.ndarray) -> float:
        """
        얇은 선만 있어도 '구조가 충분하다'고 판단하기 위한 스코어.
        라인 마스크(0~1)를 이진화 후 팽창(dilate)해서 평균을 취함.
        """
        m = (line_mask > 0.25).astype(np.uint8)

        k = int(max(0, getattr(self, "_struct_dilate_px", 3)))
        if k > 0:
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1)))

        struct_cov = float(m.mean())                 # 0~1
        full = float(max(1e-6, getattr(self, "_struct_cov_full", 0.003)))
        pow_line = float(getattr(self, "_detail_curve_pow_line", 1.2))

        d_struct = (struct_cov / full) ** pow_line
        return float(np.clip(d_struct, 0.0, 1.0))
    
    
    def _extract_line_mask(self, pil_img: Image.Image, blur_px: float = 1.0) -> np.ndarray:
    
        arr = np.array(pil_img.convert("RGB"), dtype=np.uint8, copy=True)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # 선 감지 강화
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        g2 = clahe.apply(gray)
        g_blur = cv2.GaussianBlur(g2, (0,0), 0.6)
        g_sharp = cv2.addWeighted(g2, 1.8, g_blur, -0.8, 0)

        # 적응형 임계값
        thr_ad = cv2.adaptiveThreshold(
            g_sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            21, 6
        )
        _, thr_loose = cv2.threshold(g_sharp, 250, 255, cv2.THRESH_BINARY_INV)
        binv = cv2.max(thr_ad, thr_loose)

        edges = cv2.Canny(g_sharp, 30, 120)
        
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        clean = cv2.morphologyEx(binv, cv2.MORPH_OPEN, k3, iterations=1)

        mask_line = np.maximum(clean, edges).astype(np.uint8)

        # === 핵심: 닫힌 영역 자동 채우기 ===
        H, W = mask_line.shape[:2]
        
        # 1) 윤곽 닫기 (더 공격적으로)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # 7→9
        closed = cv2.morphologyEx(mask_line, cv2.MORPH_CLOSE, k_close, iterations=3)  # 2→3
        
        # 2) 컨투어로 내부 완전히 채우기
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros((H, W), np.uint8)
        
        if contours:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # 최소 면적 조건 완화 (더 작은 객체도 인식)
                if area > 50:  # 100→50
                    cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)
        
        # 3) 구멍 메우기 (객체 내부 완전 채움)
        ff = filled.copy()
        cv2.floodFill(ff, np.zeros((H+2, W+2), np.uint8), (0, 0), 128)
        holes = (ff == 0).astype(np.uint8) * 255
        filled = cv2.bitwise_or(filled, holes)
        
        # 4) 선과 채움 결합 - **채움 우선**
        # 선보다 채워진 영역을 더 강하게 반영
        mask_final = np.maximum(mask_line, filled).astype(np.uint8)  # filled * 0.8 제거
        
        # 블러는 최소화 (선명한 경계 유지)
        if blur_px and blur_px > 0:
            k = max(1, int(round(blur_px))*2 + 1)
            mask_final = cv2.GaussianBlur(mask_final, (k, k), blur_px * 0.5)  # 0.8→0.5
        
        m = (mask_final.astype(np.float32) / 255.0)
        return np.clip(m, 0.0, 1.0)


    def prepare(self, prompt, negative_prompt, target_guidance_scale=None):
        try:
            print(f"prompt: '{prompt}', negative prompt: '{negative_prompt}'")
            # delta_value = self.delta if hasattr(self, 'delta') else 0.8
            # strength_value = 9.0
            
            # global default_strength
            # if default_strength is not None:
            #     strength_value = default_strength
            
            # guidance_scale_value = target_guidance_scale if target_guidance_scale is not None else self.guidance_scale

            delta_value = self.user_delta if self.user_delta is not None else (self.delta if hasattr(self, 'delta') else 0.8)

            strength_value = None
            if self.user_strength is not None:
                strength_value = self.user_strength
            elif hasattr(self, 'strength') and self.strength is not None:
                strength_value = self.strength
            else:
                global default_strength
                strength_value = default_strength if default_strength is not None else 9.0

            guidance_scale_value = target_guidance_scale if target_guidance_scale is not None else (
                self.user_guidance_scale if self.user_guidance_scale is not None else self.guidance_scale
            )
            
            print(f"사용參數: 引导규模={guidance_scale_value}, delta={delta_value}, strength={strength_value}")
            
            prepare_params = {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'guidance_scale': guidance_scale_value,
                'delta': delta_value,
                'num_inference_steps': 18
            }
            self.stream.prepare(**prepare_params)

            try:
                self._apply_runtime_parameters(guidance_scale_value, delta_value, strength_value)
            except Exception as sync_err:
                if DEBUG:
                    print(f"Failed to sync runtime parameters: {sync_err}")

            # 기존 옵션 유지
            if hasattr(self.stream, 'stream'):
                if hasattr(self.stream.stream, 'do_add_noise') and hasattr(self, 'do_add_noise'):
                    self.stream.stream.do_add_noise = self.do_add_noise
                if hasattr(self.stream.stream, 'enable_similar_image_filter') and hasattr(self, 'enable_similar_image_filter'):
                    self.stream.stream.enable_similar_image_filter = self.enable_similar_image_filter
                if hasattr(self.stream.stream, 'similar_image_filter_threshold') and hasattr(self, 'similar_image_filter_threshold'):
                    self.stream.stream.similar_image_filter_threshold = self.similar_image_filter_threshold
                if hasattr(self.stream.stream, 'similar_image_filter_max_skip_frame') and hasattr(self, 'similar_image_filter_max_skip_frame'):
                    self.stream.stream.similar_image_filter_max_skip_frame = self.similar_image_filter_max_skip_frame

            print(f"생성기 준비 완료")
        except Exception as e:
            print(f"准备생성기시 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def _large_shape_hint_from_mask(self, line_mask: np.ndarray):
        """
        얇은 선으로 커다란 윤곽만 있을 때도 '큰 형태'를 감지해서
        초반 예측을 쉽게 하기 위한 힌트 스코어를 계산한다.
        return: d_large[0..1], bbox_area_ratio[0..1]
        """
        m = (line_mask > 0.25).astype(np.uint8)  # 선=1, 배경=0
        H, W = m.shape[:2]

        if m.sum() < 8:
            return 0.0, 0.0

        # 컨투어로 최대 컴포넌트/바운딩박스 추출
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, 0.0

        # 가장 큰 컨투어 기준
        areas = [cv2.contourArea(c) for c in contours]
        idx = int(np.argmax(areas))
        c = contours[idx]
        x, y, w, h = cv2.boundingRect(c)
        bbox_area = float(w * h)
        bbox_area_ratio = bbox_area / float(H * W)

        # “큰 윤곽”일수록 점수를 부여. 
        # 0.12~0.45 사이에서 빠르게 올라가고(임계 하향 가능), 0.6 이상이면 거의 완전한 힌트로 봄.
        # (필요시 숫자 조정)
        d_large = np.interp(bbox_area_ratio, [0.06, 0.25, 0.45], [0.0, 0.55, 0.85])
        d_large = float(np.clip(d_large, 0.0, 0.9))

        return d_large, float(np.clip(bbox_area_ratio, 0.0, 1.0))

    def _apply_background_noise(self, gen_pil: Image.Image, line_mask: np.ndarray) -> Image.Image:
        """
        예측 결과(gen_pil)에 배경 그레인(노이즈)을 얹되,
        - 고스트 리빌(t)이 진행될수록 노이즈를 급감(처음엔 숨기기, 후반엔 거의 제거)
        - 선 주변은 소프트 마스킹으로 최소화
        """
        if not getattr(self, "_bg_noise_enable", True):
            return gen_pil

        # 현재 고스트 리빌 상태(0=숨김, 1=완전 노출)
        t = float(getattr(self, "_ghost_reveal", 1.0))
        # 거의 다 드러났으면 노이즈 생략
        if t >= 0.8:
            return gen_pil

        # 기본 세기 -> (1 - t)^2 로 급격 감쇠 (처음엔 크고, 빨리 줄어듦)
        base_level = float(getattr(self, "_bg_noise_level", 0.12))
        effective_level = base_level * (1.0 - t) * (1.0 - t)
        if effective_level <= 1e-4:
            return gen_pil

        g = np.asarray(gen_pil.convert("RGB"), np.float32) / 255.0
        H, W, _ = g.shape

        # 재현성 유지용 시드(프레임별 변화를 주고 싶으면 None로 두고 seed 설정 제거)
        if getattr(self, "_bg_noise_seed", 12345) is not None:
            np.random.seed(int(self._bg_noise_seed))

        # 기본 화이트 노이즈 → 살짝 블러(필름 그레인 질감)
        n = np.random.randn(H, W, 3).astype(np.float32)
        sigma = float(getattr(self, "_bg_noise_sigma", 0.8))
        try:
            n = cv2.GaussianBlur(n, (0, 0), sigma)
        except Exception:
            from scipy import ndimage
            n = ndimage.gaussian_filter(n, sigma=sigma)

        # [0,1] 정규화
        n = (n - n.min()) / max(1e-6, (n.max() - n.min()))

        # 선(=1) 주변을 부드럽게 제외
        m = np.clip(line_mask.astype(np.float32), 0.0, 1.0)  # 선 = 1
        inv = 1.0 - m                                        # 배경 = 1

        # 소프트닝(경계에서 노이즈가 거칠게 보이는 것 완화)
        soft = float(getattr(self, "_bg_noise_mask_soft", 0.8))  # 0.4~1.2 권장
        try:
            inv_soft = cv2.GaussianBlur(inv, (0, 0), soft)
        except Exception:
            from scipy import ndimage
            inv_soft = ndimage.gaussian_filter(inv, sigma=soft)

        # 3채널 가중치
        w3 = (effective_level * np.clip(inv_soft, 0.0, 1.0))[..., None]

        out = g * (1.0 - w3) + n * w3
        out = np.clip(out, 0.0, 1.0)
        return Image.fromarray((out * 255.0).astype(np.uint8), "RGB")

    
    def predict(self, input_image: Image.Image, new_prompt: Optional[str] = None, pattern_score: Optional[float] = None) -> Image.Image:
        try:
            # --- 안전한 RGB ---
            if input_image.mode != "RGB":
                if input_image.mode == "RGBA":
                    bg = Image.new("RGB", input_image.size, (255, 255, 255))
                    bg.paste(input_image, mask=input_image.split()[3] if len(input_image.split()) > 3 else None)
                    input_image = bg
                else:
                    input_image = input_image.convert("RGB")

            arr_full = np.array(input_image)

            # --- 패턴 스코어(선택) ---
            if pattern_score is None:
                h, w = input_image.height, input_image.width
                scale = 256 / max(h, w)
                small = cv2.resize(arr_full, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else arr_full
                pattern_score = analyze_image_pattern(small)

            # --- 텐서 준비 ---
            transform = T.Compose([T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            image_tensor = transform(input_image).unsqueeze(0)
            if torch.cuda.is_available():
                image_tensor = image_tensor.to(device="cuda", dtype=torch.float16)
            else:
                image_tensor = image_tensor.to(self.stream.device if hasattr(self, 'stream') else "cpu")

            # --- 프롬프트 업데이트 ---
            try:
                if hasattr(self.stream, "stream") and hasattr(self.stream.stream, "update_prompt"):
                    self.stream.stream.update_prompt(new_prompt if new_prompt else "")
            except Exception as e:
                print(f"[predict] prompt update failed: {e}")

            # PATCH: prepare 호출에 사용할 캐시
            if new_prompt is not None:
                self._last_prompt_cache = new_prompt
            # # negative_prompt가 함수 인자로 있다면 동일하게 갱신
            # if 'negative_prompt' in locals() and negative_prompt is not None:
            #     self._last_negative_cache = negative_prompt

            # --- 모델 추론 ---
            try:
                gen_out = self.stream(image=image_tensor, prompt=new_prompt)
            except Exception as e:
                print(f"[predict] inference error: {e}")
                gen_out = None

            # --- 예측 PIL ---
            if isinstance(gen_out, torch.Tensor):
                t = gen_out
                if t.dim() == 4: t = t.squeeze(0)
                t = t.clamp(0, 1).cpu().float()
                if t.shape[0] == 3:
                    gen_arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    gen_pil = Image.fromarray(gen_arr, mode="RGB")
                else:
                    gen_pil = input_image.copy()
            elif isinstance(gen_out, Image.Image):
                gen_pil = gen_out.convert("RGB")
            else:
                gen_pil = input_image.copy()

            if gen_pil.size != input_image.size:
                gen_pil = gen_pil.resize(input_image.size, Image.BICUBIC)

            # --- 선 마스크 & 커버리지 ---
            line_mask = self._extract_line_mask(input_image, blur_px=self._mask_blur_px)
            line_coverage = float(np.clip(line_mask.mean(), 0.0, 1.0))
            drawn_px = int((line_mask > 0.25).sum())

            # --- 라인 구조 기반 d_struct (선만 있어도 디테일↑) ---
            d_struct = self._structure_score_from_lines(line_mask)

            # --- 큰 윤곽 힌트(d_large): 내부가 비어있어도 큰 형태만 보이면 초반 가속 ---
            d_large, bbox_ratio = self._large_shape_hint_from_mask(line_mask)

           
            # --- 커버리지 기반 d_area ---
            cov_full = float(getattr(self, "_cov_full", 0.012))
            blank_th = float(getattr(self, "_blank_threshold", 0.003))
            cov_pow  = float(getattr(self, "_detail_curve_pow", 1.5))

            cov_norm = max(0.0, (line_coverage - blank_th)) / max(cov_full, 1e-6)
            d_area   = float(np.clip(cov_norm ** cov_pow, 0.0, 1.0))

           # (B) 최종 디테일 팩터: “선만” 있어도, “큰 윤곽”만 있어도 쉽게 올라가게
            #     => 내부를 채우지 않아도 초반 예측이 붙음
            d = float(np.clip(max(d_area, d_struct, d_large), 0.0, 1.0))


            # 목표 steps/guidance/delta
            steps_min = int(getattr(self, "_detail_steps_min", 10))
            steps_max = int(getattr(self, "_detail_steps_max", 22))
            guid_min  = float(getattr(self, "_detail_guid_min", 0.5))
            guid_max  = float(getattr(self, "_detail_guid_max", 1.2))
         

            # --- 창의성 램프 ---
            # d: 0(초반, 보수적) → 1(후반, 창의적)
            creative = float(np.clip(d, 0.0, 1.0))

            # 1) steps/guidance: 완만히 증가(너무 과하지 않게)
            target_steps = int(round(steps_min + creative * (steps_max - steps_min)))
            target_guid  = float(guid_min + creative * (guid_max - guid_min))

            # 2) delta: 그릴수록 ↑ (기존과 반대 방향) → 수렴 완화 & 다양성↑
            target_delta = float(self._delta_low + creative * (self._delta_high - self._delta_low))

            # 3) strength(denoise): 그릴수록 ↑ → 입력(내 선)에 덜 고정
            target_strength = float(self._strength_min + creative * (self._strength_max - self._strength_min))

            if self.user_guidance_scale is not None:
                target_guid = float(self.user_guidance_scale)
            if self.user_delta is not None:
                target_delta = float(self.user_delta)
            if self.user_strength is not None:
                target_strength = float(self.user_strength)


            # 선이 어느 정도 있으면 너무 낮게 떨어지지 않도록 하한만 살짝 보정
            if drawn_px >= int(getattr(self, "_trigger_px", 4)):
                target_guid  = max(target_guid, float(getattr(self, "_guidance_floor_with_lines", 1.0)))
                target_steps = max(target_steps, int(getattr(self, "_steps_floor_with_lines", 16)))

            # 너무 잦은 prepare 방지: 버킷화 후 필요 시에만 갱신
            bucket = (target_steps, round(target_guid, 1), round(target_delta, 2))
            if bucket != getattr(self, "_last_detail_bucket", None):
                try:
                    prep_prompt   = getattr(self, "_last_prompt_cache", "") or (new_prompt or "")
                    prep_negative = getattr(self, "_last_negative_cache", "")
                    self.stream.prepare(
                        prompt=prep_prompt,
                        negative_prompt=prep_negative,
                        guidance_scale=target_guid,
                        delta=target_delta,
                        strength=target_strength,
                        num_inference_steps=target_steps
                    )
                    try:
                        self._apply_runtime_parameters(target_guid, target_delta, target_strength)
                    except Exception as sync_err:
                        if DEBUG:
                            print(f"[detail] parameter sync failed: {sync_err}")
                    self._last_detail_bucket = bucket
                    if DEBUG:
                        print(f"[detail] d={d:.2f} (area={d_area:.2f}, struct={d_struct:.2f}) "
                            f"steps={target_steps}, guid={target_guid:.2f}, delta={target_delta:.2f}")
                except Exception as e:
                    if DEBUG:
                        print(f"[detail] prepare 갱신 실패: {e}")


            # 완전 백지는 여전히 흰 화면 반환 (진짜 아무것도 안 그렸을 때만)
            if drawn_px < getattr(self, "_trigger_px", 4):
                return Image.new("RGB", input_image.size, (255, 255, 255))
            

            
            # 0) 새 스트로크(그림 증가) 감지 시 페이드 리셋
            if drawn_px >= getattr(self, "_trigger_px", 4):
                if (drawn_px - self._prev_drawn_px) >= self._rearm_eps:
                    # 새로 그리기 시작했다고 판단 → 처음엔 고스트 안 보이게
                    self._ghost_reveal = 0.0
                self._prev_drawn_px = drawn_px
            else:
                # 거의 백지면 점차 원래 상태로(리셋)
                self._ghost_reveal = min(1.0, self._ghost_reveal + (1.0 / max(1, self._ghost_reveal_frames)))

            # 1) 예측 결과 채택
            pred_pil = gen_pil.convert("RGB")
            pred_rgb = np.array(pred_pil, dtype=np.uint8, copy=True)

            # 2) 배경색 프레임(연속 메모리) 생성
            H, W = pred_rgb.shape[:2]
            bg = make_solid_rgb(H, W, getattr(self, "_ghost_bg_rgb", (10,10,10)))

            # 3) 현재 프레임에서 사용할 '가림 강도(매트)'와 '노출(고스트 리빌)' 계산
            t = float(np.clip(self._ghost_reveal, 0.0, 1.0))                 # 0→1
            matte_strength = float(np.interp(t, [0.0, 1.0],
                                            [self._ghost_matte_max, self._ghost_matte_min]))
            # 4) 단계 1: 배경과 매트(미리 합성) → 고스트 거의 안 보이는 버전
            ghost_hidden = (bg.astype(np.float32) * (1.0 - matte_strength) +
                            pred_rgb.astype(np.float32) * matte_strength).clip(0,255).astype(np.uint8)

            # 5) 단계 2: '숨김'과 '원본' 사이를 고스트 리빌(t)로 보간
            out_rgb = (ghost_hidden.astype(np.float32) * (1.0 - t) +
                    pred_rgb.astype(np.float32) * t).clip(0,255).astype(np.uint8)

            # 6) (선택) 배경 노이즈 적용: 잔상 질감 통일
            out_pil = Image.fromarray(out_rgb, "RGB")
            out_pil = self._apply_background_noise(out_pil, line_mask)

            return out_pil


        except Exception as e:
            print(f"[predict] ERROR: {e}")
            import traceback; traceback.print_exc()
            return Image.new("RGB", (512, 512), (255, 0, 0))

            
def setModelPaths(base_m: str, tiny_vae_m: str, lcm_lora_m: str, lcm_lora_m2: str = None):
    global base_model, taesd_model, lora_model, lora_model2
    
    print(f"接收到的路径: 基础模型={base_m}, VAE={tiny_vae_m}, LoRA1={lcm_lora_m}, LoRA2={lcm_lora_m2}")
    
    if base_m and os.path.exists(base_m):
        base_model = base_m
    else:
        print(f"警告: 未提供有效的基础模型路径或文件不存在: {base_m}")
    
    if tiny_vae_m:
        if os.path.exists(tiny_vae_m) and os.path.exists(os.path.join(tiny_vae_m, "config.json")):
            taesd_model = tiny_vae_m
        elif tiny_vae_m.startswith("madebyollin"):
            taesd_model = tiny_vae_m
        else:
            taesd_model = "madebyollin/taesd"
    
    if lcm_lora_m and os.path.exists(lcm_lora_m):
        lora_model = lcm_lora_m
    else:
        lora_model = None
        
    if lcm_lora_m2 and os.path.exists(lcm_lora_m2):
        lora_model2 = lcm_lora_m2
    else:
        lora_model2 = None
    
def loadPipeline(w: int, h: int, seed: int, use_vae: bool, use_lora: bool,
                 acc_mode: str, positive_prompt: str, negative_prompt: str, strength: float = None, 
                 lora_scale: float = None, lora_scale2: float = None,
                 delta: Optional[float] = None, do_add_noise: bool = True,
                 enable_similar_image_filter: bool = True,
                 similar_image_filter_threshold: float = 0.2,
                 similar_image_filter_max_skip_frame: int = 1,
                 guidance_scale: Optional[float] = None):
    
    global default_strength, default_lora_scale, default_lora_scale2, pipeline_object
    
    if not base_model:
        print("错误: 没有有效的基础模型路径，无法加载Pipeline")
        return False
    
    if not os.path.exists(base_model):
        print(f"错误: 基础模型文件不存在: {base_model}")
        return False
    
    provided_strength = strength
    provided_delta = delta
    provided_guidance = guidance_scale
        
    if strength is None:
        strength = default_strength
    else:
        default_strength = strength

    if delta is None:
        delta = 0.1

    if guidance_scale is None:
        guidance_scale = 7.0
    
    if lora_scale is None:
        lora_scale = default_lora_scale
    else:
        default_lora_scale = lora_scale
        
    if lora_scale2 is None:
        lora_scale2 = default_lora_scale2
    else:
        default_lora_scale2 = lora_scale2
    
    if positive_prompt is None:
        positive_prompt = ""
    if negative_prompt is None:
        negative_prompt = ""

    if DEBUG:     
        print(f"加载Pipeline参数: 模型={base_model}, VAE={use_vae}, LoRA={use_lora}")
        print(f"强度={strength}, LoRA强度={lora_scale}/{lora_scale2}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16
    
    try:
        lora_dict_to_use = {}
        
        if use_lora and lora_model is not None and os.path.exists(lora_model):
            lora_dict_to_use[lora_model] = lora_scale
            print(f"Using LoRA1 with scale: {lora_scale}")
            
        if use_lora and lora_model2 is not None and os.path.exists(lora_model2):
            lora_dict_to_use[lora_model2] = lora_scale2
            print(f"Using LoRA2 with scale: {lora_scale2}")
            
        if not lora_dict_to_use:
            lora_dict_to_use = None
        
        pipeline_object = Pipeline(
            w=w, h=h, seed=seed, device=device, torch_dtype=torch_dtype, 
            use_vae=True,
            use_lora=use_lora,
            gc_mode="img2img",
            acc_mode=acc_mode, 
            positive_prompt=positive_prompt, 
            negative_prompt=negative_prompt, 
            model_path=base_model,
            lora_dict=lora_dict_to_use,
            cfg_type="none",
            delta=delta,
            do_add_noise=do_add_noise,
            enable_similar_image_filter=enable_similar_image_filter,
            similar_image_filter_threshold=similar_image_filter_threshold,
            similar_image_filter_max_skip_frame=similar_image_filter_max_skip_frame,
            initial_guidance_scale=guidance_scale,
            user_strength=provided_strength,
            user_guidance_scale=provided_guidance,
            user_delta=provided_delta
        )

        try:
            pipeline_object.prepare(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                target_guidance_scale=guidance_scale
            )

            return True
        except Exception as e:
            print(f"调用prepare方法时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"ERROR IN LOADING PIPELINE: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    if pipeline_object is None:
        print("Pipeline创建失败，结果为None")
        return False
    
    print("Pipeline加载成功")
    return True

def runPipeline(input_bytes, new_prompt: str, pattern_score: float = None):
    output_io = io.BytesIO()
    if pipeline_object is None:
        print("pipeline이 비어있음, 테스트 이미지 반환")
        test_image = Image.new("RGB", (512, 512), (255, 100, 100))
        test_image.save(output_io, format="PNG")
        return output_io.getvalue()
    else:
        try:
            if new_prompt is None:
                new_prompt = ""
                
            try:
                input_image = Image.open(io.BytesIO(input_bytes))            
                
                # 무조건 RGB로 변환
                if input_image.mode != "RGB":                
                    if input_image.mode == "RGBA":
                        # RGBA의 경우 흰색 배경으로 합성
                        background = Image.new("RGB", input_image.size, (255, 255, 255))
                        background.paste(input_image, mask=input_image.split()[3])
                        input_image = background
                    elif input_image.mode == "L":
                        # 그레이스케일을 RGB로
                        input_image = input_image.convert("RGB")
                    elif input_image.mode == "P":
                        # 팔레트 모드를 RGB로
                        input_image = input_image.convert("RGBA").convert("RGB")
                    else:
                        input_image = input_image.convert("RGB")
                    
                
                global bypass_mode
                if bypass_mode:
                    print("바이패스 모드 활성화, 원본 입력 이미지 직접 반환")
                    input_bytes_io = io.BytesIO()
                    input_image.save(input_bytes_io, format="PNG")
                    return input_bytes_io.getvalue()
                    
            except Exception as e:
                print(f"입력 이미지 열기 실패: {e}")
                input_image = Image.new("RGB", (512, 512), (0, 255, 0))
            
            if DEBUG: 
                print("pipeline_object.predict 호출하여 이미지 처리")
            
            global is_in_prediction
            is_in_prediction = True
            
            try:
                # predict 메서드에 pattern_score 전달
                output_image = pipeline_object.predict(input_image, new_prompt, pattern_score)
                is_in_prediction = False
                
                if isinstance(output_image, Image.Image):
                    if DEBUG: 
                        print(f"출력이 PIL 이미지, 크기: {output_image.size}, 모드: {output_image.mode}")
                    
                    # 최종 RGB 확인
                    if output_image.mode != "RGB":
                        if DEBUG: 
                            print(f"Converting output from {output_image.mode} to RGB")
                        output_image = output_image.convert("RGB")
                    
                    # 디버깅: numpy 배열로 변환하여 채널 확인
                    arr_output = np.array(output_image)
                    if DEBUG:
                        print(f"출력 배열 shape: {arr_output.shape}, dtype: {arr_output.dtype}")
                    
                    if len(arr_output.shape) == 3:
                        if arr_output.shape[2] != 3:
                            print(f"ERROR: Output has {arr_output.shape[2]} channels, fixing...")
                            if arr_output.shape[2] == 4:
                                arr_output = arr_output[:, :, :3]
                            elif arr_output.shape[2] == 1:
                                arr_output = np.stack([arr_output[:, :, 0]] * 3, axis=2)
                            output_image = Image.fromarray(arr_output.astype(np.uint8), mode='RGB')
                    
                    if DEBUG:
                        print(f"최종 출력 이미지 평균 RGB 값: R={arr_output[:,:,0].mean():.2f}, G={arr_output[:,:,1].mean():.2f}, B={arr_output[:,:,2].mean():.2f}")
                    
                    # PNG로 저장
                    output_image.save(output_io, format="PNG", compress_level=0)
                    output_bytes = output_io.getvalue()
                    if DEBUG:
                        print(f"PNG 저장 완료, 크기: {len(output_bytes)} 바이트")
                    
                    # 처음 몇 바이트 확인 (PNG 시그니처)
                    if len(output_bytes) > 8:
                        png_sig = output_bytes[:8]
                        if DEBUG:
                            print(f"PNG 시그니처: {' '.join([f'{b:02x}' for b in png_sig])}")
                    
                    return output_bytes
                
                else:
                    if DEBUG:
                        print(f"예상치 못한 출력 타입: {type(output_image)}")
                    fallback_image = Image.new("RGB", (512, 512), (0, 0, 255))
                    fallback_image.save(output_io, format="PNG")
                    return output_io.getvalue()
                    
            except Exception as e:
                is_in_prediction = False
                print(f"처리 중 오류: {e}")
                import traceback
                traceback.print_exc()
                raise
                
        except Exception as e:
            print(f"runPipeline에서 오류: {e}")
            import traceback
            traceback.print_exc()
            
            test_image = Image.new("RGB", (512, 512), (255, 165, 0))
            test_io = io.BytesIO()
            test_image.save(test_io, format="PNG")
            return test_io.getvalue()

def processData(client_socket, data):
    global pipeline_object, bypass_mode, is_linear_space, pattern_recognition_enabled, pattern_threshold
    try:
    
        try:
            data_str = data.decode('utf-8')
            #print(f"Data decoded successfully as UTF-8: {data_str[:50]}..." if len(data_str) > 50 else data_str)
        except UnicodeDecodeError:
            print(f"Failed to decode data as UTF-8, treating as binary")
            
        data = data.strip(b"|start|").strip(b"|end|")
        parts = data.split(b"||")
        #print(f"Split data into {len(parts)} parts")

        # 초기 변수 설정
        use_vae = True
        use_lora = True
        width = 512
        height = 512
        seed = -1
        strength = None
        lora_scale = None
        lora_scale2 = None
        delta = None
        do_add_noise = True
        enable_similar_image_filter = True
        similar_image_filter_threshold = 0.2
        similar_image_filter_max_skip_frame = 4
        guidance_scale = None
        command_state = 0
        command = ""
        acc_mode = "tensorrt"
        prompt = "tree, flower, stem, mountain, sun, sky, moon"
        neg_prompt = "text, hard edges, over-smoothing, llow quality, bad quality, blurry, low resolution"
        base_m = base_model
        lora_m = None
        lora_m2 = None
        image = io.BytesIO()
        pattern_score = None  # 패턴 점수 추가

        for i in range(0, len(parts), 2):
            if i + 1 >= len(parts):
                break
                
            try:
                key = parts[i].decode('utf-8')
            except UnicodeDecodeError:
                print(f"Failed to decode key at index {i}, skipping this pair")
                continue
                
            value = parts[i + 1]
            
            if key in ["command", "base_model", "taesd_model", "lora_model", "lora_model2", "prompt", "acceleration", "strength", "lora_scale", "lora_scale2", "pattern_score"]:
                try:
                    if key != "command":
                        if DEBUG:
                            print(f"Received {key}: {value.decode('utf-8', errors='replace')}")
                except Exception as e:
                    print(f"Error decoding value for {key}: {e}")

            # 패턴 점수 처리
            if key == "pattern_score":
                try:
                    pattern_score = float(value)
                    if DEBUG:
                        print(f"패턴 점수 받음: {pattern_score}")
                except ValueError:
                    print(f"Invalid pattern_score value: {value}")
                    pattern_score = None

            # 패턴 인식 활성화 설정
            elif key == "pattern_recognition":
                try:
                    pattern_value = value.decode('utf-8').lower()
                    pattern_recognition_enabled = pattern_value in ["true", "1", "yes", "y"]
                    print(f"패턴 인식 설정: {pattern_recognition_enabled}")
                except Exception as e:
                    print(f"패턴 인식 설정 파싱 오류: {e}")

            # 패턴 임계값 설정
            elif key == "pattern_threshold":
                try:
                    pattern_threshold = float(value)
                    print(f"패턴 임계값 설정: {pattern_threshold}")
                except ValueError:
                    print(f"Invalid pattern_threshold value: {value}")

            elif key == "bypass_mode":
                try:
                    bypass_value = value.decode('utf-8').lower()
                    bypass_mode = bypass_value in ["true", "1", "yes", "y"]
                    #print(f"设置bypass_mode: {bypass_mode}")
                except Exception as e:
                    print(f"解析bypass_mode参数时出错: {e}")
            
            elif key == "run":
                if command_state == 1 and len(command) > 0:
                    if DEBUG:
                        print(f"Executing command: {command}")
                    if command == "paths":
                        if DEBUG:
                            print(f"设置路径: 基础模型={base_m}, LoRA1={lora_m}, LoRA2={lora_m2}")
                        setModelPaths(base_m, taesd_model, lora_m, lora_m2)
                        try:
                            client_socket.send(b"paths_set")
                            if DEBUG:
                                print("Sent confirmation: paths_set")
                        except Exception as e:
                            if DEBUG:
                                print(f"Failed to send confirmation: {e}")
                    elif command == "load":
                        if DEBUG:
                            print(f"Loading pipeline with parameters")
                        try:
                            if bypass_mode:
                                if pipeline_object is None:
                                    try:
                                        loadPipeline(width, height, seed, use_vae, use_lora, acc_mode, prompt, neg_prompt, 
                                                    strength, lora_scale, lora_scale2, delta, do_add_noise,
                                                    enable_similar_image_filter, similar_image_filter_threshold, 
                                                    similar_image_filter_max_skip_frame, guidance_scale)
                                    except Exception as e:
                                        if DEBUG:
                                            print(f"尝试在bypass模式下加载Pipeline时出错: {e}")
                                
                                client_socket.send(b"loaded")
                                if DEBUG:
                                    print("Sent: loaded - bypass mode")
                            else:
                                loadPipeline(width, height, seed, use_vae, use_lora, acc_mode, prompt, neg_prompt, 
                                          strength, lora_scale, lora_scale2, delta, do_add_noise,
                                          enable_similar_image_filter, similar_image_filter_threshold, 
                                          similar_image_filter_max_skip_frame, guidance_scale)
                                if pipeline_object is None:
                                    client_socket.send(b"failed")
                                    print("Sent: failed - pipeline is None")
                                else:
                                    client_socket.send(b"loaded")
                                    print("Sent: loaded - pipeline loaded successfully")
                        except Exception as e:
                            if bypass_mode:
                                client_socket.send(b"loaded")
                                print(f"在bypass模式下忽略错误: {e}, 返回loaded")
                            else:
                                print(f"Error loading pipeline: {e}")
                                client_socket.send(b"failed")
                    elif command == "advance":
                        if DEBUG:
                            print(f"Advancing pipeline with prompt: {prompt}, pattern_score: {pattern_score}")

                        # # 동적 강도 업데이트
                        if strength is not None:
                            if DEBUG:
                                print(f"Using dynamic strength: {strength}")
                            
                        if hasattr(pipeline_object, 'stream') and hasattr(pipeline_object.stream, 'stream'):
                            if hasattr(pipeline_object.stream.stream, 'lora_scale'):
                                print(f"Updating LoRA scales: LoRA1={lora_scale}, LoRA2={lora_scale2}")
                                pipeline_object.stream.stream.lora_scale = lora_scale
                        
                        # 패턴 점수와 함께 파이프라인 실행
                        output_bytes = runPipeline(image.getvalue(), prompt, pattern_score)
                        if output_bytes:
                            client_socket.sendall(output_bytes)
                            client_socket.send(b"||||")
                            if DEBUG:
                                print(f"Sent generated image, size: {len(output_bytes)} bytes")
                        else:
                            print("Failed to generate image")
                            client_socket.send(b"failed")
                    elif command == "unload":
                        print("Unloading pipeline")
                        pipeline_object = None
                    else:
                        print(f"Unknown command: {command}")
                    command = ""
                    command_state = 0
            elif key == "command":
                try:
                    command = value.decode('utf-8', errors='replace')
                    #print(f"Command received: {command}")
                    command_state = 1
                except Exception as e:
                    print(f"Error decoding command: {e}")
                    command = ""
                    command_state = 0
            elif key == "width":
                try:
                    width = int(value)
                except ValueError:
                    print(f"Invalid width value: {value}")
            elif key == "height":
                try:
                    height = int(value)
                except ValueError:
                    print(f"Invalid height value: {value}")
            elif key == "seed":
                try:
                    raw_seed_value = value.decode('utf-8', errors='replace')
                    seed = int(raw_seed_value)
                    print(f"Parsed seed value: {seed}")
                except ValueError:
                    print(f"Invalid seed value: '{value}'")
                    seed = random.randint(0, 1000000)
            elif key == "strength":
                try:
                    strength_value = float(value)
                    strength = strength_value
                    global default_strength
                    default_strength = strength_value
                    if pipeline_object is not None:
                        pipeline_object.update_user_params(strength=strength_value)
                    if DEBUG:
                        print(f"Setting strength to: {strength}")
                except ValueError:
                    print(f"Invalid strength value: {value}")
            elif key == "lora_scale":
                try:
                    lora_scale_value = float(value)
                    if 0 < lora_scale_value <= 1.0:
                        lora_scale = lora_scale_value
                        global default_lora_scale
                        default_lora_scale = lora_scale_value
                        if DEBUG:
                            print(f"Setting LoRA1 scale to: {lora_scale}")
                except ValueError:
                    print(f"Invalid LoRA1 scale value: {value}")
            elif key == "lora_scale2":
                try:
                    lora_scale2_value = float(value)
                    if 0 < lora_scale2_value <= 1.0:
                        lora_scale2 = lora_scale2_value
                        global default_lora_scale2
                        default_lora_scale2 = lora_scale2_value
                        if DEBUG:
                            print(f"Setting LoRA2 scale to: {lora_scale2}")
                except ValueError:
                    print(f"Invalid LoRA2 scale value: {value}")
            elif key == "use_vae":
                try:
                    use_vae = (int(value) > 0)
                except ValueError:
                    print(f"Invalid use_vae value: {value}")
            elif key == "use_lora":
                try:
                    use_lora = (int(value) > 0)
                except ValueError:
                    print(f"Invalid use_lora value: {value}")
            elif key == "acceleration":
                try:
                    acc_mode = value.decode('utf-8', errors='replace')
                except Exception as e:
                    print(f"Error decoding acceleration: {e}")
                    acc_mode = "tensorrt"
            elif key == "prompt":
                try:
                    prompt = value.decode('utf-8', errors='replace')
                except Exception as e:
                    print(f"Error decoding prompt: {e}")
                    prompt = ""
            elif key == "neg_prompt":
                try:
                    neg_prompt = value.decode('utf-8', errors='replace')
                except Exception as e:
                    print(f"Error decoding negative prompt: {e}")
                    neg_prompt = ""
            elif key == "base_model":
                try:
                    base_m = value.decode('utf-8', errors='replace')
                except Exception as e:
                    print(f"Error decoding base_model: {e}")
            elif key == "lora_model":
                try:
                    lora_m = value.decode('utf-8', errors='replace')
                    if lora_m and os.path.exists(lora_m):
                        if DEBUG:
                            print(f"有效的LoRA1模型路径: {lora_m}")
                    else:
                        lora_m = None
                except Exception as e:
                    print(f"解析lora_model时出错: {e}")
                    lora_m = None
            elif key == "lora_model2":
                try:
                    lora_m2 = value.decode('utf-8', errors='replace')
                    if lora_m2 and os.path.exists(lora_m2):
                        if DEBUG:
                            print(f"Valid second LoRA model path: {lora_m2}")
                    else:
                        lora_m2 = None
                except Exception as e:
                    print(f"Error decoding lora_model2: {e}")
                    lora_m2 = None
            elif key == "image":
                try:
                    image = io.BytesIO(value)
                    if DEBUG:
                        print(f"Received image data, size: {len(value)} bytes")
                except Exception as e:
                    print(f"Error processing image data: {e}")
                    image = io.BytesIO()
            elif key == "image_base64":
                try:

                    base64_str = value.decode('utf-8', errors='replace')
                    img_data = base64.b64decode(base64_str)
                    image = io.BytesIO(img_data)
                    if DEBUG:
                        print(f"接收到Base64图像数据，解码后大小: {len(img_data)} 字节")
                    
                    if bypass_mode:
                        if DEBUG:
                            print("绕过模式：保持原始图像数据不变")
                    elif is_linear_space:
                        try:
                            temp_image = Image.open(image)
                            arr = np.array(temp_image)
                            if DEBUG:
                                print(f"原始图像(线性空间)平均RGB值: R={arr[:,:,0].mean():.2f}")
                            
                            arr_float = arr.astype(np.float32) / 255.0
                            arr_gamma = np.power(arr_float, 1/2.2) * 255.0
                            arr_gamma = np.clip(arr_gamma, 0, 255).astype(np.uint8)
                            
                            gamma_image = Image.fromarray(arr_gamma)
                            temp_io = io.BytesIO()
                            gamma_image.save(temp_io, format="PNG")
                            image = io.BytesIO(temp_io.getvalue())
                            
                            if DEBUG:
                                print(f"转换后(Gamma空间)平均RGB值: R={arr_gamma[:,:,0].mean():.2f}")
                        except Exception as e:
                            print(f"预处理颜色空间时出错: {e}")
                except Exception as e:
                    print(f"处理Base64图像数据时出错: {e}")
                    image = io.BytesIO()
            elif key == "delta":
                try:
                    delta_value = float(value)
                    if delta_value > 0:
                        delta = delta_value
                        if pipeline_object is not None:
                            pipeline_object.update_user_params(delta=delta_value)
                        if DEBUG:
                            print(f"设置Delta为: {delta}")
                except ValueError:
                    print(f"无效的Delta值: {value}")
            elif key == "do_add_noise":
                try:
                    do_add_noise = bool(int(value))
                    if DEBUG:
                        print(f"do_add_noise: {do_add_noise}")
                except ValueError:
                    print(f"do_add_noise error: {value}")
            elif key == "enable_similar_filter":
                try:
                    enable_similar_image_filter = bool(int(value))
                    if DEBUG:
                        print(f"enable_similar_filter: {enable_similar_image_filter}")
                except ValueError:
                    print(f"enable_similar_filter error: {value}")
            elif key == "similar_threshold":
                try:
                    threshold_value = float(value)
                    if 0 < threshold_value <= 1.0:
                        similar_image_filter_threshold = threshold_value
                        if DEBUG:
                            print(f"similar_threshold: {similar_image_filter_threshold}")
                except ValueError:
                    print(f"similar_threshold error: {value}")
            elif key == "max_skip_frame":
                try:
                    max_skip = int(value)
                    if max_skip > 0:
                        similar_image_filter_max_skip_frame = max_skip
                        if DEBUG:
                            print(f"max_skip_frame: {similar_image_filter_max_skip_frame}")
                except ValueError:
                    print(f"max_skip_frame error: {value}")
            elif key == "guidance_scale":
                try:
                    guidance_scale_value = float(value)
                    if guidance_scale_value > 0:
                        guidance_scale = guidance_scale_value
                        if pipeline_object is not None:
                            pipeline_object.update_user_params(guidance_scale=guidance_scale_value)
                        if DEBUG:
                            print(f"guidance_scale: {guidance_scale}")
                except ValueError:
                    print(f"guidance_scale error: {value}")
            elif key == "is_linear_space":
                try:
                    is_linear_value = value.decode('utf-8').lower()
                    is_linear_space = is_linear_value in ["true", "1", "yes", "y"]
                    if DEBUG:
                        print(f"Unity报告的颜色空间: {'线性空间' if is_linear_space else 'Gamma空间'}")
                except Exception as e:
                    print(f"解析颜色空间参数时出错: {e}")
                    is_linear_space = False
            # 파싱 루프에 추가
            elif key == "prompt_b64":
                try:
                    prompt = base64.b64decode(value).decode("utf-8", errors="replace")
                    if DEBUG:
                        print(f"Decoded prompt_b64 -> '{prompt[:80]}'")
                except Exception as e:
                    print(f"Failed to decode prompt_b64: {e}")

            elif key == "neg_prompt_b64":
                try:
                    neg_prompt = base64.b64decode(value).decode("utf-8", errors="replace")
                    if DEBUG:
                        print(f"Decoded neg_prompt_b64 -> '{neg_prompt[:80]}'")
                except Exception as e:
                    print(f"Failed to decode neg_prompt_b64: {e}")

            # 호환성: Unity가 현재 neg_prompt를 base64로 보내지만 키는 'neg_prompt'인 상태
            elif key == "neg_prompt":
                try:
                    # 먼저 base64로 시도
                    neg_prompt = base64.b64decode(value).decode("utf-8")
                except Exception:
                    # base64가 아니면 평문으로 처리
                    neg_prompt = value.decode("utf-8", errors="replace")

            else:
                if DEBUG:
                    print(f"Unknown data-buffer key: {key}")

        if seed == -1:
            if DEBUG:
                print("WARNING: No valid seed received! Using random seed.")
            seed = random.randint(0, 1000000)
            if DEBUG:
                print(f"Generated random seed: {seed}")
        
        if lora_scale is None:
            if DEBUG:
                print(f"WARNING: lora_scale is None, setting to default 0.85")
            lora_scale = 0.85
        elif not isinstance(lora_scale, (int, float)) or lora_scale <= 0 or lora_scale > 1:
            if DEBUG:
                print(f"WARNING: Invalid LoRA scale value: {lora_scale}, resetting to default 0.85")
            lora_scale = 0.85

        if DEBUG:
            effective_strength = strength if strength is not None else default_strength
            effective_delta = delta if delta is not None else 0.1
            effective_guidance = guidance_scale if guidance_scale is not None else 7.0
            print(f"Final parameters: Width={width}, Height={height}, Seed={seed}, Strength={effective_strength}, "
                  f"Delta={effective_delta}, Guidance={effective_guidance}, LoRA Scale={lora_scale}")
            
    except Exception as e:
        print(f"Error processing command: {e}")
        import traceback
        traceback.print_exc()
        try:
            client_socket.send(b"badreq")
        except:
            pass

def receiveCompleteData(client_socket):
    """안전하게 완전한 데이터 패킷 수신"""
    start_marker = b"|start|"
    end_marker = b"|end|"
    data_buffer = b""
    max_buffer_size = 10 * 1024 * 1024
    
    while True:
        try:
            chunk = client_socket.recv(4096)
            if not chunk:
                print("Connection closed by client")
                return b"", b""
            
            if DEBUG: 
                print(f"Received chunk: {len(chunk)} bytes")
            
            data_buffer += chunk
            
            if len(data_buffer) > max_buffer_size:
                print(f"WARNING: Buffer exceeded max size ({max_buffer_size} bytes)")
                data_buffer = data_buffer[-1024*1024:]
            
            start_idx = data_buffer.find(start_marker)
            end_idx = data_buffer.find(end_marker)
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                complete_data = data_buffer[start_idx : end_idx + len(end_marker)]
                remaining_data = data_buffer[end_idx + len(end_marker):]
                return complete_data, remaining_data

            if DEBUG:    
                print(f"Incomplete message in buffer ({len(data_buffer)} bytes), continuing...")
                
        except socket.timeout:
            print("Socket timeout while receiving data")
            return b"", data_buffer
        except socket.error as e:
            print(f"Socket error: {e}")
            return b"", b""
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return b"", b""

def startTcpServer(host='127.0.0.1', port=9999):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.settimeout(60.0)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)
    
    try:
        server_socket.bind((host, port))
        server_socket.listen(1)
        print(f"TCP Server started on {host}:{port}")
        
        while True:
            print("Waiting for a client connection...")
            try:
                client_socket, client_address = server_socket.accept()
                print(f"Client connected: {client_address}")
                
                client_socket.settimeout(60.0)
                client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)
                
                remaining_data = b""
                try:
                    while True:
                        complete_data, remaining_data = receiveCompleteData(client_socket)
                        if complete_data:
                            processData(client_socket, complete_data)
                        elif not remaining_data and not client_socket.fileno() == -1:
                            pass
                        else:
                            print("No complete data received")
                            break
                
                except ConnectionResetError:
                    print(f"Client {client_address} disconnected unexpectedly.")
                except socket.timeout:
                    print(f"Client {client_address} timed out.")
                except Exception as e:
                    print(f"Error handling client {client_address}: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    print(f"Closing connection to {client_address}")
                    try:
                        client_socket.close()
                    except:
                        pass
            except socket.timeout:
                print("Server socket timed out, continuing...")
            except Exception as e:
                print(f"Error accepting connection: {e}")
                import traceback
                traceback.print_exc()
    except KeyboardInterrupt:
        print("Server shutting down due to keyboard interrupt.")
    except Exception as e:
        print(f"Fatal server error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        print("Closing server socket.")
        try:
            server_socket.close()
        except:
            pass

if __name__ == "__main__":
    print("Starting image predictor with pattern recognition...")
    max_restarts = 5
    restart_count = 0
    
    while restart_count < max_restarts:
        try:
            print(f"Server start attempt #{restart_count+1}")
            startTcpServer()
        except KeyboardInterrupt:
            print("Server shutdown requested.")
            break
        except Exception as e:
            restart_count += 1
            print(f"Server crashed: {e}")
            import traceback
            traceback.print_exc()
            print(f"Restarting in 5 seconds... (attempt {restart_count}/{max_restarts})")
            import time
            time.sleep(5)
    
    print("Server shutdown complete.")