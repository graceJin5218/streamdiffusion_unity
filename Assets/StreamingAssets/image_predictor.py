# image_predictor.py (patched full file)

import sys, os, io, json, base64
from typing import Optional, Literal, Dict, Tuple
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import cv2

# 성능 옵션
torch.backends.cudnn.benchmark = True

DEBUG = False

# ────────────────────────────────────────────────────────────────────────────────
# 유틸
# ────────────────────────────────────────────────────────────────────────────────

def make_solid_rgb(h: int, w: int, rgb: Tuple[int, int, int]) -> np.ndarray:
    """Create solid (H,W,3) RGB uint8 contiguous image."""
    r, g, b = map(int, rgb)
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[..., 0] = r
    img[..., 1] = g
    img[..., 2] = b
    return np.require(img, dtype=np.uint8, requirements=["C"])

def parse_pipe_kv(cmd: str):
    """
    파이프 구분자 기반 커맨드 파서
    예: "|start|command||advance||key||val||...|end|"
       → ("advance", {"key":"val", ...})
    """
    parts = [p for p in cmd.split("|") if p]
    if len(parts) < 3 or parts[0] != "start" or parts[1] != "command":
        return None, {}
    kind = parts[2]
    kv = {}
    i = 3
    while i + 1 < len(parts):
        k = parts[i]
        v = parts[i + 1]
        kv[k] = v
        i += 2
        if k == "end":
            break
    return kind, kv

# ────────────────────────────────────────────────────────────────────────────────
# 모델 경로/래퍼 로딩
# ────────────────────────────────────────────────────────────────────────────────

# 현재 파일 기준으로 streamdiffusion 패키지 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "streamdiffusion"))
from utils.wrapper import StreamDiffusionWrapper  # noqa

base_dir = os.path.dirname(os.path.abspath(__file__))

# 기본 모델/경로(유니티에서 setModelPaths로 재지정됨)
base_model = ""
taesd_model = "madebyollin/taesd"
lora_model = None
lora_model2 = None

# 파이프라인 전역/기본값
default_strength = 9.0  # ⬅ Unity 값 들어오면 런타임 갱신

# 동작 플래그
bypass_mode = False
is_linear_space = False

# ────────────────────────────────────────────────────────────────────────────────
# 패턴 분석 (필요 시 간소화 가능)
# ────────────────────────────────────────────────────────────────────────────────

def analyze_image_pattern(image_array: np.ndarray) -> float:
    """이미지의 의미 있는 패턴 점수(0..1) 계산. (간소화된 버전)"""
    if image_array is None or image_array.size == 0:
        return 0.0

    # RGB/GRAY 정규화
    if image_array.ndim == 3:
        if image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]
        if image_array.shape[2] != 3:
            return 0.0
        gray = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    elif image_array.ndim == 2:
        gray = image_array.astype(np.uint8)
    else:
        return 0.0

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float((edges > 0).sum()) / float(edges.size)

    drawn_density = float((gray < 250).sum()) / float(gray.size)
    score = 0.0

    if 0.01 < edge_density < 0.3:
        score += min(edge_density * 3.0, 1.0) * 0.5
    if 0.02 < drawn_density < 0.5:
        score += drawn_density * 0.5

    return float(np.clip(score, 0.0, 1.0))

# ────────────────────────────────────────────────────────────────────────────────
# 파이프라인
# ────────────────────────────────────────────────────────────────────────────────

class Pipeline:
    def __init__(
        self,
        w: int,
        h: int,
        seed: int,
        device: torch.device,
        torch_dtype: torch.dtype,
        use_vae: bool,
        use_lora: bool,
        gc_mode: Literal["img2img", "txt2img", "stream"],
        acc_mode: str,
        positive_prompt: str,
        negative_prompt: str = "",
        preloaded_pipe=None,
        model_path: Optional[str] = None,
        lora_dict: Optional[Dict[str, float]] = None,
        cfg_type: str = "none",
        delta: float = 0.1,
        do_add_noise: bool = True,
        enable_similar_image_filter: bool = True,
        similar_image_filter_threshold: float = 0.2,
        similar_image_filter_max_skip_frame: int = 10,
    ):
        actual_model_path = model_path if model_path else base_model
        print(f"Initializing pipeline with model path: {actual_model_path}")

        # ----- 런타임 파라미터(상태) -----
        self.delta = float(delta)
        self.cfg_type = cfg_type
        self.guidance_scale = 7.0   # ⬅ 준비/예측 시 런타임 갱신
        # Lora scale 들은 wrapper 내부에서 관리 (필요시 setter 추가)

        # 기타 옵션
        self.do_add_noise = do_add_noise
        self.enable_similar_image_filter = enable_similar_image_filter
        self.similar_image_filter_threshold = similar_image_filter_threshold
        self.similar_image_filter_max_skip_frame = similar_image_filter_max_skip_frame

        # 시각/스타일 관련(원래 코드의 다양한 파라미터 유지 가능)
        self._ghost_bg_rgb = (10, 10, 10)

        if not actual_model_path or (os.path.exists(actual_model_path) is False):
            print(f"[WARN] Model path not set or not found: {actual_model_path}")

        vae_path = None
        if use_vae:
            vae_path = "madebyollin/taesd"
            if taesd_model and os.path.exists(str(taesd_model)) and os.path.isdir(str(taesd_model)):
                print(f"Using local VAE folder: {taesd_model}")
                vae_path = taesd_model

        if use_lora and lora_dict is not None:
            print(f"LoRA will be used: {lora_dict}")

        # StreamDiffusionWrapper 초기화
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
            output_type="pil",
        )

        # prepare 버킷 캐시
        self._last_detail_bucket = None
        self._last_prompt_cache = ""
        self._last_negative_cache = ""

    # ──────────────────────────────────────────────────────────────────────
    # PATCH: Unity 런타임 파라미터 적용(핵심)
    # ──────────────────────────────────────────────────────────────────────
    def apply_runtime_params(
        self,
        strength: Optional[float] = None,
        delta: Optional[float] = None,
        guidance_scale: Optional[float] = None,
    ):
        """Unity에서 보낸 실시간 파라미터를 상태에 적용."""
        global default_strength
        if strength is not None:
            try:
                default_strength = float(strength)
            except Exception:
                pass
        if delta is not None:
            try:
                self.delta = float(delta)
            except Exception:
                pass
        if guidance_scale is not None:
            try:
                self.guidance_scale = float(guidance_scale)
            except Exception:
                pass
        print(
            f"[apply_runtime_params] strength={default_strength}, "
            f"delta={self.delta}, guidance_scale={self.guidance_scale}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # PATCH: prepare 하드코딩 제거
    # ──────────────────────────────────────────────────────────────────────
    def prepare(
        self,
        prompt: str,
        negative_prompt: str,
        target_guidance_scale: Optional[float] = None,
        strength: Optional[float] = None,
        delta: Optional[float] = None,
        num_inference_steps: Optional[int] = 18,
    ):
        try:
            # 적용치 계산: 인자 > 현재 상태 > 글로벌
            g = (
                float(target_guidance_scale)
                if target_guidance_scale is not None
                else float(self.guidance_scale)
            )
            d = float(delta) if delta is not None else float(getattr(self, "delta", 0.8))
            global default_strength
            s = float(strength) if strength is not None else float(default_strength)

            print(f"[prepare] prompt='{prompt[:80]}', neg='{negative_prompt[:80]}'")
            print(f"[prepare] 적용 인자: guidance={g}, delta={d}, strength={s}, steps={num_inference_steps}")

            # wrapper.prepare 호출 (wrapper가 strength를 직접 받지 않으면, 내부 로직에서 사용)
            self.stream.prepare(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=g,
                delta=d,
                num_inference_steps=num_inference_steps or 18,
            )

            # 스트림 옵션(유지)
            if hasattr(self.stream, "stream"):
                if hasattr(self.stream.stream, "do_add_noise"):
                    self.stream.stream.do_add_noise = bool(self.do_add_noise)
                if hasattr(self.stream.stream, "enable_similar_image_filter"):
                    self.stream.stream.enable_similar_image_filter = bool(self.enable_similar_image_filter)
                if hasattr(self.stream.stream, "similar_image_filter_threshold"):
                    self.stream.stream.similar_image_filter_threshold = float(self.similar_image_filter_threshold)
                if hasattr(self.stream.stream, "similar_image_filter_max_skip_frame"):
                    self.stream.stream.similar_image_filter_max_skip_frame = int(self.similar_image_filter_max_skip_frame)

            print("[prepare] 준비 완료")
        except Exception as e:
            print(f"[prepare] 오류: {e}")
            import traceback
            traceback.print_exc()

    # ──────────────────────────────────────────────────────────────────────
    # PATCH: predict가 Unity 런타임 파라미터를 받도록 확장
    # ──────────────────────────────────────────────────────────────────────
    def predict(
        self,
        input_image: Image.Image,
        new_prompt: Optional[str] = None,
        pattern_score: Optional[float] = None,
        strength: Optional[float] = None,
        delta: Optional[float] = None,
        guidance_scale: Optional[float] = None,
    ) -> Image.Image:
        try:
            # 안전한 RGB
            if input_image.mode != "RGB":
                input_image = input_image.convert("RGB")

            # Unity 런타임 파라미터 즉시 반영
            self.apply_runtime_params(strength=strength, delta=delta, guidance_scale=guidance_scale)

            # 패턴 점수(선택)
            if pattern_score is None:
                h, w = input_image.height, input_image.width
                scale = 256 / max(h, w)
                arr_small = np.array(
                    input_image if scale >= 1.0 else input_image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
                )
                pattern_score = analyze_image_pattern(arr_small)

            # 텐서 변환
            transform = T.Compose([T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            image_tensor = transform(input_image).unsqueeze(0)
            if torch.cuda.is_available():
                image_tensor = image_tensor.to(device="cuda", dtype=torch.float16)
            else:
                image_tensor = image_tensor.to(self.stream.device if hasattr(self, "stream") else "cpu")

            # 프롬프트 업데이트
            try:
                if hasattr(self.stream, "stream") and hasattr(self.stream.stream, "update_prompt"):
                    self.stream.stream.update_prompt(new_prompt or "")
                # prepare용 캐시
                if new_prompt is not None:
                    self._last_prompt_cache = new_prompt
            except Exception as e:
                print(f"[predict] prompt update failed: {e}")

            # 추론
            try:
                gen_out = self.stream(image=image_tensor, prompt=new_prompt)
            except Exception as e:
                print(f"[predict] inference error: {e}")
                gen_out = None

            # 출력 PIL
            if isinstance(gen_out, torch.Tensor):
                t = gen_out
                if t.dim() == 4:
                    t = t.squeeze(0)
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

            return gen_pil
        except Exception as e:
            print(f"[predict] error: {e}")
            import traceback
            traceback.print_exc()
            # 실패 시 입력 그대로 반환(흰 화면 방지)
            return input_image.copy()

# ────────────────────────────────────────────────────────────────────────────────
# (선택) advance 커맨드 처리 예시 함수
# ────────────────────────────────────────────────────────────────────────────────

def handle_advance_command(pipeline: Pipeline, cmd: str):
    """
    서버 루프에서 수신한 파이프 커맨드(cmd)를 파싱/적용하고
    (이미지는 별도 경로로 수신해 PIL로 변환했다고 가정) 예측을 수행하는 예시.
    프로젝트의 실제 서버 루프에 맞게 수정/호출하세요.
    """
    kind, kv = parse_pipe_kv(cmd)
    if kind != "advance":
        return None, {"ok": False, "error": "not_advance"}

    trace_id = kv.get("trace_id")
    prompt_b64 = kv.get("prompt_b64", "")
    prompt = ""
    try:
        prompt = base64.b64decode(prompt_b64.encode("utf-8")).decode("utf-8", errors="ignore")
    except Exception:
        pass

    def fget(name):
        s = kv.get(name)
        if s is None:
            return None
        try:
            return float(s)
        except Exception:
            return None

    s = fget("strength")
    d = fget("delta")
    g = fget("guidance_scale")

    # 이미지(Base64) 포함으로 오는 경우 (너의 프로토콜에서는 포함됨)
    pil_in = None
    img_b64 = kv.get("image_base64")
    if img_b64:
        try:
            pil_in = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
        except Exception as e:
            print(f"[advance] image decode error: {e}")

    if pil_in is None:
        return None, {"ok": False, "trace_id": trace_id, "error": "no_image"}

    # 런타임 파라미터 적용 → 예측
    pipeline.apply_runtime_params(strength=s, delta=d, guidance_scale=g)
    out_pil = pipeline.predict(pil_in, new_prompt=prompt, strength=s, delta=d, guidance_scale=g)

    # Unity로 회신할 메타(이미지 전송은 프로젝트 기존 방식대로)
    meta = {
        "ok": out_pil is not None,
        "trace_id": trace_id,
        "applied_params": {"strength": s, "delta": d, "guidance_scale": g},
    }
    return out_pil, meta
