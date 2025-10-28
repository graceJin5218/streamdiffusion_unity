from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from safetensors.torch import load_file
import torch
import sys
import os
import shutil

# 사용법: python convert_model.py input.safetensors output_folder_name
if len(sys.argv) != 3:
    print("사용법: python convert_model.py input.safetensors output_folder_name")
    sys.exit(1)

input_file = sys.argv[1]
output_folder = sys.argv[2]

print(f"변환 중: {input_file} -> models/{output_folder}")

try:
    # 메모리 최적화 설정
    torch.cuda.empty_cache()  # GPU 메모리 정리
    
    # 모델 타입 자동 감지
    print("모델 타입 감지 중...")
    state_dict = load_file(input_file)
    
    # SDXL 감지: conditioner, text_encoder_2 관련 키 확인
    is_sdxl = any('conditioner' in key or 'text_model_2' in key 
                  for key in state_dict.keys())
    
    model_type = "SDXL" if is_sdxl else "SD 1.5/2.x"
    print(f"✓ 감지된 모델 타입: {model_type}")
    
    # 적절한 파이프라인 선택
    Pipeline = StableDiffusionXLPipeline if is_sdxl else StableDiffusionPipeline
    
    # 모델 로드 및 변환 (메모리 절약 모드)
    print("모델 로딩 중... (저메모리 모드)")
    pipeline = Pipeline.from_single_file(
        input_file,
        torch_dtype=torch.float16,      # 16비트로 메모리 절약
        use_safetensors=True,
        low_cpu_mem_usage=True,         # CPU 메모리 절약
        device_map="auto",              # 자동 디바이스 할당
        offload_folder="./temp_offload" # 디스크로 오프로드
    )

    # 저장
    output_path = f"models/{output_folder}"
    print(f"모델 저장 중: {output_path}")
    pipeline.save_pretrained(output_path, safe_serialization=True)

    print(f"✅ 변환 완료! {output_path}에 저장됨")
    print(f"Unity에서 _baseModelPath = \"{output_folder}\"로 설정하세요")
    
    # 임시 폴더 정리
    if os.path.exists("./temp_offload"):
        shutil.rmtree("./temp_offload")
        print("임시 파일 정리 완료")

except Exception as e:
    print(f"❌ 에러 발생: {e}")
    print("\n해결 방법:")
    print("1. 메모리가 부족할 수 있습니다 - 다른 프로그램을 종료하고 다시 시도")
    print("2. SDXL 모델은 더 많은 메모리가 필요합니다 (최소 16GB RAM 권장)")
    print("3. pip install --upgrade diffusers safetensors 로 라이브러리 업데이트")