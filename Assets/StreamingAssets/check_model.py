import torch
import sys

def check_model_info(model_path):
    """모델 파일의 정보를 확인합니다."""
    try:
        print(f"모델 파일 확인 중: {model_path}")
        
        # SafeTensors 파일 로드
        from safetensors import safe_open
        
        print("\n=== 모델 정보 ===")
        
        # 파일 크기 확인
        import os
        file_size = os.path.getsize(model_path) / (1024**3)  # GB
        print(f"파일 크기: {file_size:.2f} GB")
        
        # 텐서 키 확인
        with safe_open(model_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            key_list = list(keys)
            
            print(f"총 텐서 개수: {len(key_list)}")
            print("\n첫 10개 키:")
            for i, key in enumerate(key_list[:10]):
                print(f"  {i+1}. {key}")
            
            # 모델 타입 추정
            if any("conditioner" in key for key in key_list):
                model_type = "SDXL"
            elif any("cond_stage_model" in key for key in key_list):
                model_type = "SD 1.x/2.x (원본 체크포인트)"
            elif any("text_encoder" in key for key in key_list):
                model_type = "Diffusers 형식"
            else:
                model_type = "알 수 없음 또는 커스텀"
                
            print(f"\n추정 모델 타입: {model_type}")
            
            # UNet 관련 키 확인
            unet_keys = [key for key in key_list if "model.diffusion_model" in key or "unet" in key]
            if unet_keys:
                print(f"UNet 키 개수: {len(unet_keys)}")
                print("UNet 키 예시:")
                for key in unet_keys[:3]:
                    print(f"  - {key}")
            
            return model_type
            
    except Exception as e:
        print(f"❌ 에러: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python check_model.py model.safetensors")
        sys.exit(1)
    
    model_path = sys.argv[1]
    check_model_info(model_path)