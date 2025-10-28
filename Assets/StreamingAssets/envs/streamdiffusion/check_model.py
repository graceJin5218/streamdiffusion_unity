import os
import sys

def check_file_info(file_path):
    """파일 기본 정보만 확인"""
    try:
        if not os.path.exists(file_path):
            print(f"❌ 파일이 존재하지 않습니다: {file_path}")
            return
        
        # 파일 크기 확인
        file_size_bytes = os.path.getsize(file_path)
        file_size_gb = file_size_bytes / (1024**3)
        file_size_mb = file_size_bytes / (1024**2)
        
        print(f"📁 파일: {os.path.basename(file_path)}")
        print(f"📏 크기: {file_size_gb:.2f} GB ({file_size_mb:.0f} MB)")
        
        # 크기로 모델 타입 추정
        if file_size_gb > 6.0:
            estimated_type = "SDXL (또는 대형 모델)"
            compatibility = "❌ StreamDiffusion과 호환 어려움"
        elif 2.0 <= file_size_gb <= 4.0:
            estimated_type = "SD 1.5 기반 모델"
            compatibility = "✅ StreamDiffusion과 호환 가능성 높음"
        elif file_size_gb < 2.0:
            estimated_type = "경량화 모델 또는 LoRA"
            compatibility = "⚠️ 확인 필요"
        else:
            estimated_type = "알 수 없음"
            compatibility = "⚠️ 확인 필요"
        
        print(f"🔍 추정 타입: {estimated_type}")
        print(f"🔗 호환성: {compatibility}")
        
        # 파일 확장자 확인
        if file_path.endswith('.safetensors'):
            print("✅ SafeTensors 형식 (안전함)")
        elif file_path.endswith('.ckpt'):
            print("⚠️ Checkpoint 형식 (보안 주의)")
        
        return file_size_gb
        
    except Exception as e:
        print(f"❌ 에러: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python simple_check.py model.safetensors")
        sys.exit(1)
    
    file_path = sys.argv[1]
    file_size = check_file_info(file_path)
    
    print("\n" + "="*50)
    
    if file_size and file_size > 6.0:
        print("💡 제안: 이 모델은 SDXL일 가능성이 높습니다.")
        print("   StreamDiffusion은 SD 1.5 기반 모델을 권장합니다.")
        print("   더 작은 크기(2-4GB)의 SD 1.5 모델을 시도해보세요.")
        print("\n   추천 모델:")
        print("   - Anything V5 (2-3GB)")
        print("   - Dreamshaper V8 (2-4GB)")
        print("   - Realistic Vision V6 (2-4GB)")
    elif file_size and 2.0 <= file_size <= 4.0:
        print("💡 이 모델은 변환 가능할 것 같습니다!")
        print("   다른 변환 방법을 시도해보겠습니다.")
    else:
        print("💡 파일 크기가 예상과 다릅니다.")
        print("   모델 정보를 다시 확인해보세요.")