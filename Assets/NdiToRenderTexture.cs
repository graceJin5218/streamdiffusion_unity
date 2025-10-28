using UnityEngine;
#if KLAK_NDI
using Klak.Ndi; // KlakNDI 패키지를 사용하는 경우 필요
#endif

/// <summary>
/// NDI Receiver가 수신한 GPU 텍스처를 매 프레임 RenderTexture로 복사합니다.
/// - 권장 사용법: KlakNDI의 NdiReceiver를 씬에 두고, 이 스크립트에 참조 연결
/// - targetRT를 TestStreamUI._ndiRenderTexture에 연결하면 TestStreamUI가 이 RT를 그대로 사용합니다.
/// </summary>
public class NdiToRenderTexture : MonoBehaviour
{
#if KLAK_NDI
    [Header("NDI Receiver (KlakNDI)")]
    [Tooltip("KlakNDI의 NdiReceiver 컴포넌트")] public NdiReceiver receiver;
#else
    [Header("대체 입력 (NDI를 머티리얼에 출력하는 경우)")]
    [Tooltip("NDI 출력이 들어가는 Renderer의 머티리얼 (KlakNDI 미사용 시 대체)")] public Material inputMaterial;
#endif

    [Header("출력 RenderTexture")]
    [Tooltip("NDI 프레임을 복사해 넣을 목적지 RenderTexture")] public RenderTexture targetRT;
    [Tooltip("소스 크기에 맞춰 RT를 자동 리사이즈")] public bool autoResizeToSource = true;
    [Tooltip("소스 프레임의 수직 플립이 필요할 때 체크")] public bool flipVertical = false;

    [Header("RT 생성 옵션 (autoResizeToSource=true 일 때 사용)")]
    public RenderTextureFormat rtFormat = RenderTextureFormat.ARGB32;
    public int rtDepth = 0;

    [Header("디버그(선택)")]
    [Tooltip("복사된 결과를 씬에서 미리 보려면 연결 (RawImage/Renderer의 머티리얼 등)")]
    public Material debugOutputMaterial;

    // 내부: 플립용 머티리얼 (필요 시 동적 생성)
    static Material _flipMat;

    void OnEnable()
    {
        Application.runInBackground = true; // 에디터/빌드 포커스 없어도 동작
    }

    void Update()
    {
        var src = GetSourceTexture();
        if (src == null) return;

        EnsureTargetRT(src.width, src.height);
        if (targetRT == null) return;

        // 수직 플립이 필요하면 플립용 머티리얼로 Blit
        if (flipVertical)
        {
            if (_flipMat == null)
            {
                // 매우 단순한 플립 셰이더. 프로젝트에 아래 셰이더 파일(Shader "Hidden/NdiCopyFlip")을 추가해야 합니다.
                _flipMat = new Material(Shader.Find("Hidden/NdiCopyFlip"));
            }
            if (_flipMat != null)
            {
                Graphics.Blit(src, targetRT, _flipMat, 0);
            }
            else
            {
                // 셰이더를 찾지 못한 경우 플립 없이 복사
                Graphics.Blit(src, targetRT);
            }
        }
        else
        {
            Graphics.Blit(src, targetRT);
        }

        if (debugOutputMaterial != null)
        {
            debugOutputMaterial.mainTexture = targetRT;
        }
    }

    Texture GetSourceTexture()
    {
#if KLAK_NDI
        if (receiver == null)
        {
            Debug.LogWarning("[NdiToRenderTexture] NdiReceiver not assigned.");
            return null;
        }
        var tex = receiver.targetTexture; // KlakNDI: 수신된 External Texture
        if (tex == null)
        {
            // 수신 대기 중이거나 소스 미선택
            return null;
        }
        return tex;
#else
        if (inputMaterial == null)
        {
            Debug.LogWarning("[NdiToRenderTexture] inputMaterial not assigned (KlakNDI 미사용 경로).");
            return null;
        }
        return inputMaterial.mainTexture;
#endif
    }

    void EnsureTargetRT(int w, int h)
    {
        if (targetRT == null)
        {
            Debug.LogWarning("[NdiToRenderTexture] targetRT is null.");
            return;
        }

        if (!autoResizeToSource) // 자동 리사이즈 비활성화 시, 생성만 보장
        {
            if (!targetRT.IsCreated()) targetRT.Create();
            return;
        }

        if (targetRT.width != w || targetRT.height != h || targetRT.format != rtFormat)
        {
            // RT 재생성
            if (targetRT.IsCreated()) targetRT.Release();

            targetRT.width = w;
            targetRT.height = h;
            targetRT.format = rtFormat;
            targetRT.depth = rtDepth;
            targetRT.enableRandomWrite = false;
            targetRT.useMipMap = false;
            targetRT.autoGenerateMips = false;
            targetRT.Create();
        }
        else if (!targetRT.IsCreated())
        {
            targetRT.Create();
        }
    }
}
