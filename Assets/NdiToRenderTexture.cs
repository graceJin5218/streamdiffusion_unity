using UnityEngine;
#if KLAK_NDI
using Klak.Ndi; // KlakNDI ��Ű���� ����ϴ� ��� �ʿ�
#endif

/// <summary>
/// NDI Receiver�� ������ GPU �ؽ�ó�� �� ������ RenderTexture�� �����մϴ�.
/// - ���� ����: KlakNDI�� NdiReceiver�� ���� �ΰ�, �� ��ũ��Ʈ�� ���� ����
/// - targetRT�� TestStreamUI._ndiRenderTexture�� �����ϸ� TestStreamUI�� �� RT�� �״�� ����մϴ�.
/// </summary>
public class NdiToRenderTexture : MonoBehaviour
{
#if KLAK_NDI
    [Header("NDI Receiver (KlakNDI)")]
    [Tooltip("KlakNDI�� NdiReceiver ������Ʈ")] public NdiReceiver receiver;
#else
    [Header("��ü �Է� (NDI�� ��Ƽ���� ����ϴ� ���)")]
    [Tooltip("NDI ����� ���� Renderer�� ��Ƽ���� (KlakNDI �̻�� �� ��ü)")] public Material inputMaterial;
#endif

    [Header("��� RenderTexture")]
    [Tooltip("NDI �������� ������ ���� ������ RenderTexture")] public RenderTexture targetRT;
    [Tooltip("�ҽ� ũ�⿡ ���� RT�� �ڵ� ��������")] public bool autoResizeToSource = true;
    [Tooltip("�ҽ� �������� ���� �ø��� �ʿ��� �� üũ")] public bool flipVertical = false;

    [Header("RT ���� �ɼ� (autoResizeToSource=true �� �� ���)")]
    public RenderTextureFormat rtFormat = RenderTextureFormat.ARGB32;
    public int rtDepth = 0;

    [Header("�����(����)")]
    [Tooltip("����� ����� ������ �̸� ������ ���� (RawImage/Renderer�� ��Ƽ���� ��)")]
    public Material debugOutputMaterial;

    // ����: �ø��� ��Ƽ���� (�ʿ� �� ���� ����)
    static Material _flipMat;

    void OnEnable()
    {
        Application.runInBackground = true; // ������/���� ��Ŀ�� ��� ����
    }

    void Update()
    {
        var src = GetSourceTexture();
        if (src == null) return;

        EnsureTargetRT(src.width, src.height);
        if (targetRT == null) return;

        // ���� �ø��� �ʿ��ϸ� �ø��� ��Ƽ����� Blit
        if (flipVertical)
        {
            if (_flipMat == null)
            {
                // �ſ� �ܼ��� �ø� ���̴�. ������Ʈ�� �Ʒ� ���̴� ����(Shader "Hidden/NdiCopyFlip")�� �߰��ؾ� �մϴ�.
                _flipMat = new Material(Shader.Find("Hidden/NdiCopyFlip"));
            }
            if (_flipMat != null)
            {
                Graphics.Blit(src, targetRT, _flipMat, 0);
            }
            else
            {
                // ���̴��� ã�� ���� ��� �ø� ���� ����
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
        var tex = receiver.targetTexture; // KlakNDI: ���ŵ� External Texture
        if (tex == null)
        {
            // ���� ��� ���̰ų� �ҽ� �̼���
            return null;
        }
        return tex;
#else
        if (inputMaterial == null)
        {
            Debug.LogWarning("[NdiToRenderTexture] inputMaterial not assigned (KlakNDI �̻�� ���).");
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

        if (!autoResizeToSource) // �ڵ� �������� ��Ȱ��ȭ ��, ������ ����
        {
            if (!targetRT.IsCreated()) targetRT.Create();
            return;
        }

        if (targetRT.width != w || targetRT.height != h || targetRT.format != rtFormat)
        {
            // RT �����
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
