using UnityEngine;
using Klak.Ndi;

public class NdiToQuad : MonoBehaviour
{
    [Header("NDI")]
    public NdiReceiver receiver;          // NdiReceiver ������Ʈ drag&drop
    [Header("Target")]
    public Renderer targetRenderer;       // Quad�� MeshRenderer
    public string textureProperty = "_MainTex"; // URP Unlit�̸� "_BaseMap"

    MaterialPropertyBlock _mpb;

    void Awake()
    {
        if (!receiver) receiver = FindObjectOfType<NdiReceiver>();
        if (!targetRenderer) targetRenderer = GetComponent<Renderer>();
        _mpb = new MaterialPropertyBlock();
    }

    void Update()
    {
        // ����/������ �����ϰ�: texture�� null�̸� ����(������ ����)
        var tex = receiver ? receiver.targetTexture : null;
        if (!tex || !targetRenderer) return;

        targetRenderer.GetPropertyBlock(_mpb);
        _mpb.SetTexture(textureProperty, tex);
        targetRenderer.SetPropertyBlock(_mpb);
    }
}
