using UnityEngine;
using Klak.Ndi;

public class NdiToQuad : MonoBehaviour
{
    [Header("NDI")]
    public NdiReceiver receiver;          // NdiReceiver 컴포넌트 drag&drop
    [Header("Target")]
    public Renderer targetRenderer;       // Quad의 MeshRenderer
    public string textureProperty = "_MainTex"; // URP Unlit이면 "_BaseMap"

    MaterialPropertyBlock _mpb;

    void Awake()
    {
        if (!receiver) receiver = FindObjectOfType<NdiReceiver>();
        if (!targetRenderer) targetRenderer = GetComponent<Renderer>();
        _mpb = new MaterialPropertyBlock();
    }

    void Update()
    {
        // 연결/해제에 안전하게: texture가 null이면 유지(깜박임 방지)
        var tex = receiver ? receiver.targetTexture : null;
        if (!tex || !targetRenderer) return;

        targetRenderer.GetPropertyBlock(_mpb);
        _mpb.SetTexture(textureProperty, tex);
        targetRenderer.SetPropertyBlock(_mpb);
    }
}
