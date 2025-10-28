using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

/// <summary>
/// TestStreamUI (NDI 입력 전용)
/// - NDI Receiver가 갱신하는 텍스처를 입력으로 받아서 StreamDiffusionClient로 전송
/// - 입력 경로 2가지 중 하나를 사용:
///   1) _ndiRenderTexture: NDI Receiver의 Target Texture로 지정한 RenderTexture (권장)
///   2) _inputMaterial.mainTexture: Renderer/RawImage 등에 표시되는 머티리얼의 mainTexture
/// - 전송 직전에 GPU 텍스처를 CPU에서 읽을 수 있는 Texture2D(RGBA32)로 캡처하여 AdvancePipeline에 전달
/// - 기존 워크플로 호환을 위해 StartStreamDiff(), StartStreamdiff() 둘 다 제공
/// </summary>
public class TestStreamUI : MonoBehaviour
{
    [Header("StreamDiffusion")]
    [Tooltip("StreamDiffusionClient 참조")]
    public StreamDiffusionClient _stream;

    [Tooltip("실행 시 자동으로 파이프라인 시작")]
    [SerializeField] private bool _autoStart = false;

    [Tooltip("연속 전송 모드 (주기적으로 프레임 전송)")]
    public bool _continuousGeneration = true;

    [Tooltip("연속 전송 간격(초). 0.10~0.25 권장")]
    [Range(0.02f, 1.0f)] public float _generationInterval = 0.15f;

    [Header("입력 (NDI)")]
    [Tooltip("NDI Receiver의 Target Texture로 지정한 RenderTexture (권장)")]
    public RenderTexture _ndiRenderTexture;

    [Tooltip("NDI 프레임이 표시되는 Material. (선택) _ndiRenderTexture가 없을 때 사용")]
    public Material _inputMaterial;

    [Tooltip("StreamDiffusion 모델 입력 크기에 맞춰 캡처 이미지를 리사이즈")]
    public bool _resizeToModelSize = true;

    [Header("UI (선택)")]
    public TMP_InputField _promptInput;          // 프롬프트 입력
    public Toggle _continuousToggle;              // 연속 모드 토글
    public Button _generateOnceButton;            // 1회 전송 버튼
    public Button _startButton;                  // 시작 버튼 (선택)
    public TMP_Text _statusText;                  // 상태 텍스트

    // 내부 상태
    private Coroutine _continuousGenerationCoroutine;
    private Texture2D _capturedTex2D;             // CPU 읽기용 재사용 버퍼
    private Texture _originalInputTexture;        // 입력 머티리얼 원본 mainTexture 백업
    private bool _wasContinuous;

    // 마지막으로 캡처해 보낸 Texture2D (디버그/검사용)
    private Texture2D _lastSentTex2D;


    // ─────────────────────────────────────────────────────────────────────────────
    // Unity lifecycle
    // ─────────────────────────────────────────────────────────────────────────────
    private void Awake()
    {
        if (_inputMaterial != null)
            _originalInputTexture = _inputMaterial.mainTexture;
    }

    private void OnEnable()
    {
        if (_generateOnceButton != null) _generateOnceButton.onClick.AddListener(OnClickGenerateOnce);
        if (_continuousToggle != null) _continuousToggle.onValueChanged.AddListener(OnChangeContinuous);
        if (_startButton != null) _startButton.onClick.AddListener(OnClickStart);
    }

    private void Start()
    {
        // UI 초기 상태 반영
        if (_continuousToggle != null) _continuousToggle.isOn = _continuousGeneration;
        _wasContinuous = _continuousGeneration;

        if (_autoStart)
            StartStreamDiff();

        // 연속 루틴 시작
        if (_continuousGeneration)
            _continuousGenerationCoroutine = StartCoroutine(ContinuousGenerationRoutine());
    }

    private void OnDisable()
    {
        if (_generateOnceButton != null) _generateOnceButton.onClick.RemoveListener(OnClickGenerateOnce);
        if (_continuousToggle != null) _continuousToggle.onValueChanged.RemoveListener(OnChangeContinuous);
        if (_startButton != null) _startButton.onClick.RemoveListener(OnClickStart);
    }

    private void OnDestroy()
    {
        // 입력 머티리얼 원복(선택)
        if (_inputMaterial != null && _originalInputTexture != null)
            _inputMaterial.mainTexture = _originalInputTexture;

        if (_capturedTex2D != null)
        {
            Destroy(_capturedTex2D);
            _capturedTex2D = null;
        }
    }

    private void Update()
    {
        // 런타임에 연속 모드 토글 변화 감지하여 코루틴 스타트/스톱
        if (_continuousGeneration != _wasContinuous)
        {
            ToggleContinuous(_continuousGeneration);
            _wasContinuous = _continuousGeneration;
        }

        // 상태 출력(선택)
        if (_statusText != null && _stream != null)
        {
            var running = _stream.isRunning();
            var generating = _stream.isGenerating();
            _statusText.text = $"Running: {running} | Generating: {generating} | Mode: {(_continuousGeneration ? "CONT" : "ONCE")}";
        }

    }

    // ─────────────────────────────────────────────────────────────────────────────
    // UI: Start button & pipeline start
    // ─────────────────────────────────────────────────────────────────────────────
    private void OnClickStart()
    {
        StartStreamDiff();
    }

    /// <summary>
    /// StreamDiffusion 파이프라인 시작(로드)
    /// </summary>
    public void StartStreamDiff()
    {
        if (_stream == null)
        {
            Debug.LogWarning("[TestStreamUI] _stream is null");
            return;
        }

        if (_startButton != null) _startButton.interactable = false;

        if (_stream.isValid() && !_stream.isRunning())
        {
            _stream.LoadPipeline();
        }
        else
        {
            Debug.Log("[TestStreamUI] Stream already running or invalid.");
        }
    }

    /// <summary>
    /// 기존 프로젝트 호환용: 소문자 d 버전도 제공
    /// </summary>
    public void StartStreamdiff()
    {
        StartStreamDiff();
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // UI Handlers
    // ─────────────────────────────────────────────────────────────────────────────
    private void OnClickGenerateOnce()
    {
        if (_stream == null) { Debug.LogWarning("[TestStreamUI] _stream is null"); return; }
        if (!_stream.isRunning()) { Debug.LogWarning("[TestStreamUI] Stream is not running"); return; }
        if (_stream.isGenerating()) { Debug.Log("[TestStreamUI] Busy generating; skip"); return; }

        UpdateStreamDiff();
    }

    private void OnChangeContinuous(bool on)
    {
        _continuousGeneration = on;
    }

    private void ToggleContinuous(bool on)
    {
        if (on)
        {
            if (_continuousGenerationCoroutine == null)
                _continuousGenerationCoroutine = StartCoroutine(ContinuousGenerationRoutine());
        }
        else
        {
            if (_continuousGenerationCoroutine != null)
            {
                StopCoroutine(_continuousGenerationCoroutine);
                _continuousGenerationCoroutine = null;
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Core: Capture & Send
    // ─────────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// NDI 입력(RenderTexture 우선) 또는 입력 머티리얼의 mainTexture에서 현재 프레임을 캡처하여 RGBA32 Texture2D로 반환
    /// 필요 시 모델 입력 크기(_stream._width,_stream._height)로 리사이즈
    /// </summary>
    private Texture2D CaptureFromMaterialOrRT()
    {
        // 1) RenderTexture 우선 (권장)
        Texture src = _ndiRenderTexture != null ? (Texture)_ndiRenderTexture
                                                : (_inputMaterial != null ? _inputMaterial.mainTexture : null);
        if (src == null)
        {
            Debug.LogWarning("[TestStreamUI] Capture source is null. Assign _ndiRenderTexture or _inputMaterial.");
            return null;
        }

        int w = src.width;
        int h = src.height;

        if (_resizeToModelSize && _stream != null)
        {
            // 모델 입력 크기에 맞춰 리사이즈
            if (_stream._width > 0 && _stream._height > 0)
            {
                w = _stream._width;
                h = _stream._height;
            }
        }

        // 소스를 RenderTexture로 확보
        RenderTexture rt;
        bool needTemp = false;

        if (src is RenderTexture srcRT)
        {
            if (srcRT.width != w || srcRT.height != h)
            {
                rt = RenderTexture.GetTemporary(w, h, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Default);
                Graphics.Blit(srcRT, rt);
                needTemp = true;
            }
            else
            {
                rt = srcRT;
            }
        }
        else
        {
            // Texture2D/외부 Texture → 임시 RT로 Blit
            rt = RenderTexture.GetTemporary(w, h, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Default);
            Graphics.Blit(src, rt);
            needTemp = true;
        }

        // CPU 읽기용 Texture2D 버퍼 준비
        bool linear = QualitySettings.activeColorSpace == ColorSpace.Linear;
        if (_capturedTex2D == null || _capturedTex2D.width != w || _capturedTex2D.height != h)
        {
            if (_capturedTex2D != null) Destroy(_capturedTex2D);
            _capturedTex2D = new Texture2D(w, h, TextureFormat.RGBA32, false, linear);
        }

        // ReadPixels
        var prev = RenderTexture.active;
        RenderTexture.active = rt;
        _capturedTex2D.ReadPixels(new Rect(0, 0, w, h), 0, 0);
        _capturedTex2D.Apply(false, false);
        RenderTexture.active = prev;

        if (needTemp)
            RenderTexture.ReleaseTemporary(rt);

        return _capturedTex2D;
    }

    /// <summary>
    /// 현재 프레임을 캡처하여 바로 StreamDiffusionClient로 전송
    /// </summary>
    public void UpdateStreamDiff()
    {
        if (_stream == null) return;
        if (!_stream.isRunning() || _stream.isGenerating()) return;

        var tex = CaptureFromMaterialOrRT();
        if (tex == null) return;

        _lastSentTex2D = tex; // 디버그용 보관

        string prompt = _promptInput != null ? _promptInput.text : string.Empty;
        _stream.AdvancePipeline(tex, prompt);
    }

    private IEnumerator ContinuousGenerationRoutine()
    {
        var wait = new WaitForSeconds(_generationInterval);
        while (_continuousGeneration)
        {
            if (_stream != null && _stream.isRunning() && !_stream.isGenerating())
            {
                var tex = CaptureFromMaterialOrRT();
                if (tex != null)
                {
                    _lastSentTex2D = tex;
                    string prompt = _promptInput != null ? _promptInput.text : string.Empty;
                    _stream.AdvancePipeline(tex, prompt);
                }
            }
            yield return wait;
        }
    }
}
