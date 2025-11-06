using System;
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

    [Header("Idle Noise Generation")]
    [Tooltip("입력이 멈췄을 때도 미세한 노이즈를 섞어 지속적으로 변화하는 느낌을 줍니다.")]
    public bool _enableIdleNoise = true;

    [Tooltip("최근 프레임과의 차이가 이 비율보다 낮으면 '정지 상태'로 간주합니다.")]
    [Range(0.0f, 0.05f)] public float _idleChangeThreshold = 0.01f;

    [Tooltip("정지 상태로 판단 후 노이즈를 적용하기까지 대기하는 시간(초)")]
    [Range(0.0f, 2.0f)] public float _idleGracePeriod = 0.4f;

    [Tooltip("정지 직후 적용할 최소 노이즈 강도")]
    [Range(0.0f, 0.3f)] public float _idleNoiseStrengthMin = 0.02f;

    [Tooltip("노이즈 강도 (최대값, 0이면 노이즈 없음)")]
    [Range(0.0f, 0.3f)] public float _idleNoiseStrength = 0.08f;

    [Tooltip("픽셀 변화를 샘플링할 때의 간격. 값이 낮을수록 더 정확하지만 연산량이 증가합니다.")]
    [Range(1, 16)] public int _idleSampleStep = 4;

    [Tooltip("픽셀 변화로 간주하기 위한 채널 합산 차이 허용치 (0~255*3)")]
    [Range(0, 255)] public int _idleColorTolerance = 12;

    [Tooltip("정지 직후 적용할 최소 노이즈 커버리지 (0~1)")]
    [Range(0.0f, 1.0f)] public float _idleNoiseCoverageMin = 0.15f;

    [Tooltip("노이즈를 섞을 픽셀의 비율 최대값 (0~1)")]
    [Range(0.0f, 1.0f)] public float _idleNoiseCoverage = 0.45f;

    [Tooltip("노이즈 강도/커버리지가 최대치에 도달하는 데 걸리는 시간(초)")]
    [Range(0.0f, 5.0f)] public float _idleNoiseRampDuration = 2.0f;

    private Color32[] _previousFramePixels;
    private Texture2D _idleNoiseTexture;
    private Color32[] _idleNoisePixels;
    private System.Random _idleNoiseRandom = new System.Random();
    private float _lastActivityTime;
    private bool _idleNoiseActive;

    // ─────────────────────────────────────────────────────────────────────────────
    // Unity lifecycle
    // ─────────────────────────────────────────────────────────────────────────────
    private void Awake()
    {
        if (_inputMaterial != null)
            _originalInputTexture = _inputMaterial.mainTexture;

        _lastActivityTime = Time.time;
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

        if (_idleNoiseTexture != null)
        {
            Destroy(_idleNoiseTexture);
            _idleNoiseTexture = null;
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
        //_stream.AdvancePipeline(tex, prompt);
        SendFrameToStream(tex, prompt);
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
                    //_stream.AdvancePipeline(tex, prompt);
                    SendFrameToStream(tex, prompt);
                }
            }
            yield return wait;
        }
    }

    private void SendFrameToStream(Texture2D tex, string prompt)
    {
        if (_stream == null)
            return;

        Texture2D textureToSend = tex;
        Color32[] currentPixels = null;

        if (_enableIdleNoise)
        {
            currentPixels = tex.GetPixels32();
            bool hasSignificantChange = EvaluateFrameChange(currentPixels, out float diffRatio);
            float idleTime = 0f;

            if (hasSignificantChange)
            {
                _lastActivityTime = Time.time;
                _idleNoiseActive = false;
            }
            else
            {
                idleTime = Time.time - _lastActivityTime;
                _idleNoiseActive = idleTime >= _idleGracePeriod;
            }

            if (_idleNoiseActive)
            {
                float strengthMin = Mathf.Clamp01(Mathf.Min(_idleNoiseStrengthMin, _idleNoiseStrength));
                float strengthMax = Mathf.Clamp01(Mathf.Max(_idleNoiseStrengthMin, _idleNoiseStrength));
                float coverageMin = Mathf.Clamp01(Mathf.Min(_idleNoiseCoverageMin, _idleNoiseCoverage));
                float coverageMax = Mathf.Clamp01(Mathf.Max(_idleNoiseCoverageMin, _idleNoiseCoverage));

                float rampDuration = Mathf.Max(0f, _idleNoiseRampDuration);
                float rampProgress = rampDuration > 0f
                    ? Mathf.Clamp01((idleTime - _idleGracePeriod) / rampDuration)
                    : 1f;

                // diffRatio는 임계값보다 작지만, 최근 프레임의 변화량이 컸을수록 초기 노이즈를 더 빠르게 올려준다
                float changeFactor = Mathf.Clamp01(diffRatio / Mathf.Max(_idleChangeThreshold, 1e-5f));
                float noiseProgress = Mathf.Clamp01(Mathf.Max(rampProgress, changeFactor));
                float dynamicStrength = Mathf.Lerp(strengthMin, strengthMax, noiseProgress);
                float dynamicCoverage = Mathf.Lerp(coverageMin, coverageMax, noiseProgress);

                textureToSend = GetNoiseAugmentedTexture(tex, currentPixels, dynamicStrength, dynamicCoverage);
            }
        }

        _stream.AdvancePipeline(textureToSend, prompt);
    }

    private bool EvaluateFrameChange(Color32[] currentPixels, out float diffRatio)
    {
        diffRatio = 1f;

        if (currentPixels == null || currentPixels.Length == 0)
        {
            return true;
        }

        if (_previousFramePixels == null || _previousFramePixels.Length != currentPixels.Length)
        {
            _previousFramePixels = new Color32[currentPixels.Length];
            Array.Copy(currentPixels, _previousFramePixels, currentPixels.Length);
            diffRatio = 1f;
            return true;
        }

        int step = Mathf.Clamp(_idleSampleStep, 1, 64);
        int tolerance = Mathf.Clamp(_idleColorTolerance, 0, 765);
        int diffCount = 0;
        int sampleCount = 0;

        for (int i = 0; i < currentPixels.Length; i += step)
        {
            Color32 current = currentPixels[i];
            Color32 prev = _previousFramePixels[i];

            int delta = Mathf.Abs(current.r - prev.r)
                        + Mathf.Abs(current.g - prev.g)
                        + Mathf.Abs(current.b - prev.b);

            if (delta > tolerance)
            {
                diffCount++;
            }
            sampleCount++;
        }

        diffRatio = sampleCount > 0 ? (float)diffCount / sampleCount : 0f;

        Array.Copy(currentPixels, _previousFramePixels, currentPixels.Length);

        return diffRatio >= Mathf.Max(0f, _idleChangeThreshold);
    }

    private Texture2D GetNoiseAugmentedTexture(Texture2D source, Color32[] sourcePixels, float strength, float coverage)
    {
        if (sourcePixels == null)
            return source;

        if (_idleNoiseTexture == null || _idleNoiseTexture.width != source.width || _idleNoiseTexture.height != source.height)
        {
            if (_idleNoiseTexture != null)
            {
                Destroy(_idleNoiseTexture);
            }

            bool linear = QualitySettings.activeColorSpace == ColorSpace.Linear;
            _idleNoiseTexture = new Texture2D(source.width, source.height, TextureFormat.RGBA32, false, linear)
            {
                name = "IdleNoise_WorkingTex"
            };
        }

        if (_idleNoisePixels == null || _idleNoisePixels.Length != sourcePixels.Length)
        {
            _idleNoisePixels = new Color32[sourcePixels.Length];
        }

        Array.Copy(sourcePixels, _idleNoisePixels, sourcePixels.Length);

        strength = Mathf.Clamp01(strength);
        coverage = Mathf.Clamp01(coverage);

        if (strength > 0f && coverage > 0f)
        {
            for (int i = 0; i < _idleNoisePixels.Length; i++)
            {
                if (_idleNoiseRandom.NextDouble() > coverage)
                {
                    continue;
                }

                Color32 c = _idleNoisePixels[i];
                float noise = (float)_idleNoiseRandom.NextDouble() * 2f - 1f;
                int delta = Mathf.RoundToInt(noise * strength * 255f);

                int r = Mathf.Clamp(c.r + delta, 0, 255);
                int g = Mathf.Clamp(c.g + delta, 0, 255);
                int b = Mathf.Clamp(c.b + delta, 0, 255);

                _idleNoisePixels[i] = new Color32((byte)r, (byte)g, (byte)b, c.a);
            }
        }

        _idleNoiseTexture.SetPixels32(_idleNoisePixels);
        _idleNoiseTexture.Apply(false, false);

        return _idleNoiseTexture;
    }
}
