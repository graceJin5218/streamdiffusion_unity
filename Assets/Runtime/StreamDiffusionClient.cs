using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Globalization;
using PimDeWitte.UnityMainThreadDispatcher;
using UnityEngine;

/// <summary>
/// StreamDiffusion客户端，用于与Python服务端通信并控制图像生成
/// 所有参数都可在Unity Inspector中配置：
/// - 模型路径：相对于StreamingAssets/models目录的路径
/// - 图像尺寸：生成图像的宽高
/// - 种子：随机种子，影响生成效果
/// - 加速模式：使用的加速技术，如tensorrt, cuda等
/// - 提示词：控制图像生成内容的文本描述
/// - 强度参数：控制生成过程的各种强度系数
/// </summary>
public class StreamDiffusionClient : MonoBehaviour
{
    [Tooltip("基础模型路径，相对于StreamingAssets/models目录")]
    public string _baseModelPath = "kohaku-v2.1";

    [Tooltip("VAE模型路径，相对于StreamingAssets/models目录")]
    public string _tinyVaeModelPath = "taesd";

    [Tooltip("第一个LoRA模型路径，相对于StreamingAssets/models目录")]
    public string _loraModelPath = "lcm-lora-sdv1-5";

    [Tooltip("第二个LoRA模型路径，相对于StreamingAssets/models目录")]
    public string _loraModelPath2 = "";

    [Tooltip("加速模式：tensorrt, cuda等")]
    public string _acceleration = "tensorrt";

    [Tooltip("图像宽度")]
    public int _width = 512;

    [Tooltip("图像高度")]
    public int _height = 512;

    [Tooltip("随机种子")]
    public int _seed = 603665;

    [Tooltip("是否使用TinyVAE")]
    public bool _useTinyVae = true;

    [Tooltip("是否使用LCM LoRA")]
    public bool _useLcmLora = true;

    [Tooltip("第一个LoRA强度")]
    [Range(0.1f, 1.0f)]
    public float _loraScale = 0.85f;

    [Tooltip("第二个LoRA强度")]
    [Range(0.1f, 1.0f)]
    public float _loraScale2 = 0.5f;

    [Header("图像预处理参数")]
    [Tooltip("亮度调整")]
    [Range(0.5f, 3f)]
    public float _brightness = 1.0f;

    [Tooltip("对比度调整")]
    [Range(0.5f, 1.5f)]
    public float _contrast = 1.0f;

    [Tooltip("饱和度调整")]
    [Range(0.0f, 2.0f)]
    public float _saturation = 1.0f;

    [Space]
    [Tooltip("是否显示Python控制台")]
    public bool _showPythonConsole = false;

    [Tooltip("图像生成强度")]
    [Range(0.1f, 10.0f)]
    public float _strength = 1.0f;

    [Header("高级参数")]
    [Tooltip("Delta参数，控制噪声添加量")]
    [Range(0.1f, 1.0f)]
    public float _delta = 0.8f;

    [Tooltip("是否在每步添加噪声")]
    public bool _doAddNoise = true;

    [Tooltip("是否启用相似图像过滤")]
    public bool _enableSimilarFilter = true;

    [Tooltip("相似图像过滤阈值")]
    [Range(0.1f, 0.99f)]
    public float _similarThreshold = 0.6f;

    [Tooltip("最大跳过帧数")]
    [Range(1, 30)]
    public int _maxSkipFrame = 10;

    [Tooltip("引导尺度")]
    [Range(0.1f, 10.0f)]
    public float _guidanceScale = 1.0f;

    [Space]
    [Tooltip("绕过模式，直接返回输入图像而不经过AI处理")]
    public bool _bypassMode = false;

    [Space]
    [Tooltip("提示词")]
    public string _defaultPrompt = "";

    [Tooltip("负面提示词")]
    public string _defaultNegativePrompt = "";

    [Header("동적 변환 설정")]
    [Tooltip("초기 그리기 단계에서의 변환 강도")]
    [Range(0.0f, 1.0f)]
    public float _initialStrength = 0.15f;

    [Tooltip("중간 단계에서의 변환 강도")]
    [Range(0.0f, 1.0f)]
    public float _midStrength = 0.5f;

    [Tooltip("완성 단계에서의 변환 강도")]
    [Range(0.0f, 1.0f)]
    public float _finalStrength = 0.8f;

    [Tooltip("디테일 레벨 감지를 위한 임계값 - 초기→중간")]
    [Range(0.1f, 0.9f)]
    public float _detailThreshold1 = 0.2f;

    [Tooltip("디테일 레벨 감지를 위한 임계값 - 중간→완성")]
    [Range(0.1f, 0.9f)]
    public float _detailThreshold2 = 0.6f;

    [Tooltip("동적 시스템 사용 여부")]
    public bool _useDynamicSystem = true;

    private float _smoothedStrength = -1f;
    [Range(0.0f, 1.0f)] public float _strengthSmoothing = 0.2f;

    public Material _resultMaterial;
    private Texture2D _resultTexture;
    private System.Diagnostics.Process _backgroundProcess;

    private string _serverIP = "127.0.0.1";
    private int _serverPort = 9999;
    private int _pipelineLoaded = 0;
    private bool _isRunning = false, _isAdvancing = false;
    private bool _restartRequested = false;

    private int _receiveBufferSize = 65536;
    private int _receiveTimeout = 120000;
    private int _sendTimeout = 30000;
    private int _maxRetryCount = 3;

    private TcpClient _client;
    private NetworkStream _stream;
    private Thread _clientThread;
    private byte[] _advancedData;

    // ──────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────
    private static string F(float v) => v.ToString(CultureInfo.InvariantCulture);
    private static string F(int v) => v.ToString(CultureInfo.InvariantCulture);

    public bool isValid() => _isRunning;
    public bool isGenerating() => _isAdvancing;
    public bool isPending() => _isRunning && _pipelineLoaded < 0;
    public bool isRunning() => _isRunning && _pipelineLoaded > 0;

    // 현재 그림의 복잡도를 분석하는 함수
    private float AnalyzeImageComplexity(Texture2D inputImage)
    {
        if (inputImage == null) return 0f;

        Color[] pixels = inputImage.GetPixels();
        int totalPixels = pixels.Length;
        int drawnPixels = 0;
        float totalVariation = 0f;

        foreach (Color pixel in pixels)
        {
            float brightness = (pixel.r + pixel.g + pixel.b) / 3f;
            if (brightness < 0.95f)
            {
                drawnPixels++;
            }
        }

        float drawnRatio = (float)drawnPixels / totalPixels;

        int width = inputImage.width;
        int height = inputImage.height;

        for (int y = 1; y < height - 1; y++)
        {
            for (int x = 1; x < width - 1; x++)
            {
                Color current = pixels[y * width + x];
                Color right = pixels[y * width + (x + 1)];
                Color down = pixels[(y + 1) * width + x];

                float variation = Mathf.Abs(current.r - right.r) +
                                  Mathf.Abs(current.g - right.g) +
                                  Mathf.Abs(current.b - right.b) +
                                  Mathf.Abs(current.r - down.r) +
                                  Mathf.Abs(current.g - down.g) +
                                  Mathf.Abs(current.b - down.b);

                totalVariation += variation;
            }
        }

        float avgVariation = totalVariation / (width * height);
        float complexity = (drawnRatio * 0.7f) + (avgVariation * 0.3f);

        return Mathf.Clamp01(complexity);
    }

    // 복잡도에 따른 Strength 결정
    private float GetDynamicStrength(float complexity)
    {
        if (complexity < _detailThreshold1)
        {
            return _initialStrength;
        }
        else if (complexity < _detailThreshold2)
        {
            float t = (complexity - _detailThreshold1) / (_detailThreshold2 - _detailThreshold1);
            t = Mathf.Clamp01(t);
            t = Mathf.SmoothStep(0f, 1f, t);
            return Mathf.Lerp(_initialStrength, _midStrength, t);
        }
        else
        {
            float t = (complexity - _detailThreshold2) / (1f - _detailThreshold2);
            return Mathf.Lerp(_midStrength, _finalStrength, t);
        }
    }

    [SerializeField]
    private string _stylePreset =
        "consistent style, clean lines, soft colors, subtle watercolor, gentle lighting";

    // 기본 프롬프트만 사용
    private string GetStageAppropriatePrompt(float complexity, string basePrompt)
    {
        return basePrompt;
    }

    // AdvancePipelineWithDynamicStrength
    public void AdvancePipelineWithDynamicStrength(Texture2D tex, string prompt)
    {
        if (string.IsNullOrEmpty(prompt))
        {
            prompt = _defaultPrompt;
        }

        if (!_useDynamicSystem)
        {
            AdvancePipeline(tex, prompt);
            return;
        }

        float complexity = AnalyzeImageComplexity(tex);
        float dynamicStrength = GetDynamicStrength(complexity);

        if (_smoothedStrength < 0f) _smoothedStrength = dynamicStrength;
        _smoothedStrength = Mathf.Lerp(_smoothedStrength, dynamicStrength, _strengthSmoothing);

        Debug.Log($"이미지 복잡도: {complexity:F3}, 사용할 Strength: {dynamicStrength:F3}");
        Debug.Log($"사용할 프롬프트: {prompt}");

        string adjustedPrompt = GetStageAppropriatePrompt(complexity, prompt);
        Debug.Log($"조정된 프롬프트: {adjustedPrompt}");

        // 동적 strength 사용
        AdvancePipeline(tex, adjustedPrompt, _smoothedStrength);
    }

    private string GetFullModelPath(string relativePath)
    {
        return System.IO.Path.Combine(Application.streamingAssetsPath, "models", relativePath)
                           .Replace("\\", "/");
    }

    private void ConnectToServer()
    {
        try
        {
            _client.Connect(_serverIP, _serverPort);
            _stream = _client.GetStream();
            _isRunning = true;

            setModelPaths();

            Thread receiveThread = new Thread(new ThreadStart(ReceiveMessages));
            receiveThread.IsBackground = true;
            receiveThread.Start();
        }
        catch (System.Exception e)
        {
            Debug.LogError("Error connecting to server: " + e.Message);
        }
    }

    private void ReceiveMessages()
    {
        try
        {
            byte[] tempBuffer = new byte[_receiveBufferSize];

            while (_isRunning)
            {
                if (_stream.DataAvailable)
                {
                    try
                    {
                        int bytesRead = _stream.Read(tempBuffer, 0, tempBuffer.Length);
                        if (bytesRead > 0)
                        {
                            byte[] receivedBytes = new byte[bytesRead];
                            System.Buffer.BlockCopy(tempBuffer, 0, receivedBytes, 0, bytesRead);

                            UnityMainThreadDispatcher.Instance().Enqueue(() =>
                            {
                                UpdateServerData(receivedBytes, bytesRead);
                            });
                        }
                    }
                    catch (System.IO.IOException ioEx)
                    {
                        Debug.LogWarning($"IO异常: {ioEx.Message}，尝试继续接收...");
                        Thread.Sleep(100);
                    }
                }
                else
                {
                    Thread.Sleep(10);
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"接收消息时出错: {e.Message}");

            if (_isRunning && !_restartRequested)
            {
                Debug.Log("尝试重新连接服务器...");
                TryReconnect();
            }
        }
    }

    private void UpdateServerData(byte[] data, int length)
    {
        byte[] buffer = new byte[length];
        System.Buffer.BlockCopy(data, 0, buffer, 0, length);

        if (_pipelineLoaded <= 0)
        {
            if (length < 10)
            {
                string result = Encoding.UTF8.GetString(buffer);
                if (result == "loaded") _pipelineLoaded = 1;
                else _pipelineLoaded = 0;
            }
        }
        else if (length > 10)
        {
            byte[] lastFourBytes = new byte[4];
            System.Array.Copy(buffer, buffer.Length - 4, lastFourBytes, 0, 4);
            bool validEnd = true;
            byte pipeByte = (byte)'|';
            for (int i = 0; i < 4; i++)
            {
                if (lastFourBytes[i] != pipeByte) validEnd = false;
            }

            int len0 = (_advancedData == null ? 0 : _advancedData.Length);
            int len1 = (validEnd ? (buffer.Length - 4) : buffer.Length);
            byte[] totalData = new byte[len0 + len1];
            if (_advancedData != null)
                System.Buffer.BlockCopy(_advancedData, 0, totalData, 0, len0);
            System.Buffer.BlockCopy(buffer, 0, totalData, len0, len1);
            _advancedData = totalData;

            if (validEnd)
            {
                Debug.Log($"接收到完整图像数据，大小: {_advancedData.Length} 字节");
                if (_resultTexture == null)
                {
                    _resultTexture = new Texture2D(2, 2, TextureFormat.RGBA32, false,
                        QualitySettings.activeColorSpace == ColorSpace.Linear);
                    if (_resultMaterial != null)
                    {
                        _resultMaterial.mainTexture = _resultTexture;
                        _resultMaterial.SetColor("_BGColor", new Color(0.12f, 0.12f, 0.12f, 1f));
                        _resultMaterial.SetFloat("_MatteStrength", 1.0f);
                        Debug.Log("将新纹理分配给材质");
                    }
                    else
                        Debug.LogError("结果材质为空，无法分配纹理");
                }

                try
                {
                    string dataPrefix = "";
                    for (int i = 0; i < System.Math.Min(20, _advancedData.Length); i++)
                    { dataPrefix += _advancedData[i].ToString("X2") + " "; }
                    Debug.Log($"图像数据前缀: {dataPrefix}");

                    bool isPng = (_advancedData.Length > 8 &&
                        _advancedData[0] == 0x89 && _advancedData[1] == 0x50 &&
                        _advancedData[2] == 0x4E && _advancedData[3] == 0x47);

                    bool isJpeg = (_advancedData.Length > 3 &&
                        _advancedData[0] == 0xFF && _advancedData[1] == 0xD8 &&
                        _advancedData[2] == 0xFF);

                    bool success = _resultTexture.LoadImage(_advancedData, !isPng);
                    Debug.Log($"加载图像{(success ? "成功" : "失败")}，宽度={_resultTexture.width}, 高度={_resultTexture.height}");

                    if (success)
                    {
                        _resultTexture.Apply();

                        if (_resultMaterial != null)
                        {
                            _resultMaterial.mainTexture = _resultTexture;
                            _resultMaterial.SetColor("_BGColor", new Color(0.12f, 0.12f, 0.12f, 1f));
                            _resultMaterial.SetFloat("_MatteStrength", 1.0f);
                        }
                        else
                        {
                            Debug.LogWarning("결과 머티리얼이 비어 있습니다. 텍스처는 로드되었으나 머티리얼에 할당되지 않았습니다.");
                        }

#if UNITY_EDITOR
                        try
                        {
                            var px = _resultTexture.GetPixels32();
                            if (px != null && px.Length > 0)
                            {
                                double sR = 0, sG = 0, sB = 0;
                                foreach (var c in px) { sR += c.r; sG += c.g; sB += c.b; }
                                double n = px.Length * 255.0;
                                Debug.Log($"[DEBUG] 평균 RGB: R={(sR / n):F3}, G={(sG / n):F3}, B={(sB / n):F3}");
                            }
                        }
                        catch { }
#endif
                    }
                    else
                    {
                        Debug.LogError("이미지를 텍스처로 로드하지 못했습니다.");
                        _resultTexture = new Texture2D(512, 512, TextureFormat.RGBA32, false,
                                                       QualitySettings.activeColorSpace == ColorSpace.Linear);

                        Color[] testPixels = new Color[512 * 512];
                        for (int i = 0; i < testPixels.Length; i++) testPixels[i] = new Color(1, 1, 0, 1);
                        _resultTexture.SetPixels(testPixels);
                        _resultTexture.Apply();
                        if (_resultMaterial != null)
                        {
                            _resultMaterial.mainTexture = _resultTexture;
                            _resultMaterial.SetColor("_BGColor", new Color(0.12f, 0.12f, 0.12f, 1f));
                            _resultMaterial.SetFloat("_MatteStrength", 1.0f);
                        }
                    }
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"处理图像数据时出错: {e.Message}");
                }

                _advancedData = null;
                _isAdvancing = false;
            }
        }
        else
        {
            string result = Encoding.UTF8.GetString(buffer);
            _advancedData = null;
            _isAdvancing = false;

            // ★ 파이썬이 회신한 적용값(JSON) 로깅: {"ok":true,"trace_id":...,"applied_params":{...}}
            if (!string.IsNullOrEmpty(result) && result.TrimStart().StartsWith("{"))
            {
                Debug.Log($"[PY RESP] {result}");
            }
            else
            {
                Debug.Log("Received message: " + result);
            }
        }
    }

    public byte[] PreprocessImage(byte[] imageBytes, int width, int height)
    {
        if (imageBytes == null || imageBytes.Length == 0)
            return imageBytes;

        Texture2D texture = new Texture2D(width, height, TextureFormat.RGB24, false);
        Color32[] pixels = new Color32[width * height];

        for (int i = 0; i < width * height; i++)
        {
            int byteIndex = i * 3;
            if (byteIndex + 2 < imageBytes.Length)
            {
                pixels[i] = new Color32(
                    imageBytes[byteIndex],
                    imageBytes[byteIndex + 1],
                    imageBytes[byteIndex + 2],
                    255);
            }
        }

        for (int i = 0; i < pixels.Length; i++)
        {
            Color linearColor = new Color(
                pixels[i].r / 255f,
                pixels[i].g / 255f,
                pixels[i].b / 255f);

            linearColor *= _brightness;

            if (_contrast != 1f)
            {
                linearColor.r = (linearColor.r - 0.5f) * _contrast + 0.5f;
                linearColor.g = (linearColor.g - 0.5f) * _contrast + 0.5f;
                linearColor.b = (linearColor.b - 0.5f) * _contrast + 0.5f;
            }

            if (_saturation != 1f)
            {
                float luminance = linearColor.r * 0.3f + linearColor.g * 0.59f + linearColor.b * 0.11f;
                linearColor.r = Mathf.Lerp(luminance, linearColor.r, _saturation);
                linearColor.g = Mathf.Lerp(luminance, linearColor.g, _saturation);
                linearColor.b = Mathf.Lerp(luminance, linearColor.b, _saturation);
            }

            linearColor.r = Mathf.Clamp01(linearColor.r);
            linearColor.g = Mathf.Clamp01(linearColor.g);
            linearColor.b = Mathf.Clamp01(linearColor.b);

            pixels[i] = new Color32(
                (byte)(linearColor.r * 255),
                (byte)(linearColor.g * 255),
                (byte)(linearColor.b * 255),
                255);
        }

        byte[] result = new byte[width * height * 3];
        for (int i = 0; i < pixels.Length; i++)
        {
            int byteIndex = i * 3;
            result[byteIndex] = pixels[i].r;
            result[byteIndex + 1] = pixels[i].g;
            result[byteIndex + 2] = pixels[i].b;
        }
        return result;
    }

    public void setModelPaths()
    {
        string fullBaseModelPath = GetFullModelPath(_baseModelPath);
        string fullVaePath = GetFullModelPath(_tinyVaeModelPath);

        string fullLoraPath = string.IsNullOrEmpty(_loraModelPath) ? "" : GetFullModelPath(_loraModelPath);
        string fullLoraPath2 = string.IsNullOrEmpty(_loraModelPath2) ? "" : GetFullModelPath(_loraModelPath2);

        Debug.Log(
            $"Sending paths to Python server: Base={fullBaseModelPath}, VAE={fullVaePath}, LoRA1={fullLoraPath}, LoRA2={fullLoraPath2}"
        );

        string cmd0 =
            $"|start|command||paths||base_model||{fullBaseModelPath}||taesd_model||{fullVaePath}"
            + $"||lora_model||{fullLoraPath}||lora_model2||{fullLoraPath2}||run||0|end|";
        SendCommandToPython(cmd0);
    }

    public void LoadPipeline()
    {
        int vae = (_useTinyVae ? 1 : 0),
            lora = (_useLcmLora ? 1 : 0);

        Debug.Log(
            $"준비발송파라미터到Python: 宽度={_width}, 高度={_height}, 种子={_seed}, 强度={_strength}, LoRA1强度={_loraScale}, LoRA2强度={_loraScale2}"
        );
        Debug.Log(
            $"고급파라미터: Delta={_delta}, 添加噪声={_doAddNoise}, 相似图像过滤={_enableSimilarFilter}, 阈值={_similarThreshold}, 最大跳帧={_maxSkipFrame}"
        );
        Debug.Log(
            $"引导尺度: {_guidanceScale}, 绕过模式: {_bypassMode}"
        );

        string seedStr = _seed.ToString();

        string pB64 = Convert.ToBase64String(Encoding.UTF8.GetBytes(_defaultPrompt ?? ""));
        string npB64 = Convert.ToBase64String(Encoding.UTF8.GetBytes(_defaultNegativePrompt ?? ""));

        string loadCmd =
            $"|start|command||load||width||{_width}||height||{_height}||seed||{seedStr}"
            + $"||use_vae||{vae}||use_lora||{lora}||strength||{F(_strength)}||lora_scale||{F(_loraScale)}||lora_scale2||{F(_loraScale2)}"
            + $"||delta||{F(_delta)}||do_add_noise||{(_doAddNoise ? 1 : 0)}||enable_similar_filter||{(_enableSimilarFilter ? 1 : 0)}"
            + $"||similar_threshold||{F(_similarThreshold)}||max_skip_frame||{_maxSkipFrame}||guidance_scale||{F(_guidanceScale)}||acceleration||{_acceleration}"
            + $"||bypass_mode||{(_bypassMode ? "true" : "false")}"
            + $"||prompt_b64||{pB64}||neg_prompt||{npB64}||run||0|end|";

        _pipelineLoaded = -1;
        SendCommandToPython(loadCmd);
    }

    private void SendCommandToPython(string command)
    {
        try
        {
            if (_stream != null && _stream.CanWrite)
            {
                byte[] messageBytes = Encoding.UTF8.GetBytes(command);
                _stream.Write(messageBytes, 0, messageBytes.Length);
            }
            else
                Debug.LogError("Cannot send command - stream is null or not writable");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error sending command to Python: {e.Message}");
        }
    }

    // AdvancePipeline 메서드 (InvariantCulture + trace_id + 인자 포함)
    public void AdvancePipeline(
        Texture2D tex,
        string prompt,
        float? strength = null,
        float? loraScale = null,
        float? loraScale2 = null
    )
    {
        try
        {
            if (_stream == null || !_stream.CanWrite || tex == null)
            {
                Debug.LogError("Cannot advance pipeline - stream is null or not writable, or texture is null");
                return;
            }

            // 프롬프트 기본값
            if (string.IsNullOrEmpty(prompt))
            {
                prompt = _defaultPrompt;
                Debug.Log($"빈 프롬프트 감지, 기본 프롬프트 사용: {prompt}");
            }

            // 사용할 파라미터
            float currentStrength = strength ?? _strength;
            float currentLoraScale = loraScale ?? _loraScale;
            float currentLoraScale2 = loraScale2 ?? _loraScale2;

            bool isLinearSpace = QualitySettings.activeColorSpace == ColorSpace.Linear;

            Texture2D processTexture = tex;
            bool needsConversion = false;

            // PNG 인코딩 미지원/읽기불가 포맷 대비
            if (tex.format == TextureFormat.DXT1 ||
                tex.format == TextureFormat.DXT5 ||
                tex.format == TextureFormat.ETC_RGB4 ||
                tex.format == TextureFormat.ETC2_RGBA8 ||
                tex.format == TextureFormat.ASTC_4x4 ||
                tex.format == TextureFormat.ASTC_6x6 ||
                tex.format == TextureFormat.ASTC_8x8 ||
                tex.format == TextureFormat.ASTC_10x10 ||
                tex.format == TextureFormat.ASTC_12x12 ||
                tex.format == TextureFormat.PVRTC_RGB2 ||
                tex.format == TextureFormat.PVRTC_RGBA2 ||
                tex.format == TextureFormat.PVRTC_RGB4 ||
                tex.format == TextureFormat.PVRTC_RGBA4 ||
                !tex.isReadable)
            {
                needsConversion = true;
                Debug.LogWarning($"텍스처 포맷 {tex.format}은 EncodeToPNG를 지원하지 않습니다. 변환합니다.");
            }

            if (needsConversion)
            {
                RenderTexture renderTexture = RenderTexture.GetTemporary(
                    tex.width,
                    tex.height,
                    0,
                    RenderTextureFormat.ARGB32,
                    RenderTextureReadWrite.Default
                );

                Graphics.Blit(tex, renderTexture);

                processTexture = new Texture2D(
                    tex.width,
                    tex.height,
                    TextureFormat.RGBA32,
                    false,
                    isLinearSpace
                );

                RenderTexture previousActive = RenderTexture.active;
                RenderTexture.active = renderTexture;

                processTexture.ReadPixels(new Rect(0, 0, tex.width, tex.height), 0, 0);
                processTexture.Apply();

                RenderTexture.active = previousActive;
                RenderTexture.ReleaseTemporary(renderTexture);
            }

            // 평균 RGB(디버깅)
            Color[] pixels = processTexture.GetPixels();
            float avgR = 0, avgG = 0, avgB = 0;
            foreach (Color p in pixels)
            {
                avgR += p.r; avgG += p.g; avgB += p.b;
            }
            if (pixels.Length > 0)
            {
                avgR /= pixels.Length; avgG /= pixels.Length; avgB /= pixels.Length;
                Debug.Log($"원본 이미지 평균 RGB값: R={avgR:F2}, G={avgG:F2}, B={avgB:F2}");
            }

            byte[] imageBytes = ImageConversion.EncodeToPNG(processTexture);

            if (needsConversion && processTexture != tex)
            {
                Destroy(processTexture);
            }

            if (imageBytes == null || imageBytes.Length == 0)
            {
                Debug.LogError("PNG 인코딩 실패");
                return;
            }

            string base64Image = System.Convert.ToBase64String(imageBytes);
            Debug.Log($"PNG 인코딩 사용, 크기: {imageBytes.Length} 바이트, Base64 길이: {base64Image.Length}");

            string promptForWire = prompt;
            string promptB64 = System.Convert.ToBase64String(Encoding.UTF8.GetBytes(promptForWire));

            // predict/advance 헤더: 모든 수치 InvariantCulture + trace_id 포함
            string traceId = System.Guid.NewGuid().ToString("N");
            string command =
                $"|start|command||advance" +
                $"||trace_id||{traceId}" +
                $"||prompt_b64||{promptB64}" +
                $"||strength||{F(currentStrength)}" +
                $"||lora_scale||{F(currentLoraScale)}" +
                $"||lora_scale2||{F(currentLoraScale2)}" +
                $"||guidance_scale||{F(_guidanceScale)}" +
                $"||delta||{F(_delta)}" +
                $"||do_add_noise||{(_doAddNoise ? 1 : 0)}" +
                $"||is_linear_space||{isLinearSpace.ToString().ToLower()}" +
                $"||bypass_mode||{_bypassMode.ToString().ToLower()}" +
                $"||image_base64||{base64Image}" +
                $"||run||0|end|";

            Debug.Log($"전송할 프롬프트: '{prompt}'");

            // 큰 메시지는 청크 전송
            byte[] commandBytes = Encoding.UTF8.GetBytes(command);
            int chunkSize = 16384;
            int sentBytes = 0;

            while (sentBytes < commandBytes.Length)
            {
                int remaining = commandBytes.Length - sentBytes;
                int currentChunkSize = System.Math.Min(remaining, chunkSize);

                _stream.Write(commandBytes, sentBytes, currentChunkSize);
                sentBytes += currentChunkSize;

                if (sentBytes < commandBytes.Length)
                {
                    Thread.Sleep(5);
                }
            }
            _isAdvancing = true;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"AdvancePipeline 에러: {e.Message}");
            _isAdvancing = false;
        }
    }

    private void OutputDataReceived(object sender, System.Diagnostics.DataReceivedEventArgs e)
    {
        if (!string.IsNullOrEmpty(e.Data))
            Debug.Log(e.Data);
    }

    private void StartTcpClient()
    {
        Debug.Log("Start TCP client...");
        _client = new TcpClient();

        _client.ReceiveBufferSize = _receiveBufferSize;
        _client.SendBufferSize = _receiveBufferSize;
        _client.ReceiveTimeout = _receiveTimeout;
        _client.SendTimeout = _sendTimeout;
        _client.NoDelay = true;

        _clientThread = new Thread(new ThreadStart(ConnectToServer));
        _clientThread.IsBackground = true;
        _clientThread.Start();
    }

    void Update()
    {
        if (Input.GetKey(KeyCode.LeftControl) && Input.GetKeyDown(KeyCode.R))
        {
            Debug.Log("重启StreamDiffusion服务的快捷键被按下 (Ctrl+R)");
            RestartStreamDiffusionService();
        }
    }

    public void RestartStreamDiffusionService()
    {
        Debug.Log("正在重启StreamDiffusion服务...");
        _restartRequested = true;

        _isRunning = false;
        _pipelineLoaded = 0;

        if (_client != null && _client.Connected)
        {
            try
            {
                if (_stream != null)
                    _stream.Close();
                _client.Close();
            }
            catch (System.Exception e)
            {
                Debug.LogError($"关闭连接时出错: {e.Message}");
            }
        }

        if (_backgroundProcess != null && !_backgroundProcess.HasExited)
        {
            try
            {
                _backgroundProcess.Kill();
                _backgroundProcess.WaitForExit(5000);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"终止Python进程时出错: {e.Message}");
            }
        }

        StartPythonProcess();

        _advancedData = null;
        _isAdvancing = false;
        Invoke("StartTcpClient", 8.0f);

        _restartRequested = false;
        Debug.Log("StreamDiffusion服务重启请求已처리");
    }

    private void StartPythonProcess()
    {
        var pythonHome = $"{Application.streamingAssetsPath}/envs/streamdiffusion";
        var exePath = $"{pythonHome}/python.exe";

        if (System.IO.File.Exists(exePath))
        {
            Debug.Log("启动背景服务器...");
            System.Diagnostics.ProcessStartInfo startInfo = new System.Diagnostics.ProcessStartInfo(exePath);
            startInfo.FileName = exePath;
            startInfo.Arguments = "image_predictor.py";
            startInfo.WorkingDirectory = $"{Application.streamingAssetsPath}";

            if (_showPythonConsole)
            {
                startInfo.UseShellExecute = true;
                startInfo.WindowStyle = System.Diagnostics.ProcessWindowStyle.Normal;
            }
            else
            {
                startInfo.UseShellExecute = false;
                startInfo.CreateNoWindow = true;
                startInfo.RedirectStandardOutput = true;
                startInfo.RedirectStandardError = true;
                startInfo.WindowStyle = System.Diagnostics.ProcessWindowStyle.Hidden;
            }

            _backgroundProcess = new System.Diagnostics.Process();
            _backgroundProcess.StartInfo = startInfo;

            if (!_showPythonConsole)
            {
                _backgroundProcess.OutputDataReceived += OutputDataReceived;
                _backgroundProcess.ErrorDataReceived += OutputDataReceived;
            }

            if (!_backgroundProcess.Start())
            {
                Debug.LogError("启动image_predictor.py失败");
            }

            if (!_showPythonConsole)
            {
                _backgroundProcess.BeginOutputReadLine();
                _backgroundProcess.BeginErrorReadLine();
            }
        }
        else
        {
            Debug.LogError("找不到Python.exe");
        }
    }

    private void TryReconnect()
    {
        int retryCount = 0;
        bool reconnected = false;

        while (!reconnected && retryCount < _maxRetryCount && _isRunning && !_restartRequested)
        {
            try
            {
                Debug.Log($"重新连接尝试 #{retryCount + 1}...");
                if (_client != null)
                {
                    _client.Close();
                }

                _client = new TcpClient();
                _client.ReceiveBufferSize = _receiveBufferSize;
                _client.SendBufferSize = _receiveBufferSize;
                _client.ReceiveTimeout = _receiveTimeout;
                _client.SendTimeout = _sendTimeout;
                _client.NoDelay = true;

                _client.Connect(_serverIP, _serverPort);
                _stream = _client.GetStream();

                reconnected = true;
                Debug.Log("重新连接成功！");

                Thread receiveThread = new Thread(new ThreadStart(ReceiveMessages));
                receiveThread.IsBackground = true;
                receiveThread.Start();
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"重连尝试 #{retryCount + 1} 失败: {ex.Message}");
                retryCount++;
                Thread.Sleep(2000);
            }
        }

        if (!reconnected && _isRunning && !_restartRequested)
        {
            Debug.LogError("重连失败，将重启服务...");

            UnityMainThreadDispatcher.Instance().Enqueue(() =>
            {
                RestartStreamDiffusionService();
            });
        }
    }

    void Start()
    {
        Application.runInBackground = true;

        var pythonHome = $"{Application.streamingAssetsPath}/envs/streamdiffusion";
        var projectHome = $"{Application.streamingAssetsPath}/streamdiffusion";
        var scripts = $"{pythonHome}/Scripts";

        var path = System.Environment.GetEnvironmentVariable("PATH")?.TrimEnd(';');
        path = string.IsNullOrEmpty(path)
            ? $"{pythonHome};{scripts}"
            : $"{pythonHome};{scripts};{path}";
        System.Environment.SetEnvironmentVariable("PATH", path, System.EnvironmentVariableTarget.Process);

        StartPythonProcess();
        Invoke("StartTcpClient", 8.0f);
    }

    void OnDestroy()
    {
        _isRunning = false;
        _pipelineLoaded = 0;
        if (_backgroundProcess != null && !_backgroundProcess.HasExited)
            _backgroundProcess.Kill();
        if (_clientThread != null)
            _clientThread.Interrupt();
        if (_client != null)
            _client.Close();
    }

    // InputImage (동적/고정 모드 모두 prompt 확실히 전달)
    public void InputImage(byte[] imageBytes, int width, int height)
    {
        if (!isRunning() || imageBytes == null || imageBytes.Length != width * height * 3)
        {
            Debug.LogError($"InputImage실패: 서비스 미실행 또는 이미지 데이터 무효");
            return;
        }

        try
        {
            Texture2D tex = new Texture2D(width, height, TextureFormat.RGB24, false);
            tex.LoadRawTextureData(imageBytes);
            tex.Apply();

            string promptToUse = string.IsNullOrEmpty(_defaultPrompt) ? "beautiful artwork" : _defaultPrompt;
            Debug.Log($"InputImage에서 사용할 프롬프트: {promptToUse}");

            if (_useDynamicSystem)
            {
                AdvancePipelineWithDynamicStrength(tex, promptToUse);
            }
            else
            {
                AdvancePipeline(tex, promptToUse);
            }

            Destroy(tex);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"InputImage처리 오류: {e.Message}");
        }
    }
}