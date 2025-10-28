Shader "Unlit/MatteToBG"
{
    Properties
    {
        _MainTex("Texture", 2D) = "white" {}
        _BGColor("Background Color", Color) = (0.12, 0.12, 0.12, 1)
        _MatteStrength("Matte Strength (0..1)", Range(0,1)) = 1.0
    }
        SubShader
        {
            Tags { "Queue" = "Transparent" "RenderType" = "Transparent" }
            Cull Off ZWrite Off ZTest LEqual
            Blend SrcAlpha OneMinusSrcAlpha

            Pass
            {
                CGPROGRAM
                #pragma vertex vert
                #pragma fragment frag
                #include "UnityCG.cginc"

                sampler2D _MainTex;
                float4 _MainTex_ST;
                float4 _BGColor;
                float  _MatteStrength;

                struct appdata {
                    float4 vertex : POSITION;
                    float2 uv     : TEXCOORD0;
                    float4 color  : COLOR;
                };
                struct v2f {
                    float4 pos : SV_POSITION;
                    float2 uv  : TEXCOORD0;
                    float4 col : COLOR;
                };

                v2f vert(appdata v) {
                    v2f o;
                    o.pos = UnityObjectToClipPos(v.vertex);
                    o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                    o.col = v.color;
                    return o;
                }

                fixed4 frag(v2f i) : SV_Target {
                    fixed4 c = tex2D(_MainTex, i.uv) * i.col;

                // 배경과 매트(미리 합성): 경계 고스트 억제
                float3 matteRGB = lerp(_BGColor.rgb, c.rgb, c.a);
                float3 outRGB = lerp(c.rgb, matteRGB, _MatteStrength);

                // 알파도 살짝 올리면 더 깔끔 (투명 유지 원하면 _MatteStrength 낮추기)
                float  outA = lerp(c.a, 1.0, _MatteStrength);

                return fixed4(outRGB, outA);
            }
            ENDCG
        }
        }
            FallBack Off
}