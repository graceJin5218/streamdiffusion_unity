using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ActivateAllDisplays : MonoBehaviour
{
    void Start()
    {
        // Display 0은 기본 활성. 나머지를 켠다.
        for (int i = 1; i < Display.displays.Length; i++)
            Display.displays[i].Activate(); // 필요하면 (width, height, refresh) 지정 가능
    }

}
