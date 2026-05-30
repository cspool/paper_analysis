## Cross-Scan Mechanism (CSM) in Mamba-2 Multimodal

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cross-Scan Mechanism (CSM) 是MVSS模块中的另一种2D扫描策略，将视觉patch特征沿四个对角线方向展开为1D序列并分别送入Mamba-2层处理。四个方向包括：左上→右下、右下→左上、右上→左下、左下→右上。CSM借鉴VMamba的Cross-Scan Module设计思想——对角线方向扫描可以捕获传统行列扫描难以建模的斜向空间关系（如物体的对角线边界、纹理等）。与BSM（2个方向）相比，CSM（4个方向）提供更密集的空间上下文覆盖，但计算量也翻倍。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// 四方向对角线扫描
directions = [
    "左上→右下",  // scan from top-left to bottom-right
    "右下→左上",  // scan from bottom-right to top-left
    "右上→左下",  // scan from top-right to bottom-left
    "左下→右上",  // scan from bottom-left to top-right
]

for each dir in directions:
    V_seq = flatten_grid_along_direction(V_2d_grid, dir)  // 按dir展开为1D
    V_out_dir = Mamba2_Block(V_seq)  // SSM scan
    V_recovered = reshape_to_grid(V_out_dir, dir)  // 恢复为2D
    V_accumulated += V_recovered

V_final = V_accumulated / 4  // 平均
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CSM实现需要4次Mamba-2的forward pass，计算量是BSM的2倍。消融实验（Table 7）显示CSM在TextVQA（+0.11 over BSM）和POPE（+0.2）上略有优势——可能是因为对角线扫描有助于OCR任务中的文本线条检测和物体边界判断。但由于BSM在更广泛benchmark上总体更优且计算更高效，ML-Mamba最终选用BSM作为默认配置。适用于需要精细空间推理（特别是斜向纹理/文字检测）的视觉-语言任务。

涉及论文标题：
- ML-Mamba__Efficient_Multi-Modal_Large_Language_Model_Utilizing_Mamba-2

---
