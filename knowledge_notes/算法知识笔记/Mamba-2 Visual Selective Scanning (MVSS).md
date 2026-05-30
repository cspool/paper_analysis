## Mamba-2 Visual Selective Scanning (MVSS)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mamba-2 Visual Selective Scanning (MVSS) 是MSC模块的核心组件，利用Mamba-2层的selective scan机制处理2D视觉patch序列。MVSS探索两种2D扫描机制：(1) Bidirectional-Scan Mechanism (BSM)——沿前后两个方向扫描patch序列，捕获互补的上下文信息；(2) Cross-Scan Mechanism (CSM)——沿四个对角线方向扫描，捕获更丰富的2D空间关系。MVSS的核心insight是将2D视觉数据通过有结构的scan pattern转化为类似1D序列的输入，使Mamba-2的因果SSM能够有效处理非因果的视觉信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// BS机制：前后向扫描
V_img_patches = flatten(image_patches_2d)  // ∈ R^{729×D}, 27×27 grid → 1D
V_f = Mamba2_Block(V_img_patches)          // forward scan
V_b = Mamba2_Block(reverse(V_img_patches)) // backward scan
V_out_bsm = V_f + reverse(V_b)            // merge

// CS机制：四方向对角线扫描
V_out_csm = zeros_like(V_img_patches)
for each direction d in [↘, ↖, ↙, ↗]:  // 4 diagonal directions
    V_d = scan_along_direction(V_img_patches, d)  // 沿方向d展开为1D序列
    V_d_out = Mamba2_Block(V_d)                    // SSM处理
    V_out_csm += unscan_to_grid(V_d_out, d)        // 恢复为2D grid并累加
V_out_csm = V_out_csm / 4  // 平均四个方向
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MVSS的实现基于Mamba-2官方kernel。消融实验（Table 7）显示BSM在大多数benchmark上优于CSM（VQAv2: 75.26 vs 75.14, GQA: 60.68 vs 60.13, VizWiz: 45.17 vs 44.89），但CSM在TextVQA（52.31 vs 52.20）和POPE（88.5 vs 88.3）上略优。这表明BSM的前后向扫描更通用，CSM的对角线扫描在某些需要细粒度空间推理的任务上受益。MVSS的设计借鉴了Vim（Vision Mamba）的双向扫描和VMamba的交叉扫描机制，但使用Mamba-2（而非Mamba-1）作为核心扫描模块，效率更高（Mamba-2 scan比Mamba-1快2-8倍）。适用于将Mamba-2用于视觉特征建模的任何场景。

涉及论文标题：
- ML-Mamba__Efficient_Multi-Modal_Large_Language_Model_Utilizing_Mamba-2

---
