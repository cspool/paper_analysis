## Bidirectional-Scan Mechanism (BSM) in Mamba-2 Multimodal

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bidirectional-Scan Mechanism (BSM) 是MVSS模块中的一种2D扫描策略。它将视觉encoder输出的patch特征序列沿前后两个方向分别送入Mamba-2层处理：前向扫描保持原始patch展开顺序（行优先扫描），后向扫描反转顺序（逆序扫描），然后将两路输出合并。BSM的设计哲学源于Vim（Vision Mamba）的核心insight：自然语言有因果方向，但图像没有——双向扫描可以让每个patch在SSM的处理中"看到"两边的邻居，从而捕获2D receptive field，而不像纯前向SSM那样只能看到"左边"的patch。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// 假设 27×27 grid of visual patches
// 行优先展开为729-length 1D序列

// Mamba2_Block内部selective scan:
// For每个token位置t (0..728):
//   前向扫描: h_t = A_bar_f[t]·h_{t-1} + B_bar_f[t]·x[t]
//             y_f[t] = C_f[t]·h_t
//   后向扫描: h_t = A_bar_b[t]·h_{t-1} + B_bar_b[t]·x[728-t]
//             y_b[t] = C_b[t]·h_t
//   Mamba-2的A_bar_f/B_bar_f等参数由当前token x[t]动态生成（data-dependent）

// 合并: y_bsm[t] = y_f[t] + y_b[728-t]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
BSM的实现复用同一Mamba-2 Block的参数处理前向和后向序列（不额外增加参数），仅需flip操作和一次额外的scan。BSM在ML-Mamba消融中被选为默认扫描机制（优于CSM），因为其实现简单、计算开销小（仅2x scan vs CSM的4x scan），且在VQAv2、GQA、VizWiz、VSR四个benchmark上表现最优。适用于希望通过双向上下文增强SSM视觉处理的场景。

涉及论文标题：
- ML-Mamba__Efficient_Multi-Modal_Large_Language_Model_Utilizing_Mamba-2

---
