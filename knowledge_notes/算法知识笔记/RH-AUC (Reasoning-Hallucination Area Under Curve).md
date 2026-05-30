## RH-AUC (Reasoning-Hallucination Area Under Curve)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RH-AUC 是一种评估多模态推理模型的综合指标，由 Liu 等人于 2025 年在 NeurIPS 论文 "More Thinking, Less Seeing?" 中提出（arXiv:2505.21523）。RH-AUC 量化模型在不同推理长度 T 下，Reasoning 准确率 R_T 与 Perception/Hallucination 准确率 H_T 之间的权衡。计算方式：对不同 T 采样多组 (R_T, H_T)，将两者 min-max 归一化到 [0,1]，通过梯形法则计算 H 相对于 R 的曲线下面积：RH-AUC = Σ_{i=0}^{n-2} (R_{T(i+1)} - R_{T(i)})/2 · (H_{T(i+1)} + H_{T(i)})。更高的 RH-AUC 表示模型在提升推理深度的同时更好地保持视觉接地。传统指标（固定长度的准确率、幻觉率）无法捕捉推理深度与感知可靠性之间的动态 trade-off。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# RH-AUC 计算
T_values = [short, medium, long, ...]
for T in T_values:
    R_T.append(eval_reasoning(model, T))     # 500 samples
    H_T.append(eval_perception(model, T))    # 500 samples

R_norm = (R - min(R)) / (max(R) - min(R))
H_norm = (H - min(H)) / (max(H) - min(H))

RH_AUC = 0
for i in range(len(T) - 1):
    width = R_norm[i+1] - R_norm[i]
    height = (H_norm[i+1] + H_norm[i]) / 2
    RH_AUC += width * height
```

ECRD 在 Qwen2.5-VL-7B 上 RH-AUC: 0.51 → 0.58（+0.07）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
RH-AUC 配套 RH-Bench（1000 samples：500 reasoning + 500 perception），reasoning 样本来自 MathVision、MathVista、MMMU、ScienceQA；perception/hallucination 样本来自 MMHalu、MMVP、HallusionBench、VMCBench。包含多项选择和开放式问题。RH-AUC 已成为评估 visually-grounded reasoning 方法的关键指标之一。

涉及论文标题：
- See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs
