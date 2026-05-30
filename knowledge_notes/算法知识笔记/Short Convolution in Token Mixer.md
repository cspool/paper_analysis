## Short Convolution in Token Mixer

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Short Convolution 是现代线性 RNN/SSM token mixer 中广泛使用的轻量级局部混合组件，典型配置为 kernel size=4 的 causal depthwise 1D 卷积。在 Gated DeltaNet 中，q/k/v 路径在核心计算前先经：Linear Proj → ShortConv(kernel=4) → SiLU → (可选 L2 norm)。其作用是为 token mixer 提供最近 4 个 token 的局部上下文感知，弥补线性 RNN 状态压缩可能丢失的细粒度局部模式。类似设计见于 Mamba/Mamba2（conv window=4）、HGRN/HGRN2 等。Gated DeltaNet 消融（Table S.1）证实移除 short conv 导致 perplexity +1.6（27.35→28.95）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
x_conv[t] = Σ_{i=0}^{3} w_i · x_proj[t-i]  // causal depthwise 1D conv, kernel=4
x_act = SiLU(x_conv)
q_t = L2Norm(x_act)  // 对 query/key 额外加 L2 norm
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 中通常用 nn.Conv1d(groups=D) 实现 depthwise causal conv。参数量仅 kernel_size × D，相比主模型参数的 D×D 可忽略。多篇论文消融一致显示移除导致 1-2 ppl 下降，是 token mixer 中性价比最高的组件之一。

SAMBA 论文对 Short Convolution 的消融分析（Table 10）：将 SC 添加到不同线性递归模型中效果不同：(1) SC + SWA：perplexity 从 11.12→10.83 显著改善，说明 depthwise conv 的局部平滑对所有 token mixer 都有益；(2) SC + Sliding GLA：改善不显著（10.43→10.39），因为 GLA 已有 channel 级细粒度衰减，depthwise conv 未增加额外有用的归纳偏置；(3) SC + Sliding RetNet：改善明显（10.38→10.25），弥补了 RetNet 固定衰减的灵活度不足；(4) 在 hybrid 模型中同时给 SWA 和线性注意力层加 SC 反而产生负面效果。这些发现验证了 SC 的通用价值，但也揭示其效果依赖于 token mixer 的现有表达能力。

涉及论文标题：
- Gated_Delta_Networks__Improving_Mamba2_with_Delta_Rule
- Samba__Simple_Hybrid_State_Space_Models_for_Efficient_Unlimited_Context_Language_Modeling

---
