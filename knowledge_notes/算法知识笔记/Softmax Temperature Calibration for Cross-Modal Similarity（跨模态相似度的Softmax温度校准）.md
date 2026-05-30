## Softmax Temperature Calibration for Cross-Modal Similarity（跨模态相似度的Softmax温度校准）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Softmax Temperature Calibration 是 MMTok 中用于对齐 text-vision 和 vision-vision 相似度矩阵分布的技术。由于 T-V 相似度（基于投影后对齐 LLM embedding 的 vision tokens 与 text tokens 的内积）和 V-V 相似度（基于投影前的原始 vision tokens 的内积）具有不同的量纲和分布形状，直接相加会导致一个覆盖项主导另一个。因此 MMTok 对两个相似度矩阵分别做 temperature-scaled softmax 归一化：M'_{i,j} = exp(M_{i,j}/τ) / Σⱼ exp(M_{i,j}/τ)。温度 τ 控制分布的锐度：τ 越小，softmax 越接近 one-hot（强调最相似的 token pair）；τ 越大，分布越平滑（多个 token pair 都有显著权重）。MMTok 固定 τ_t=0.02（文本-视觉温度更低，文本查询通常关注少数相关视觉区域）和 τ_v=0.2（视觉-视觉温度更高，全图信息需要更多 token 协同覆盖）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Cross-modal Softmax Calibration
# M_tv ∈ R^(m×n): text-vision 相似度 (cosine, after projection)
# M_vv ∈ R^(n×n): vision-vision 相似度 (cosine, before projection)
# τ_t, τ_v: 温度参数

# Per-row softmax with temperature
M_tv_calibrated = softmax(M_tv / τ_t, dim=-1)
# M_tv_calibrated[i,:] 是第 i 个 text token 对所有 vision token 的概率分布
# τ_t=0.02 → 锐利分布, text 主要关注 1-2 个最相关 vision region

M_vv_calibrated = softmax(M_vv / τ_v, dim=-1)
# M_vv_calibrated[i,:] 是第 i 个 vision token 对所有 vision token 的概率分布
# τ_v=0.2 → 平滑分布, 让多个 vision token 参与覆盖全图信息

# 合并覆盖目标
f(S) = f(S; M_tv_calibrated) + α * f(S; M_vv_calibrated)
```

温度选择的直觉：
- τ_t < τ_v：因为 text-vision 语义对齐更精确（投影层专门训练用于对齐），高置信度的匹配应获得更高权重
- modality gap：vision tokens（投影前）与 vision tokens 之间的相似度天然更高（同一模态），需要更高温度调整到与 T-V 可比的量级
- 消融实验（Table 9）：将 τ_v 替换为自适应搜索策略（MMTok_Adapt），在不同温度候选 {0.05, 0.1, 0.15, 0.2} 中通过 bi-section 搜索使 f(N; M^{tv'}) ≈ f_k(N; M^{vv'}) 的 τ_v，性能几乎不变，说明方法对温度不敏感

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现使用 PyTorch 的 `F.softmax(M / tau, dim=-1)`。在 MMTok 代码中，temperature 作为可配置参数。使用建议：(1) 对于 VLM 架构不变的场景，使用默认值 τ_t=0.02, τ_v=0.2, α=0.5 即可；(2) 对于新的 VLM 架构或模态（如 video），可运行自适应温度搜索（MMTok_Adapt），在验证集上搜索最优 τ_v；(3) 温度校准使相似度矩阵行归一化为概率分布，等价于将覆盖问题从 absolute similarity maximization 转化为 relative relevance maximization。该技术的通用性使其可应用于任何需要融合异源相似度矩阵的场景（如 cross-modal retrieval, multi-view clustering 等）。

涉及论文标题：
- MMTok__Multimodal_Coverage_Maximization_for_Efficient_Inference_of_VLMs
