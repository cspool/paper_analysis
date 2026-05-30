## Temporal Similarity Regularization（Lsim，时序相似性正则化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Temporal Similarity Regularization（Lsim）是 ResAdapt 提出的正则化损失，用于打破 Allocator 对视觉相似相邻帧分配相同 scale 的对称性。CAPO 确定 accuracy-cost 的全局运行点，但无法阻止优化器对相似相邻帧赋予相同预算——这导致实际行为接近 FixedScale（均匀缩放），浪费了自适应分配的能力。Lsim 通过余弦相似度门控权重，仅在相邻帧超过相似度阈值时，惩罚它们的联合高预算分配。消融实验（Figure 7/13/14）显示：去除 Lsim 后 scale trace 坍缩为常数分布；恢复 Lsim 后 scale histogram 变为双峰、per-video range 扩大、Gini 系数上升——策略从退化均匀分配转型为真正的选择性分配。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Lsim 的具体计算公式：
```
# 输入: per-frame coarse features f_t, predicted scales s_t
# 超参数: τ_sim (cosine threshold), γ_sim (temperature), η_sim (margin)

L_sim = 0
for t = 1 to T-1:
    # 余弦相似度门控权重（仅在相似帧上激活）
    w_t = σ((cos(f_t, f_{t+1}) - τ_sim) / γ_sim)
    # 对数尺度联合惩罚（s_t * s_{t+1} > e^{-η_sim} 时才生效）
    penalty = max(0, log(s_t) + log(s_{t+1}) + η_sim)
    L_sim += w_t * penalty

L_sim /= (T - 1)
```

门控机制的关键：当 cos(f_t, f_{t+1}) << τ_sim 时 w_t→0（帧不相似，无需惩罚），当相似度超过阈值时 w_t 渐进上升。对数尺度惩罚确保 penalty 仅在 s_t·s_{t+1} > exp(−η_sim) 时激活——即两帧的联合分配超过下界时才受惩罚，而非无条件惩罚所有分配。Total Allocator loss: L_alloc = L_θ + λ_sim·L_sim + λ_con·L_con。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Lsim 的实现：(1) 在 Allocator 的 Transformer decoder 中缓存粗粒度特征 f_t；(2) 在训练循环中计算成对余弦相似度和 scale 的联合惩罚；(3) 与 Concentration Loss（Lcon = max(0, α_t+β_t−κ_max)/T）配合使用——Lsim 打破帧间对称性，Lcon 防止 Beta 分布坍缩为确定性。代码：https://github.com/Xnhyacinth/ResAdapt。

涉及论文标题：
- ResAdapt__Adaptive_Resolution_for_Efficient_Multimodal_Reasoning

涉及论文标题：
- Atlas__Multi-Scale_Attention_Improves_Long_Context_Image_Modeling
- DyCoke__Dynamic_Compression_of_Tokens_for_Fast_Video_Large_Language_Models
- Free-MoRef__Instantly_Multiplexing_Context_Perception_Capabilities_of_Video-MLLMs_within_Single_Inference
- LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling
- Owl-1__Omni_World_Model_for_Consistent_Long_Video_Generation
