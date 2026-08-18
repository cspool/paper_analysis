## 交替去噪特征复用（Alternating Denoising with Feature Reuse，去噪级冗余利用）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
迭代级去噪冗余利用策略：反向扩散相邻步的 attention/FFN 输出特征相似度 >98%（最低仍 >70%，且随去噪进程逐步下降），同时每步重复加载相同权重造成 60.1% 重复外部访存。该策略以粗粒度"步"为单位跳过与上一步近乎相同的计算：被跳过步省略完整 attention/FFN 计算与权重重载，仅保留低代价残差噪声更新。与 Cambricon-D/Ditto 的差分计算、EXION 的细粒度稀疏不同，它在 attention 和 FFN 两个块同时消除计算与访存，且规避了细粒度控制开销与对 GELU 等非线性算子的不兼容。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
跳过步选择（离线）：计算相邻步 attention/FFN 特征相似度 → 阈值初筛候选跳过步 → 评估跳过每个候选步引入的 MSE loss（早期步对 loss 不敏感）→ 确定最终跳过集。执行流伪代码：
```
full denoise at step t                       # 先做完整计算
for t' in schedule:
    if t' in skip_set:
        feat_attn[t'], feat_ffn[t'] = cached(feat_attn[t'-1], feat_ffn[t'-1])  # 特征复用
        x_{t'-1} = residual_noise_update(x_t')      # 仅低代价残差噪声更新
        # 省略 QKV/FFN GEMM 与外部权重访问
    else:
        x_{t'-1} = full_attention_ffn(x_t')          # 完整计算
```
关键性质：动作规划 DiT 的相邻步相似度全矩阵都高（图像 DiT 只在对角附近高、非对角迅速降为 0），因此适合"整步粗粒度跳过"而非逐区域细粒度复用；注意力与 FFN 相似度趋势一致，可统一按步优化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：软件框架 S2 策略，跳过集离线按相似度 + MSE 判据生成；硬件由 multimodal scheduler 的迭代索引表存储跳过配置。使用：约 40% 去噪迭代被消除；与"每 20 个跳过迭代重插完整去噪"配合重置累积误差；消融中在动作冗余之上再贡献 2.90× 总加速（1.74× → 2.90×）；特征复用带来的轻微"动量效应"使成功率甚至略高于 baseline。

涉及论文标题：
- DiTPA A DiT-based Action Planner Accelerator Exploiting Action–Denoising–Multimodality Redundancy for Embodied Artificial Intelligence
