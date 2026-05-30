## Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression

- baseline方法是什么？
  Baseline 方法包括：(1) **StreamingLLM**：始终保留前几个 token（attention sink）和最近 n 个 token 的滑动窗口，丢弃中间 token；(2) **K-Norm**（Devoto et al., 2024）：基于 Key 的 L2 范数评估 KV pair 重要性，保留低范数的 KV pairs；(3) **SnapKV**：利用 prompt 末尾部分的注意力分数选择重要 KV pairs，需物化注意力矩阵，因此与 FlashAttention 不兼容。
  
  全栈执行例子（以 K-Norm 为代表性 baseline）：
  - **算法层**：计算每个 Key 向量的 L2 范数 $||K_t^h||_2$，保留范数最小的 KV pairs。这个启发式基于经验观察（低范数 Key 对应高平均注意力），但忽略了 Key 向量在 Query 主方向上的角度分量，近似精度有限。
  - **系统框架层**：使用 HuggingFace Transformers + KVPress 库，在 prefill 完成后或生成过程中对 KV Cache 进行压缩。论文未明确说明 Serving 框架层面的修改。
  - **编译框架/算子层**：不涉及编译框架修改。K-Norm 的范数计算需要在每次推理时显式计算，每次需要 $O(L \times d_H)$ 的浮点操作，与 Q-Filters 的标量积计算复杂度相当。论文未明确说明 kernel 层面的修改。
  - **硬件架构/芯片设计层**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Q-Filters**，通过分析 Query-Key 几何特性，发现 Query 分布具有各向异性（anisotropic），存在一个单一主方向 $u^h$（由 Query 矩阵 SVD 的第一右奇异向量给出），Key 在该方向上的投影可以精确近似期望注意力分数。与 baseline 的对比：
  
  | 缺陷 | Q-Filters 解决方案 |
  |------|-------------------|
  | K-Norm 仅用 L2 范数忽略角度信息 | Q-Filters 同时捕捉 Key 在 Query 主方向上的投影（含范数和角度），与注意力分数的 Spearman 相关性显著高于 K-Norm |
  | SnapKV 需物化注意力矩阵，与 FlashAttention 不兼容 | Q-Filters 仅需一次标量积投影，不访问注意力权重，完全兼容 FlashAttention |
  | StreamingLLM 固定保留 attention sink + 滑动窗口，丢弃中间关键信息 | Q-Filters 基于数据驱动的每头重要性估计，动态选择全局最重要的 KV pairs |
  | 许多方法需要微调 | Q-Filters 完全训练无关，仅需一次离线 SVD 校准（<3 分钟） |

  全栈执行例子（Q-Filters 对比 K-Norm baseline）：
  - **算法层**：(a) 离线从校准数据收集各层各头的 Query 激活矩阵 $Q^h$，SVD 分解得 Q-Filter $v_1^+$；(b) 推理时计算 $s_t^h = \langle K_t^h, v_1^+ \rangle$ 作为重要性得分，取 top-k 保留。定理保证 $\mathbb{E}(\langle Q_i^h, K_j^h \rangle) \approx \kappa^h \langle K_j^h, u^h \rangle$，其中 $\kappa^h > 0$ 为常数。对 GQA，组内 Q-Filters 取平均。这比 K-Norm 多捕捉了 Key 在 Query 主方向上的投影角度分量 $\cos(K_j^h, u^h)$。
  - **系统框架层**：基于 KVPress 库实现，作为 KV Cache 压缩的 plugin 插入 HuggingFace 推理 pipeline。Q-Filters 校准只需前向传播若干样本提取 Query 激活（无需反向传播），推理时在每次 KV Cache 更新后执行 top-k 筛选。论文未明确说明 Serving 框架层面的进一步修改。
  - **编译框架/算子层**：Q-Filters 的标量积计算 $K \cdot v_1^+$ 与 K-Norm 的 L2 范数计算复杂度相当（均为 $O(L \times d_H)$），但 Q-Filters 避免了 FlashAttention 之外显式物化注意力矩阵的需求。FlashAttention 兼容性意味着 kernel 执行路径更短：prefill 用 FlashAttention 高效计算 attention，压缩仅需额外的矩阵-向量乘法和 top-k，不破坏 FlashAttention 的内存优化。论文未明确说明编译框架或 kernel 层面的修改。
  - **硬件架构/芯片设计层**：论文未明确说明。
