## SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers

- baseline方法是什么？
  Baseline 是 PyramidKV / PyramidInfer 等基于注意力权重的 token eviction 方法。这些方法的核心思想是：计算每个 token 的累积注意力分数（accumulated attention score），根据分数从低到高淘汰 token，且在各层间采用金字塔形分配（浅层保留多、深层保留少）。Baseline 全栈执行过程：

  **算法pipeline**：对每个 prefill request，在每层计算完整 attention 矩阵 $A = \text{softmax}(QK^T/\sqrt{d_h})$，然后基于观察窗口 $l_w$ 内的累积注意力分数 $ac_{i,a} = \sum_{b=l-l_w}^{l-1} A_{i,a,b} / (l-a)$ 选择 Top-K token 保留 KV cache。GQA 模型中对同组内所有 Q head 的 ac 取平均（这会丢失 per-head 精度）。每层的保留 token 数按 $\lambda$ 层深线性递减（金字塔形）。后续 decode 阶段仅使用保留的 KV cache 子集计算 attention。

  **系统框架 (Serving)**：论文未明确说明 serving 框架。推理过程为标准 HuggingFace Transformers 流程：prefill 阶段计算全量 attention 并 evict 低分 token → decode 阶段仅在保留的 KV cache 子集上计算 attention。

  **编译框架**：论文未明确说明。

  **kernel调度**：论文未明确说明。eviction 操作通过 PyTorch 张量索引完成（argTopK + gather）。

  **硬件架构/芯片设计**：论文未明确说明。仅提到在单张 RTX 3090 GPU 上测推理速度。

  Baseline 缺陷：(1) 浅层压缩效果差——浅层 attention 分布均匀（不稀疏），eviction 会丢弃大量仍有用的 token；(2) GQA 兼容性差——对同组内 Q head 取平均 ac 后统一淘汰会丢失细粒度 head 差异信息；(3) 忽略了浅层 KV cache 的"构成性冗余"（constituent redundancy），即不同 token 的 KV 向量之间存在高余弦相似性，可通过分解为基础向量组合来压缩。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  SpindleKV 的核心设计是"双重策略平衡浅层和深层"：(1) 深层继续使用 attention weight based eviction（与 baseline 一致但有 GQA 优化）；(2) 浅层使用基于余弦相似度的 codebook replacement 压缩构成性冗余；(3) GQA 处理通过"先 unfold（repeat KV vectors）→ eviction → codebook 压缩"三步走，利用 unfold 引入的冗余被后续 codebook 压缩消除。

  **算法pipeline**：
  - 深层 eviction：与 PyramidKV 类似的金字塔形分配 $r_c(\lambda) = r_c(0) + (r_c(m-1)-r_c(0))/(m-1) \cdot \lambda$，但 GQA 模型选择 unfold KV head（repeat $h_n$ 次）而非取平均 ac，从而避免 per-head 信息丢失。
  - 浅层 codebook：计算所有保留 KV token 的两两余弦相似度矩阵 $S_\Gamma$，设阈值 $\theta_K=0.98, \theta_V=0.95$ 构建邻接图 $G_\Gamma$。贪心迭代：每次选图中度数最高节点加入 codebook $C_\Gamma$，将其邻居节点全部映射到该 codebook entry，通过 `matmul(¬G_Γ[ι]^T, ¬G_Γ[ι])` mask 从图中移除已处理节点。同时记录每个 token 的 L2 magnitude $m_\Gamma$ 和 codebook 引用索引 $r_\Gamma$。最终存储开销 = |codebook entries| + |indices (int)| + |magnitudes (float)|，远小于原始 KV cache。
  - 推理重建：$\Gamma_r = C_\Gamma[r_\Gamma] \otimes m_\Gamma$，然后对重建的 K 重新应用 RoPE（论文论证 RoPE 是稀疏矩阵乘法，不增加显著时间开销）。
  - GQA 全流程：unfold KV → eviction → codebook → 压缩。Unfold 增加的 KV cache 大小被后续 codebook 压缩抵消（unfold 引入的重复向量余弦相似度为 1，极易被 codebook 合并）。

  **系统框架**：论文未明确说明。

  **编译框架/kernel调度/硬件架构/芯片设计**：论文未明确说明。

  为什么有效：浅层的 KV cache 中 token 向量之间余弦相似度极高（超过 0.9），这些 token 虽然 attention 分数不稀疏（eviction 难以淘汰），但其 KV 向量可被少数几个"基础向量"（codebook entries）线性表示。深层的 attention 存在强稀疏性，eviction 即可有效压缩。两者互补，使得在不同 KV cache 保留率下均优于 PyramidKV/PyramidInfer。例如在 LLaMA3-8B 上 40% KV cache 保留率时，SpindleKV LongBench 平均分 41.13 vs PyramidKV 39.86；在 15% KV cache 保留率时，Needle-in-a-Haystack 准确率 0.979 vs PyramidKV 0.938。
