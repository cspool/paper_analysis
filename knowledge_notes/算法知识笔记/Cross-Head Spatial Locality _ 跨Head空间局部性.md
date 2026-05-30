## Cross-Head Spatial Locality / 跨Head空间局部性

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Cross-Head Spatial Locality 是 LessIsMore 论文在推理模型注意力模式中发现的第一个稳定局部性结构：在长程推理（long-horizon reasoning）的每个 decoding step 中，不同 attention head 的 top-k 重要 token 排名存在显著重叠。具体表现为：在同一 K-V group（GQA 模型）内，多个 query head 的 top-4K token 集合高度重叠（黄色区域），且跨所有 attention head 也存在大量共同关注 token（红色区域，图 2）。这一现象与传统假设——不同 attention head 功能高度特异化、需要独立的 token subset——直接矛盾。Cross-head spatial locality 在模型的各层和各 decoding step 中持续存在（论文附录 A.6，图 9-10）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Cross-Head Spatial Locality 的定量刻画**（基于 Qwen3-8B, AIME-24）：
1. 在某一 decoding step（如第 20K 步）和某一层（如 Layer 4）：
   a. 对每个 head i，计算其 ground-truth top-4K token 集合 T_i
   b. 计算 pairwise Jaccard similarity: J_{ij} = |T_i ∩ T_j| / |T_i ∪ T_j|
   c. 对同一 KV group 内的 4 个 heads 计算平均 Jaccard：~0.6-0.8（黄色区域，图 2）
   d. 对所有 32 heads 计算全局平均 Jaccard：~0.4-0.6（红色区域，图 2）
2. 追踪不同 decoding step（10K/15K/20K/25K 步）：重叠区域保持稳定（附录图 9）。
3. 追踪不同层：重叠模式从早期层到后期层持续存在。
4. 追踪不同任务（AIME-24/25, GPQA）：重叠模式跨任务一致。

**与传统假设的对比**：
- 传统假设：head i 关注 token A/B/C, head j 关注 token X/Y/Z → 需要独立 token 子集
- 实际观察：head i 和 head j 的关注 token 高度重叠 → 全局统一 token 子集同样有效
- 关键推论：按 head 独立 top-k 选择不仅冗余（相同 token 被多个 head 重复选择浪费 budget），还会引入 head 特定噪声（个别 head 的"错误"选择不被全局一致性纠正）

术语一般如何实现？如何使用？

Cross-Head Spatial Locality 被 LessIsMore 直接用于推导 CUSA 的跨 head 统一 token 选择机制：(1) 各 head 仍独立提案（保留 head 之间的细微差异），但 (2) 通过 UnionFlatten 聚合后全局排名，使得被多数 head 认同的 token 优先保留，个别 head 的噪声选择被自然淘汰。跨 head 空间局部性也是 CUSA 低频重选可行的理论基础——因为 token 重要性是全局一致的，早期层的选择可在后期层复用而不退化。

涉及论文标题：
- Less Is More: Fast and Accurate Reasoning with Cross-Head Unified Sparse Attention
