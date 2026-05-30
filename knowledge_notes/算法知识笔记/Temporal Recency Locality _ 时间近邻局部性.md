## Temporal Recency Locality / 时间近邻局部性

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Temporal Recency Locality 是 LessIsMore 论文在推理模型注意力模式中发现的第二个稳定局部性结构：在长程推理的每个 decoding step 中，最近生成的 token（recently generated tokens）始终获得异常高的 attention score，且注意力分布中分配给近邻 token 的比例在整个 decoding 过程中保持稳定。具体表现为：(1) 在任意 decoding step 的 attention 分布中，最后 ~25% 的 token 获得约 25% 的总 attention mass；(2) 这个比例在不同 token budget（2K-8K）、不同 decoding step（1K-32K）、不同推理任务（AIME-24/25, GPQA）中保持稳定（论文附录 A.6，图 9-10）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Temporal Recency Locality 的定量刻画**（基于 Qwen3-8B, 多种 token budget）：
1. 在任意 decoding step t，对所有 32 heads 计算 attention distribution A_i[t]：
   $$A_i[t] = \text{softmax}(q_t K_{1:t}^T / \sqrt{d}) \in \mathbb{R}^{1 \times t}$$
2. 定义 recency ratio r_obs(t) = sum of A_i[t] on last W tokens / total attention mass
   其中 W = t（随序列增长递增）
3. 关键发现：r_obs(t) 在各 decoding step 中维持在 ~0.2-0.3（约 20-30% attention mass 分配给最近的约 25% tokens）。
4. 与 token budget K 的关联：近邻 token 占总关键 token 的比例保持恒定（recency ratio），不受 budget 绝对大小影响（图 8）。

**与 StreamingLLM 固定窗口的关键区别**：
- StreamingLLM：固定大小 sliding window（如最近 256 tokens），不随 token budget 变化
- LessIsMore Stable Recency Window：固定比例 r=0.25，即 K·r 个 token，随 budget K 自适应缩放
- 原因：Temporal recency locality 显示比例关系恒定（~25%），而非绝对大小恒定

术语一般如何实现？如何使用？

Temporal Recency Locality 直接导致 LessIsMore 的 Stable Recency Window 设计：在 CUSA 的 token 选择中，固定比例 r=0.25 的 token budget 专门分配给最近 K·r 个 token。这个设计确保：(1) 无论使用多少 token budget，近邻上下文始终占固定比例；(2) 历史 token 和近邻 token 之间的资源分配是动态平衡的，反映推理中"逐步构建在前一步基础上"的增量性质。消融实验（附录 A.1.2，图 8）验证：仅 25% 近邻 + 75% cross-head 选择的组合达到最高 attention recall 并成功解题；纯近邻（100%）因丢弃长程上下文而 recall 最低；纯 cross-head 但 0% 近邻也无法解题——证明推理同时依赖长程依赖和逐步近邻推理。

涉及论文标题：
- Less Is More: Fast and Accurate Reasoning with Cross-Head Unified Sparse Attention
