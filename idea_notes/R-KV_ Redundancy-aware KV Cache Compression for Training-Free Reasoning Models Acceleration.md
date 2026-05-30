## R-KV: Redundancy-aware KV Cache Compression for Training-Free Reasoning Models Acceleration

- baseline方法是什么？
  Baseline 是标准 attention-based KV cache eviction 方法，以 SnapKV 为代表。SnapKV 的 token 选择完全依赖 attention scores：计算最后 α 个 observation tokens 对 key tokens 的注意力，通过滑动窗口 max-pooling 稳定化后取平均作为 per-token importance score，保留 importance 最高的 B_budget 个 token。核心缺陷：推理模型（如 DeepSeek-R1）的长 CoT 生成中含有大量冗余内容——反复的自我验证、迭代推理、冗长的自言自语。这些重复内容"self-attend 到自己"，产生高 attention score，导致 SnapKV 保留大量语义冗余 token，挤占了真正关键的推理中间步骤。论文观察表明：推理模型生成长度是 ground truth 的 8-14×，1-gram 和 2-gram 重复频率是 ground truth 的 5-7×（Fig. 2）。在 SnapKV 的 selected token 可视化中（Fig. 3），大量被选中的 token 集中在对同一结论的反复重述（如 "10% of 30 is 3. So 3 students are leaving early" 被重复数十次并被高 attention 选中）。

  全栈执行例子（SnapKV baseline, DeepSeek-R1-Distill-Llama-8B, 16K generation, A100 80G）：
  - **算法层**：每 B_buffer=128 步触发压缩。选取最后 α=8 个 tokens 作为 Q_obs，对 GQA 的各 query head 分别计算 attention A^{h,g} = softmax(Q^{h,g}·(K^h)^T/√d)，对所有 query head 做 mean-pooling 聚合得到 final attention matrix。滑动窗口 max-pooling 后取均值得到 I_i^h = (1/α)·Σ_{j=0}^{α-1} max(A_{j,i-W:h,i+W})。选择 I_i^h 最高的 B_budget 个 token。问题：如果一个冗余 token 反复出现，同一个 key vector 会被周围的 query token 频繁 attend（因为语义高度相似），导致 I_i^h 膨胀，over-retain 冗余内容。
  - **系统框架层**：未修改 serving 框架，直接在 HuggingFace Transformers 的 forward pass 中插入 KV cache selection 逻辑。压缩后内存需要重新分配（移动压缩后的 KV cache），可能引入内存管理 overhead。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 PyTorch matmul + attention kernel。压缩操作（top-k 选择）为 PyTorch CPU/GPU 操作，非定制 CUDA kernel。SnapKV 与 R-KV 在 kernel 层等价——差异仅在 selection score 的计算方式。
  - **硬件架构层**：NVIDIA A100 80G。无专用硬件修改。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  R-KV 在 attention-based importance scoring 的基础上额外引入 **redundancy estimation via cosine similarity of key vectors**，通过 joint selection score Z = λ·I − (1−λ)·R 同时平衡重要性（I）和去冗余性（R），从根源解决 SnapKV 的冗余过保留问题。

  **(1) 算法层——冗余感知的 Joint Selection Score**：
  R-KV 的三阶段计算：

  **阶段 A：Importance Scoring**（继承并改进 SnapKV）
  - 对 GQA 使用 max-pooling 替代 mean-pooling 聚合 query head 的 attention（实现细节，详见 Appendix A.2）。Max-pooling 能更好地保留每个 query head 中最重要的 token，避免 mean-pooling 抹平关键信号。
  - 滑动窗口 max-pooling 稳定化 + 取均值得到 I_i^h（与 SnapKV 相同）。
  - 这一阶段确保 critical reasoning context（如问题中的数值、关键中间步骤）被保留。

  **阶段 B：Redundancy Estimation via Key Vector Cosine Similarity**
  - 这是 R-KV 的核心创新。对每层每 head，将 key vectors L2 归一化后计算余弦相似度矩阵 S = K̄ K̄^T ∈ R^{n×n}（Eq. 5）。
  - 对角线置零（防止 self-redundancy）；对每个 token i，找到 S_{:,i} > T 的高相似 token 集合，保留其中最近的 β 个（largest indices），将其 similarity 置零——这确保即使 token 高度重复，最近出现的那几个仍被保留（因为它们接近当前解码位置，contextual relevance 更高）。
  - 计算每 token 的平均相似度 S̄_i = mean(S_{:,i})，再通过 softmax 归一化得到 R_i^h ∈ [0,1]（Eq. 6）。高 R_i^h 表示 token i 的 key vector 与许多其他 token 高度相似 → 冗余。

  核心洞察：冗余 token 的 key vectors 在向量空间中高度聚集。通过余弦相似度矩阵 S，R-KV 在向量空间层面（而非 token 表面）捕捉语义冗余。这解决了 SnapKV 仅看 attention weight（标量）无法区分的"高 attention 但高度冗余"的 token：这些 token 的 I_i^h 高但 R_i^h 也高，joint score Z_i^h = λ·I_i^h − (1−λ)·R_i^h 被拉低，从而在 top-B_budget 选择中自然被淘汰。

  **阶段 C：Joint Selection 与跨 Head 聚合**
  - Z_i^h = λ·I_i^h − (1−λ)·R_i^h（Eq. 7），λ=0.1（通过消融确定，Fig. 5-6）。
  - 跨 head 聚合：AggScore_k = mean_h(Z_{k,h})，取 top-B_budget 保留。
  - 为何 λ 偏小（0.1）？因为 I_i^h 的值分布高度稀疏（少数 outlier 主导），而 R_i^h 经过 softmax 后分布较均匀。λ=0.1 时 redundancy 项的权重(1−λ)=0.9 足以有效抑制冗余，λ 增大到 0.5 以上则退化为近似纯 attention-based selection。

  **(2) 对比 Baseline 的改进**：

  | 维度 | SnapKV（baseline） | R-KV（论文方法） |
  |------|-------------------|-----------------|
  | Token 选择信号 | 仅 attention weight（I_i^h） | Joint: λ·I_i^h − (1−λ)·R_i^h |
  | 冗余检测 | 无。冗余 token 常因高 attention 被误保留 | Key vector 余弦相似度显式测量冗余 |
  | GQA 聚合 | Mean-pooling of attention scores | Max-pooling（更好保留每 query head 关键 token）|
  | 近期 token 保护 | 仅保留最后 α 个 observation tokens | observation tokens + 显式保留最近 β 个高相似 token |
  | 推理模型适配 | 未针对长 CoT 冗余特性优化 | 专门针对推理模型的重复/反射模式设计 |
  | AIME24 (R1-Llama-8B, 10% budget) | ~20% pass@1 vs FullKV 49.79% | ~51.56% pass@1（lossless @1536 budget）|

  **(3) 全栈执行例子（R-KV, DeepSeek-R1-Distill-Llama-8B, 16K generation, A100 80G）**：
  - **算法层**：每 128 tokens 触发压缩（B_buffer=128, B_budget=1536, α=8, λ=0.1）。Importance scoring：对 32 heads × 8 attention groups (GQA group_size=4)，每个 GQA group 内 4 个 query heads 独立计算 attention → max-pooling 聚合 → sliding window max-pooling → 得到 I_i^h ∈ R^{1536+128-8=1656}。Redundancy estimation：对同一批 1656 个 key vectors L2 归一化 → 1656×1656 余弦相似度矩阵 → 抑制对角线和最近 β 个高相似 token → 行均值 → softmax 归一化 → R_i^h。Joint: Z_i^h = 0.1·I_i^h − 0.9·R_i^h → mean 跨 head 聚合 → 取 top 1536 个 token + 8 observation tokens = 1544 KV tokens 保留。相比 FullKV 的 16000 KV tokens，压缩比 ~10%，节省 ~90% KV cache 内存。
  - **系统框架层**：未修改 serving 框架。论文在 Limitations（Appendix D）中明确指出现有 serving 框架（如 vLLM）若不提供 KV cache compression 专用接口，则需要 reallocate 内存来存放压缩后的 cache 并 deallocate 原始 cache，这会引入额外开销。论文表示通过 dedicated KV compression interfaces 可以避免此问题。目前实现为 HuggingFace 级别的 prototype。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 PyTorch 操作。重要性计算 overhead O(α·B_budget)（~8·1536 量级），冗余估计 overhead O(B_budget²)（~1536² 量级），总计约 ~2.4M FLOPs per compression step（每 128 tokens 一次），远小于 attention 计算量。无定制 CUDA kernel。
  - **硬件架构层**：NVIDIA A100 80G。R-KV 通过压缩 KV cache 释放 memory 来增加 batch size（因为 batch size 受 KV cache 内存限制），从 batch=30 (FullKV @16K) 提升至 batch=402 (R-KV @16K, fixed budget=1024)，端到端 throughput 从 347 tok/s 提升至 3189 tok/s（9.2×）。单独 batch=1 时 throughput 提升有限（80.95 vs 69.41 tok/s），证明 primary gain 来自 batch size scaling 而非 per-step latency reduction。

  效果量化（R1-Llama-8B）：
  - MATH-500: 34% KV cache budget → lossless（82.34% vs FullKV 82.38%）, 16% budget → 105% of FullKV
  - AIME24: 10% KV cache budget → lossless（51.56% vs FullKV 49.79%, pass@1）, 16% budget → 52.29%（超越 FullKV）
  - 8K generation, 10% ratio budget: 90% memory saving, 479 vs 62 max batch size, 6.6× throughput
  - 16K generation, fixed 1024 budget: 93.75% memory saving, 402 vs 30 max batch size, 9.2× throughput
