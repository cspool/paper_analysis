## GatePro Parameter-Free Expert Selection Optimization for Mixture-of-Experts Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - GatePro 提出一种**无参数的 MoE gating 优化方法**，通过局部竞争机制直接提升 expert 选择的多样性（diversity），而非仅关注负载均衡。核心实现：(1) **Gate Similarity Computation**：计算 gating weight 矩阵各行向量之间的 cosine similarity 矩阵 S∈R^{N×N}，识别功能冗余的 expert 对；(2) **Localized Competition Mechanism**：对每个 expert i 找到最相似的 expert j*(i)=argmax_{j≠i} S_{ij}，按 token 级 logit 比较决定 winner/loser，对 loser 施加固定惩罚 λ=10^{-4} 抑制其被选中，防止功能相似的 expert 被同时激活。该方法无额外可学习参数，可 hot-swappable（训练中途启用/禁用），计算开销极小（cosine similarity O(N²d)，竞争选择 O(N)/token）。
  - 实验比较：(1) GatePro vs baseline MoE（含 load balance loss）在 Seed-MoE-0.7B/7B 和 Seed-MoE-1.3B/13B 两种规模上从 100B→1.2T tokens 的 pretrain 全程性能跟踪；(2) Continuous Training (CT) 阶段的性能对比；(3) 在开源 OLMoE-1B/7B 架构上的泛化验证；(4) Expert utilization analysis：zero token count 随训练步数的下降曲线；(5) Expert gating similarity analysis：average cosine similarity、average angle、spectral entropy 三个 diversity 指标；(6) Hot-swappable analysis：不同 GatePro→MoE 切换时间点的性能影响；(7) 不同 expert pool size：128 vs 256 experts 的扩展实验。

- 硬件平台是什么，配置是什么。
  - 训练硬件：8 节点，共 64 GPUs（论文未明确说明 GPU 型号，但基于 ByteDance Seed 基础设施推断为 NVIDIA H800/A100 级别）。分布式训练使用 FSDP (Zhao et al. 2023) 和 Flash Attention (Dao et al. 2022)。
  - OLMoE 实验：论文未明确说明具体 GPU 配置，但遵循 OLMoE 原版开源配置。

- 模型是什么。数据集和bench分别是什么。
  - 模型系列：
    - Seed-MoE-0.7B/7B：0.7B 激活参数 / 7B 总参数，top-k=6，默认 128 experts（扩展至 256）
    - Seed-MoE-1.3B/13B：1.3B 激活参数 / 13B 总参数，top-k=6，128/256 experts
    - OLMoE-1B/7B：开源 MoE 架构，遵循原版配置
  - 训练 tokens：Seed-MoE-0.7B/7B 最多 500B tokens；Seed-MoE-1.3B/13B 最多 1.2T tokens；OLMoE 训练 400B tokens。
  - 评估 benchmark：
    - 主要（Seed-MoE）：MMLU-Pro, MMLU, BBH, HellaSwag, GSM8K, MBPP
    - OLMoE 扩展：MMLU, HellaSwag, ARC-Challenge, PIQA, COPA
  - Hot-swappable 实验：GatePro-MoE 0.7B/14B, 256 experts, 500B tokens total。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未明确给出代码开源链接。论文发表于 2025 年 10 月，作者来自 ByteDance Seed + UC Berkeley。可参考 OLMoE 开源实现验证泛化性（https://github.com/allenai/OLMoE）。
  - GatePro 算法 pipeline 伪代码（基于 Algorithm 1）：

```python
# === GatePro MoE Forward Pass ===
# Input: token x ∈ R^d, gating weights W_g ∈ R^{N×d}, penalty λ=1e-4,
#        experts {E_1, ..., E_N} (each a FFN)

# Step 1: Original logits
logits = W_g @ x  # [N], router logits

# Step 2: Gate similarity matrix (pre-computed, updated periodically)
# S_{i,j} = <W_g[i], W_g[j]> / (|W_g[i]| * |W_g[j]|)  # cosine similarity
S = cosine_similarity(W_g)  # [N, N]
# 对角线除外: S[i, i] = -inf

# Step 3: For each expert i, find its most similar counterpart
j_star = argmax(S, dim=1)  # [N], j*(i) for each i

# Step 4: Localized competition
penalty_mask = zeros(N)
for i in range(N):
    if logits[i] < logits[j_star[i]]:
        penalty_mask[i] = -lambda  # loser gets penalized

# Step 5: Suppressed logits
logits_tilde = logits + penalty_mask  # [N]

# Step 6: Top-k expert selection on suppressed logits
topk_indices = topk(logits_tilde, k=6)  # k=6 in paper

# Step 7: Softmax over selected experts
alpha = softmax(logits_tilde[topk_indices])  # [k]

# Step 8: Sparse weighted combination
output = sum(alpha[j] * E[topk_indices[j]](x) for j in range(k))
return output
```

  - GatePro 关键张量计算流（inference 单 token，N=128 experts，k=6）：
    - input token x (1×d) → gating projection W_g·x (128) → GatePro penalty: 对 128 experts 逐一检查 loser (128 次比较) → suppressed logits (128) → top-6 selection → 6 expert FFN compute (forward: Linear[d→αd]→GeLU→Linear[αd→d]) → weighted sum → output (1×d)
    - 对比 baseline MoE：仅多了 cosine similarity 预计算 (offline/periodic) + per-expert penalty 比较 (online O(N))，无额外参数。
  - GatePro 的 hot-swappable 特性：`penalty_mask` 可在任意训练步通过标志位启用/禁用，模型权重完全不变。论文实验证明先用 GatePro 训练 400B tokens 再切换回标准 MoE 训练 100B tokens，性能几乎等同于全程 GatePro (500B) 训练，表明 GatePro 建立的 expert diversity 具有"训练遗产效应" (training legacy effect)。
