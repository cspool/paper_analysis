## GatePro

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

GatePro 是 ByteDance Seed 提出的一种无参数（parameter-free）MoE gating 优化方法，通过局部竞争机制直接提升 expert 选择的多样性（diversity），而非仅关注 token 分配的负载均衡。核心思路：(1) 计算 gating weight matrix W_g 各行向量间的 cosine similarity matrix S_{ij} = ⟨w_{g,i}, w_{g,j}⟩ / (|w_{g,i}|·|w_{g,j}|) 来识别功能相似的 expert 对；(2) 对每个 expert i 找到最相似的 j*(i) = argmax_{j≠i} S_{ij}；(3) 在 token 级根据 logit 比较决定竞争 winner，对 loser 施加固定惩罚 λ=10^{-4} 抑制其激活。该方法无额外可学习参数，可 hot-swappable（训练中途启用/禁用），计算开销极小（cosine similarity O(N²d)，per-token competition O(N)）。GatePro 与辅助平衡损失（LBL）互补而非替代——LBL 保证 token 分配的统计均衡，GatePro 保证 expert 选择的功能多样性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

GatePro 在标准 MoE 的前向传播中插入 gate-level competition：

```
# === GatePro MoE Forward Pass ===
# Input: token x, gating weights W_g, penalty λ=1e-4

# Step 1: Original logits
logits = W_g @ x  # [N], N=128/256

# Step 2: Gate similarity (periodically updated, not per-token)
S = cosine_similarity(W_g)  # [N, N], diagonal set to -inf

# Step 3: For each expert i, find most similar counterpart
j_star = argmax(S, dim=1)  # [N]

# Step 4: Localized competition — loser gets penalty
l_competitor = gather(logits, j_star)  # competitor logits
mask = (logits < l_competitor)         # loser positions
logits_tilde = logits + mask * (-lambda)  # apply penalty

# Step 5-8: Standard top-k + softmax + weighted combination
topk_idx = topk(logits_tilde, k=6)
alpha = softmax(logits_tilde[topk_idx])
output = sum(alpha[j] * E[topk_idx[j]](x) for j in range(6))
```

实验数据：Seed-MoE-0.7B/7B (128 experts): MMLU-Pro 21.8% vs baseline 20.5% (500B tokens); GSM8K 45.0% vs 43.0%。Seed-MoE-1.3B/13B (1.2T tokens): MMLU-Pro 31.6% vs 30.6%, BBH 50.7% vs 49.8%。OLMoE-1B/7B (400B tokens): Overall 62.5% vs 61.8%。Expert utilization: Layer 14 零激活 convergence 从 3000 steps 缩短至 1500 steps。256 experts 下深层的加速优势更显著。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GatePro 以 hook 形式注入现有 MoE 代码的 gating 模块：在 top-k 选择前插入 competition penalty 计算，维护 gating similarity buffer 周期性更新 S 矩阵，使用 boolean flag 控制 hot-swap。适用场景：MoE pretrain（N≥64 experts）、continuous training、深层 MoE 层的 diversity 增强。可与任何 top-k routing 方案（softmax、sigmoid）兼容。

涉及论文标题：
- GatePro Parameter-Free Expert Selection Optimization for Mixture-of-Experts Models
