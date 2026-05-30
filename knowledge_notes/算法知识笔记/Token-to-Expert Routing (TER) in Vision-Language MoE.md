## Token-to-Expert Routing (TER) in Vision-Language MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Token-to-Expert Routing (TER) 是 MoE 架构中 Router（门控网络）决定每个输入 token 分配给哪个/哪些 expert 处理的核心机制。Router 是一个小型可学习线性层 `W_g ∈ R^{D × K}`（D = hidden dim, K = num_experts），对每个输入 token x 计算 `logits = x @ W_g`，经 Softmax 归一化得到 routing probabilities `P(x) ∈ R^K`，Top-k 选择后对被选中 expert 输出加权求和：`MoE(x) = Σ P(x)_i * Expert_i(x)`。

在 LVLM 场景下，TER 同时处理两类模态 token：CLIP visual encoder 提取的 vision tokens（~576 per image）和 language tokens（输入 text sequence）。传统 TER 对所有 token 统一施加 load balancing 约束 `L_balancing = K * Σ F_i * G_i` 以鼓励均匀路由分配，但 LTDR 发现 vision tokens 服从 long-tailed distribution（少量高信息 foreground + 大量低信息 background），load balancing 将 critical foreground tokens 打散到不同 expert，阻碍 expert 专业化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

**统一 TER Pipeline（Baseline MoE-LLaVA）**：
```
# Vision + Language token concatenation
x = [v_1,...,v_M, t_1,...,t_N]  # (M+N)×D

# Router forward: all tokens share same W_g
logits = x @ W_g                  # (M+N)×K
probs = Softmax(logits)           # (M+N)×K
selected = TopK(probs, k=2)       # select k=2 experts per token
norm_probs = probs[selected] / sum(probs[selected])

# Load balancing: uniform constraint on ALL tokens
for i in 1..K:
    F_i = fraction of (M+N) tokens routed to expert i
    G_i = mean(probs[:, i])
L_bal = K * Σ F_i * G_i  # applied to vision + language equally

# Expert computation
output = Σ norm_probs[j] * ExpertFFN_j(x)
```

**LTDR TER Pipeline（Modal-aware + Long-tailed aware）**：
```
# Step 1: Router forward (same as baseline)
logits_v = V @ W_g  # M×K, vision
logits_t = T @ W_g  # N×K, language

# Step 2: MsDaR - Language-only load balancing
for i in 1..K:
    F_i = count(T routed to i) / N    # only language tokens
    G_i = mean(softmax(logits_t)[:, i])
L_bal = K * Σ F_i * G_i  # vision tokens excluded!

# Step 3: VsDEA - Vision tail token identification
RPV = Variance(softmax(logits_v), dim=1)  # per-vision-token RPV
threshold = Mean(RPV)                       # dynamic threshold
is_tail = RPV > threshold                   # ~13% of vision tokens

# Step 4: Differentiated expert activation
for each token x:
    if x is vision tail token:
        # activate a=4 (LTDR) > k=2 (baseline) experts
        indices = TopK(probs(x), a=4)
    else:
        indices = TopK(probs(x), k=2)
    weights = Softmax(logits(x)[indices])  # renormalize
    output = Σ weights[j] * ExpertFFN_j(x)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **标准实现**：`nn.Linear(D, K)` + Softmax + TopK。Mixtral-8x7B 用 Top-2 gating，DeepSeek-V3 用 Sigmoid 替代 Softmax 扩展至 256 experts
- **LVLM MoE 实现**：MoE-LLaVA 将指定层 FFN 替换为 MoE layer（每 2 个 Transformer block 中 1 个），4 experts Top-2；Molmo 使用 64 experts Top-8
- **LTDR 改动**：对现有 MoE 框架改动极小——仅修改 `L_balancing` 计算范围（排除 vision tokens）和 tail token 的 TopK 参数（k→a）。与 HuggingFace Transformers + PyTorch 完全兼容，不需额外框架支持
- **训练配置**：epoch=1, LR=2e-5 cosine, weight decay=0, batch size=16/GPU, FP16, L_balancing coefficient=0.01, a=4 (MoE-LLaVA) / a=12 (Molmo)

涉及论文标题：
- Long-Tailed Distribution-Aware Router For Mixture-of-Experts in Large Vision-Language Model
