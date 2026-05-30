## AffinityBinning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

AffinityBinning 是 LYNX 提出的用于 MoE 推理中 batch 级别动态专家选择的离散化技术。核心思想：将每个 token 对每个 expert 的 router 置信度（通过 log-ratio 到 top-1 expert 衡量）离散化为有限数量的 bin，bin 的宽度和数量仅由模型架构的 sparsity ratio (k/N) 决定，而非 workload 或 task。这使得 LYNX 成为 self-calibrating 系统——自动适配任何 MoE 架构，无需 profiling 或 tuning。

具体实现：对于每层的每个 token，计算相对于 top-1 expert 的 log-ratio：log_ratio(e) = logit[e] - logit[top1]（即 softmax 概率比的对数等价形式）。然后将这些值按 α（bin width 的倒数控制参数）和 β（最大 bin 数限制参数）离散化：bin[e] = clamp(floor(log_ratio[e] * α), -β, 0)。bin=0 表示与 top-1 亲和力最高，bin 越负表示亲和力差距越大。对于 sigmoid-based router（如 DeepSeek），使用 pre-sigmoid scores 的差值代替 log-ratio。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

AffinityBinning 在 LYNX MoE 推理 pipeline 中的位置和伪代码：

```
# 输入：batch size B, router logits L[B][N], N experts, top-k
# 输出：per-token bin assignments B[B][k]

# Step 1: Router 前向（标准 MoE）
for t in 0..B-1:
    probs[t] = softmax(L[t])          # [N]
    topk_idx[t], topk_prob[t] = topk(probs[t], k)

# Step 2: AffinityBinning（LYNX 核心）
α = compute_alpha(k/N)   # 由 sparsity ratio 决定
β = compute_beta(k/N)    # 通常 5-8

for t in 0..B-1:
    top1_logit = max(L[t])
    for each expert e in topk_idx[t]:
        log_ratio = L[t][e] - top1_logit     # router logits 之差
        bin[t][e] = clamp(floor(log_ratio * α), -β, 0)

# Step 3: Batch-level Adaptive Scoring（使用 bin 值）
for each expert e:
    score[e] = 0
    for t in 0..B-1:
        if e in topk_idx[t]:
            score[e] += B^{bin[t][e]}  # batch_size 为底数的指数加权

# 效果：高置信度 token (bin=0) 贡献 B^0=1
#       低置信度 token (bin=-5) 贡献 B^{-5}≈0.00006 (B=16)
#       被多个高置信度 token 偏好的 expert 得分指数级更高
```

关键参数：
- Qwen2-57B (k=8, N=64, k/N=0.125): α 产生约 6 个 bin (β=5)
- Mixtral-8x7B (k=2, N=8, k/N=0.25): 更宽的 bin (更少的划分)
- DeepSeek-V3 (k=8, N=256, k/N=0.03): 更细的 bin (更多的划分)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LYNX 将 AffinityBinning 实现为 Triton fused kernel（Kernel 1 — Token-wise Binning），拦截 vLLM 每层 MoE router 输出后执行。kernel 对 batch 中所有 token 并行执行 log-ratio 计算和 discrete binning，融合了原本需数百个 PyTorch element-wise ops（subtract, floor, clamp）。α 和 β 在模型加载时根据 sparsity ratio 计算一次，随后所有 forward pass 重用。

涉及论文标题：
- LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection
