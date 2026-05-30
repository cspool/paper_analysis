## United Experts (United Expert Model / 联合专家模型)

术语解释
United Experts 是 BrownoutServe 提出的一种 MoE 模型蒸馏机制：通过知识蒸馏将每层多个原始 expert 的知识合并到单个同参数规模的 "united expert" 中，减少推理时的 expert 访问次数，提升 GPU 利用率和降低延迟。

术语是什么？
United Experts 的核心思想：在 MoE 推理中，大量 token 被路由到少数 hot experts，而多数 cold experts 只处理极少量 token，GPU SM 大量空闲。United Experts 将冷门 experts 的 token 聚合到 united expert 中批量处理，增大 effective batch size。

给定每 Transformer 层有 m 个原始 routed experts，按 way=k 分组得 ⌈m/k⌉ 组。每组训练一个 united expert：以组内 k 个原始 experts 为 teacher（输出 hidden states），united expert 为 student，MSE loss 最小化两者的 hidden states 差异：

$$\mathcal{L}_{\text{MSE}}^{j} = \frac{1}{k} \left( \sum_{i=0}^{k-1} \left\| H_{u}^{j} - H_{o}^{j \times k+i} \right\|^{2} \right)$$

其中 H_u^j 是第 j 个 united expert 的 hidden states，H_o^{j×k+i} 是第 j 组中第 i 个原始 expert 的 hidden states。训练完成后，每个 united expert 具备等价于 k 个原始 experts 的综合知识表达。

从算法pipeline角度拆解术语：
United Experts 的完整 pipeline 分为离线训练和在线推理两阶段：

**离线训练（以 Qwen1.5-MoE-A2.7B, m=60, k=8 为例）**：
```
# 每层 60 个原始 experts → ⌈60/8⌉ = 8 个 united experts
for each transformer layer:
    for group_j in [0, 7]:   # 8 groups
        UE_j = init_expert(hidden_dim)  # 与原 expert 同参数规模
        
        for batch in training_data:
            # Teacher: k=8 个原始 experts
            H_o_list = []
            for i in [0, 7]:
                expert_idx = j * 8 + i   # expert 0-7 for group 0
                H_o_list.append(Expert_{expert_idx}(batch))
            
            # Student: 1 个 united expert
            H_u_j = UE_j(batch)
            
            # MSE Loss
            loss = (1/8) * sum(||H_u_j - H_o||^2 for H_o in H_o_list)
            loss.backward()
            optimizer.step()

# 保存每层的 ⌈m/k⌉ 个 united expert 权重
```

**在线推理 - Partial-Brownout 调用**：
```
# BrownoutMoE forward path
输入: token hidden states x_t, threshold=0.7

# 1. Gate routing
for each token t:
    scores[t] = softmax(TopK(x_t @ E_centroids, K=2))

# 2. Token 统计与划分
expert_counts = [count tokens per expert]  # 60 个 experts
A = sort experts by count descending
S1 = experts with cumulative count >= total_tokens * threshold  # hot experts
S2 = experts for united expert processing                       # cold experts

# 3. S1: 原始 experts FFN
for e in S1:
    outputs += FFN_e(tokens_e)

# 4. S2: United experts FFN
for group in group_experts(S2, k=8):  # 按 index 分组
    concat_tokens = concat(all tokens in this group)
    outputs += UE_{group_id}(concat_tokens)
```

术语一般如何实现？如何使用？
- **训练方式**：知识蒸馏（teacher=原始 experts，student=united expert），使用 MSE loss。训练数据与下游任务相关。
- **存储**：United expert 权重与原 expert 等参数规模，需常驻 GPU 显存或在 CPU memory/disk 间切换。way=k 越大，united expert 数越少但每个 united expert 集成的知识越多（精度可能越低）。
- **代码**：BrownoutServe 开源（https://github.com/beyondHJM/BrownoutServe），但预训练 United Expert 权重需联系作者获取。
- **适用场景**：MoE 模型（Qwen1.5-MoE 等）、多 expert（≥8/layer）的模型效果更好。

涉及论文标题：
- BrownoutServe: SLO-Aware Inference Serving under Bursty Workloads for MoE-based LLMs
