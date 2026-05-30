## Unimportance Degree Score (Expert Importance Estimation for Mixed Precision Offloading)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Unimportance Degree Score 是 HOBBIT 提出的动态评估 MoE expert 重要性的方法，用于决定 cache-miss 时加载高精度还是低精度 expert。基于 MoE 输出公式 y = Σ G(x)_{e_i} E_{e_i}(x)，expert e_i 的贡献为 G(x)_{e_i}E_{e_i}(x)。由于无法在加载权重前计算 E_{e_i}(x)，使用 gating output ||G(x)_{e_i}|| 作为代理——实验验证二者 Pearson 相关系数为 0.99。将所有 top-K experts 按 ||G(x)_{e_i}|| 降序排列后，计算累积 unimportance score：s_{e_i} = Σ_{j=0}^{i-1} ||G(x)_{e_j}||（i>0），top-1 expert 始终 s=0。双阈值 T1/T2 划分三组：高精度（≤T1）、低精度（≤T2）、跳过（>T2）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Expert Importance Scoring Pipeline (per token, per MoE layer)
输入: x ∈ R^M (hidden state), W_gate ∈ R^{E×M} (gating weights), K=2

# Step 1: 标准 gating
gate_logits = W_gate @ x               # [E]
gate_probs = softmax(gate_logits)       # [E]
topk_vals, topk_ids = topk(gate_probs, k=K)  # top-2

# Step 2: 归一化 gate weights
gate_norm = topk_vals / sum(topk_vals)  # [K], 确保累积 ≤1

# Step 3: 按 gate weight 降序排列
sorted_order = argsort(gate_norm, descending=True)

# Step 4: 计算 unimportance degree score
scores = zeros(K)
cumulative = 0.0
for rank in range(K):
    e_idx = sorted_order[rank]
    scores[e_idx] = cumulative        # 累积前面所有 expert 的 weight
    cumulative += gate_norm[e_idx]

# Step 5: 双阈值决策 (T1=0.6, T2=0.9)
for i, e in enumerate(topk_ids):
    if scores[i] <= 0.6:
        load_high_precision(e)        # FP16/INT8
    elif scores[i] <= 0.9:
        load_low_precision(e)         # INT4/INT2
    else:
        skip_expert(e)                # 不加载

# 结果分布 (Mixtral-8x7B): 67% high / 30% low / 3% skip
# top-1 expert 始终 score=0，保持高精度 (50% 选择)
```

关键设计点：
- Pearson r=0.99 验证 ||G(x)|| 是 ||G(x)E(x)|| 的有效代理，避免计算 E(x) 的开销
- s_{e_i} 的累积性质确保：排位越低的 expert（gate weight 越小）得分越高，更可能被降精度或跳过
- top-1 expert 始终得分为 0，确保最重要的 expert 始终高精度
- 阈值通过 profiling 一次 ||G(x)|| 分布确定，无需逐样本调整

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 计算开销极小：||G(x)|| 就是 softmax 输出本身，无需额外计算
- 归一化：所有 ||G(x)_{e_i}|| 除以 Σ||G(x)_{e_j}|| 使 score 在 [0,1] 范围内
- 阈值设定：在 calibration dataset 上运行一次推理，收集所有 ||G(x)|| 值，按分位数确定 T1/T2。例如 Mixtral-8x7B 的 T1=0.6 覆盖 67% expert，T2=0.9 覆盖 97% expert
- 精度保持：GSM8K accuracy 从 0.52 降至 0.51（FP16→FP16+INT4），TruthfulQA 基本不变
- 变体（MoE-APEX）：相同公式但 cache policy 改用 LCU (Least Costly Used) 替代 LHU

涉及论文标题：
- HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference
- MoE-APEX: An Efficient MoE Inference System with Adaptive Precision Expert Offloading
