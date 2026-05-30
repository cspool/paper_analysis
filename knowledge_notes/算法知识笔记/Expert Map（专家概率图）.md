## Expert Map（专家概率图）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Map 是 FineMoE 提出的核心数据结构，用于在 **iteration-level**（而非 request-level）追踪 MoE 模型中 gate network 对所有 experts 的选择偏好。每个 expert map 记录一次 inference iteration 中所有 L 个 MoE 层的 gate network 输出的完整概率分布 P_l^{(i)} ∈ R^J（而非 binary activation 或 hit count），其中 map_i = {P_1^{(i)}, ..., P_L^{(i)}}，P_l^{(i)} = {p_{l,1}^{(i)}, ..., p_{l,J}^{(i)}}, Σp = 1。直观上，expert map 不仅记录"哪些 experts 被选择"，更捕获了 gate network 对每个 expert 的 confidence/preference 分布——包含 "expert A 以 0.65 概率被选，expert B 以 0.20 概率被选" 等细粒度 confidence 信息。

与 MoE-Infinity Expert Activation Matrix 的关键区别：
- Activation Matrix: request-level, binary hit count（“expert_3 activated 5 times”）
- Expert Map: iteration-level, full probability distribution（“iteration i, layer l: expert_3 p=0.65”）
Expert Map 可通过 top-K selection + iteration aggregation 退化恢复 Activation Matrix，因此是 Activation Matrix 的 generalization。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Expert Map 构建流程（单 iteration，Mixtral-8×7B, L=32, J=8, K=2）：

for l in range(L):
    # Step 1: self-attention
    attn_out = self_attention_layer[l](hidden_states)

    # Step 2: gate network 输出 probability distribution
    logits = gate_network[l](attn_out)           # R^{J=8}
    P_l = softmax(logits)                         # R^{J=8}, Σp = 1
    # 例: P_l = [0.02, 0.45, 0.03, 0.01, 0.38, 0.05, 0.04, 0.02]

    # Step 3: top-K expert selection (用于实际计算)
    top_k_experts = topk(P_l, K=2)               # 例: [1, 4] (expert_1:0.45, expert_4:0.38)

    # Step 4: expert computation
    expert_out = sum(expert[e](attn_out) for e in top_k_experts)

    # Step 5: 记录 expert map 条目
    map[l] = P_l  # 完整概率分布，不只是 top-K

# 最终 expert_map = {P_0, P_1, ..., P_31} ∈ R^{32×8}
```

Expert Map 的关键优势：
1. Fine granularity: per-iteration 而非 per-request → Shannon entropy 低 → 可预测性高
2. Probability information: 不仅知道哪些 experts 被选，还知道 gate network 对各 expert 的 confidence
3. Degradability: top-K + 聚合 = 退化恢复 Activation Matrix，保证向后兼容
4. Trajectory comparability: probability distributions 向量可直接 cos_sim 比较

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FineMoE 用 PyTorch/NumPy ndarray 存储 expert maps。每个 map 包含：(1) L×J float32 概率值，(2) 1×d_model semantic embedding，(3) 可用于 trajectory comparison 的 flattened probability vector。Map Store 容量 1K maps（<200MB CPU memory）。去重：通过 unified redundancy score 计算 pairwise redundancy，保留覆盖更多 pattern 空间的 map 集合。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading
