## Layer-wise Synchronized Execution in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Layer-wise Synchronized Execution（逐层同步执行）是 MoE 推理的固有执行模型：transformer 的前向传播严格遵守 layer-by-layer 的顺序，每一层内 attention 和 expert 计算之间存在同步屏障。对于每个 layer ℓ：(1) 所有 data-parallel AW 独立执行 attention 计算（生成 token embeddings）；(2) 每个 AW 通过 gating network 选 top-k experts，将 token embeddings 发送到对应 EWs；(3) 所有 AW 等待所有选中 experts 返回输出（同步屏障），加权聚合后才进入 layer ℓ+1。EW 侧也遵守此模式——按 layer-wise batch 聚合同层同 expert 的 tokens，完成当前层后才前进到下一层。这一同步模式是保证 GPU 批处理效率的关键（避免碎片化的 per-request 执行导致 GPU 低利用率），但也意味着任一 worker 故障会导致整条 pipeline 在同步屏障处停滞。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
for layer ℓ in 1..L:                    // L 层 transformer
    // === All AWs (data-parallel) ===
    for each AW in parallel:
        // Attention computation
        Q, K, V = W_Q@h, W_K@h, W_V@h
        attn_out = softmax(Q@K^T/√d_k) @ V
        h_mid = LayerNorm(h + attn_out)
        
        // Gating (select top-k experts)
        gate_scores = softmax(W_gate @ h_mid)
        top_k_experts = TopK(gate_scores, k=2)
        
        // Dispatch to EWs (scatter)
        for each expert e in top_k_experts:
            rdma_send(EW[e], token_embedding=h_mid, weight=gate_scores[e])
    
    // === Synchronization Barrier (ALL AWs wait for ALL experts) ===
    
    // === All EWs (expert-parallel) ===
    for each EW in parallel:
        for each hosted expert e:
            batch = gather_tokens_from_AWs(layer=ℓ, expert=e)
            expert_out = FFN(batch)  // Linear1 → GeLU → Linear2
            for each token in batch:
                rdma_send(token.source_AW, expert_out[token] * token.weight)
    
    // === Synchronization Barrier ===
    
    // === Back to AWs ===
    for each AW:
        h_next = aggregate_expert_outputs(received_outputs)
        h = LayerNorm(h_mid + h_next)
```

关键属性：同步屏障是**全局的**——任一 worker 慢（straggler 或故障）都会阻塞整条 pipeline；层间完全串行——layer ℓ+1 必须等 layer ℓ 完成所有 token 的所有 expert。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 在 MegaScale-Infer 和 Tarragon 等解耦系统中，同步屏障由 AW-EW 间的请求-响应匹配隐式实现，不依赖显式的 NCCL barrier。
- Tarragon 的创新在于**部分打破**此同步屏障：EW 侧自愈允许 EW 在收到足够 AW 输入时即开始计算（不等所有 AW），将同步条件从 "all" 放松到 "enough"。
- 约束：batch 大小必须至少达到 GPU 效率拐点（NVIDIA A100 约 256-512 tokens），因此不能无限放松。
- 自适应：expert batch threshold 可根据 expert kernel 的 throughput-knee-point 动态配置。

涉及论文标题：
- Making MoE-based LLM Inference Resilient with Tarragon

---
