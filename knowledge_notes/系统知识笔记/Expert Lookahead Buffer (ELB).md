## Expert Lookahead Buffer (ELB)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Lookahead Buffer (ELB) 是 MoE-SpeQ 中 Expert Scheduler 的核心数据结构，用于捕获 speculative draft model 产生的多步 expert 需求预测。对 draft length=k 和 L 层 MoE layers，ELB 是一个 k×L 结构，每个条目 ELB[i][j] = (expert_id, confidence_score)。expert_id 来自量化 draft model 的 router top-k selection；confidence_score 来自 gating network logits（未量化，因 router 保持 FP16），度量 draft model 对该预测的确信度。ELB 在 CPU 端以 non-blocking 方式构建——每个 draft token 生成后，GPU 传回 router logits → CPU 解析并追加 ELB 条目，与 GPU 并行的 draft 计算不产生额外延迟。

从系统架构角度拆解术语：
```
Cycle Start:
    ELB = []  # k x L 空矩阵
    for i in 1..k:  # 草稿生成
        t_i = draft_model.forward(prev_token)
        logits_cpu = gpu_to_cpu_async(draft_router_logits)
        # GPU 同时继续下一 token draft
        
        for layer j in 1..L:
            expert_id = argmax(softmax(logits_cpu[layer_j]))
            score = max(softmax(logits_cpu[layer_j]))
            ELB[i][j] = (expert_id, score)
    
    # Phase I: Cache Priming — ELB 前部条目本地命中
    # Phase II: Adaptive Prefetch — ELB 中部高 confidence 条目预取
    # Phase III: Cache Saturation — ELB 尾部全部缺失 experts 预取

    # Verify: ELB 用于 computation reordering
    reordered = group_by(ELB_all_tokens, expert_id)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：CPU 端维护 ELB 数据结构（Python list of lists/numpy array），非阻塞 GPU→CPU 解析。confidence_score 当前以 expert_id 为主（论文指出 score-based risk-aware prefetching 为未来方向）。
- 关键设计要点：(1) 构建 non-blocking（与 draft GPU 计算并行）；(2) 必须在 verify 前完整（否则 cache miss 触发同步 I/O）；(3) 支持按 expert_id 查询所有 token 索引（用于 computation reordering）。

涉及论文标题：
- MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts
